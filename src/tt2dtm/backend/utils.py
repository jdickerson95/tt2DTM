"""Utility functions associated with backend functions."""

from typing import Literal

import roma
import torch
from torch_fourier_slice import extract_central_slices_rfft_3d

from tt2dtm.utils.cross_correlation import handle_correlation_mode


def normalize_template_projection(
    projections: torch.Tensor,  # shape (batch, h, w)
    small_shape: tuple[int, int],  # (h, w)
    large_shape: tuple[int, int],  # (H, W)
) -> torch.Tensor:
    """Subtract mean of edge values and set variance to 1 (in large shape).

    This function uses the fact that variance of a sequence, Var(X), is scaled by the
    relative size of the small (unpadded) and large (padded with zeros) space. Some
    negligible error is introduced into the variance (~1e-4) due to this routine.

    Parameters
    ----------
    projections : torch.Tensor
        Real-space projections of the template (in small space).
    small_shape : tuple[int, int]
        Shape of the template.
    large_shape : tuple[int, int]
        Shape of the image (in large space).

    Returns
    -------
    torch.Tensor
        Edge-mean subtracted projections, still in small space, but normalized
        so variance of zero-padded projection is 1.
    """
    # Constants related to scaling the variance
    npix_padded = large_shape[0] * large_shape[1] - small_shape[0] * small_shape[1]
    relative_size = small_shape[0] * small_shape[1] / (large_shape[0] * large_shape[1])

    # Extract edges while preserving batch dimensions
    top_edge = projections[..., 0, :]  # shape: (..., W)
    bottom_edge = projections[..., -1, :]  # shape: (..., W)
    left_edge = projections[..., 1:-1, 0]  # shape: (..., H-2)
    right_edge = projections[..., 1:-1, -1]  # shape: (..., H-2)
    edge_pixels = torch.concatenate(
        [top_edge, bottom_edge, left_edge, right_edge], dim=-1
    )

    # Subtract the edge pixel mean and calculate variance of small, unpadded projection
    edge_mean = edge_pixels.mean(dim=-1)
    projections -= edge_mean[..., None, None]

    # # Calculate variance like cisTEM (does not match desired results...)
    # variance = (projections**2).sum(dim=(-1, -2), keepdim=True) * relative_size - (
    #     projections.mean(dim=(-1, -2), keepdim=True) * relative_size
    # ) ** 2

    # Fast calculation of mean/var using Torch + appropriate scaling.
    # Scale the variance such that the larger padded space has variance of 1.
    variance, mean = torch.var_mean(projections, dim=(-1, -2), keepdim=True)
    mean += relative_size
    variance *= relative_size
    variance += (1 / npix_padded) * mean**2

    return projections / torch.sqrt(variance)


def do_iteration_statistics_updates(
    cross_correlation: torch.Tensor,
    euler_angles: torch.Tensor,
    defocus_values: torch.Tensor,
    mip: torch.Tensor,
    best_phi: torch.Tensor,
    best_theta: torch.Tensor,
    best_psi: torch.Tensor,
    best_defocus: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    H: int,
    W: int,
) -> None:
    """Helper function for updating maxima and tracked statistics.

    NOTE: The batch dimensions are effectively unraveled since taking the
    maximum over a single batch dimensions is much faster than
    multi-dimensional maxima.

    NOTE: Updating the maxima was found to be fastest and least memory
    impactful when using torch.where directly. Other methods tested were
    boolean masking and torch.where with tuples of tensor indexes.

    Parameters
    ----------
    cross_correlation : torch.Tensor
        Cross-correlation values for the current iteration. Has either shape
        (batch, H, W) or (defocus, orientations, H, W).
    euler_angles : torch.Tensor
        Euler angles for the current iteration. Has shape (orientations, 3).
    defocus_values : torch.Tensor
        Defocus values for the current iteration. Has shape (defocus,).
    mip : torch.Tensor
        Maximum intensity projection of the cross-correlation values.
    best_phi : torch.Tensor
        Best phi angle for each pixel.
    best_theta : torch.Tensor
        Best theta angle for each pixel.
    best_psi : torch.Tensor
        Best psi angle for each pixel.
    best_defocus : torch.Tensor
        Best defocus value for each pixel.
    correlation_sum : torch.Tensor
        Sum of cross-correlation values for each pixel.
    correlation_squared_sum : torch.Tensor
        Sum of squared cross-correlation values for each pixel.
    H : int
        Height of the cross-correlation values.
    W : int
        Width of the cross-correlation values.
    """
    max_values, max_indices = torch.max(cross_correlation.view(-1, H, W), dim=0)
    max_defocus_idx = max_indices // euler_angles.shape[0]
    max_orientation_idx = max_indices % euler_angles.shape[0]

    # using torch.where directly
    update_mask = max_values > mip

    torch.where(update_mask, max_values, mip, out=mip)
    torch.where(
        update_mask, euler_angles[max_orientation_idx, 0], best_phi, out=best_phi
    )
    torch.where(
        update_mask, euler_angles[max_orientation_idx, 1], best_theta, out=best_theta
    )
    torch.where(
        update_mask, euler_angles[max_orientation_idx, 2], best_psi, out=best_psi
    )
    torch.where(
        update_mask, defocus_values[max_defocus_idx], best_defocus, out=best_defocus
    )

    correlation_sum += cross_correlation.view(-1, H, W).sum(dim=0)
    correlation_squared_sum += (cross_correlation.view(-1, H, W) ** 2).sum(dim=0)


def cross_correlate_particle_stack(
    particle_stack_dft: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    euler_angles: torch.Tensor,  # (3, N)
    projective_filters: torch.Tensor,  # (N, h, w)
    mode: Literal["valid", "same"] = "valid",
    batch_size: int = 1024,
) -> torch.Tensor:
    """Cross-correlate a stack of particle images against a template.

    Here, the argument 'particle_stack_dft' is a set of RFFT-ed particle images with
    necessary filtering already applied. The zeroth dimension corresponds to unique
    particles.

    Parameters
    ----------
    particle_stack_dft : torch.Tensor
        The stack of particle real-Fourier transformed and un-fftshifted images.
        Shape of (N, H, W).
    template_dft : torch.Tensor
        The template volume to extract central slices from. Real-Fourier transformed
        and fftshifted.
    euler_angles : torch.Tensor
        The Euler angles for each particle in the stack. Shape of (3, N).
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    mode : Literal["valid", "same"], optional
        Correlation mode to use, by default "valid". If "valid", the output will be
        the valid cross-correlation of the inputs. If "same", the output will be the
        same shape as the input particle stack.
    batch_size : int, optional
        The number of particle images to cross-correlate at once. Default is 1024.
        Larger sizes will consume more memory. If -1, then the entire stack will be
        cross-correlated at once.

    Returns
    -------
    torch.Tensor
        The cross-correlation of the particle stack with the template. Shape will depend
        on the mode used. If "valid", the output will be (N, H-h+1, W-w+1). If "same",
        the output will be (N, H, W).
    """
    # Helpful constants for later use
    device = particle_stack_dft.device
    num_particles, H, W = particle_stack_dft.shape
    d, h, w = template_dft.shape
    # account for RFFT
    W = 2 * (W - 1)
    w = 2 * (w - 1)

    if batch_size == -1:
        batch_size = num_particles

    if mode == "valid":
        output_shape = (num_particles, H - h + 1, W - w + 1)
    elif mode == "same":
        output_shape = (num_particles, H, W)

    out_correlation = torch.zeros(output_shape, device=device)

    # Loop over the particle stack in batches
    for i in range(0, num_particles, batch_size):
        batch_particles_dft = particle_stack_dft[i : i + batch_size]
        batch_euler_angles = euler_angles[i : i + batch_size]
        batch_projective_filters = projective_filters[i : i + batch_size]

        # Convert the Euler angles into rotation matrices
        rot_matrix = roma.euler_to_rotmat(
            "ZYZ", batch_euler_angles, degrees=True, device=device
        )

        # Extract the Fourier slice and apply the projective filters
        fourier_slice = extract_central_slices_rfft_3d(
            volume_rfft=template_dft,
            image_shape=(h,) * 3,
            rotation_matrices=rot_matrix,
        )
        fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
        fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
        fourier_slice *= -1  # flip contrast

        fourier_slice *= batch_projective_filters

        # Inverse Fourier transform and normalize the projection
        projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
        projections = torch.fft.ifftshift(projections, dim=(-2, -1))
        projections = normalize_template_projection(projections, (h, w), (H, W))

        # Padded forward FFT and cross-correlate
        projections_dft = torch.fft.rfftn(projections, dim=(-2, -1), s=(H, W))
        projections_dft = batch_particles_dft * projections_dft.conj()
        cross_correlation = torch.fft.irfftn(projections_dft, dim=(-2, -1))

        # Handle the output shape
        cross_correlation = handle_correlation_mode(
            cross_correlation, output_shape, mode
        )

        out_correlation[i : i + batch_size] = cross_correlation

    return out_correlation
