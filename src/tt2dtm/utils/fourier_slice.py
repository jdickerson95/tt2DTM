"""Useful functions for extracting and filtering Fourier slices."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_fourier_slice import extract_central_slices_rfft_3d
from torch_fourier_slice.dft_utils import fftshift_3d, ifftshift_2d
from torch_grid_utils import fftfreq_grid


def _sinc2(shape: tuple[int, ...], rfft: bool, fftshift: bool) -> torch.Tensor:
    """Helper function for creating a sinc^2 filter."""
    grid = fftfreq_grid(
        image_shape=shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
    )

    return torch.sinc(grid) ** 2


# tidy up these next two functions, get someting going for now
def fft_volume(
    volume: torch.Tensor,
    fftshift: bool = True,
) -> torch.Tensor:
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))  # volume center to array origin
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    if fftshift:
        dft = torch.fft.fftshift(
            dft,
            dim=(
                -3,
                -2,
            ),
        )  # actual fftshift of 3D rfft
    return dft


def extract_fourier_slice(
    dft_volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    volume_shape: torch.Tensor,
) -> torch.Tensor:

    # print device of all three tensors
    #print(f"dft_volume device: {dft_volume.device}")
    #print(f"rotation_matrices device: {rotation_matrices.device}")
    #print(f"volume_shape device: {volume_shape.device}")

    # make projections by taking central slices
    projections = extract_central_slices_rfft_3d(
        volume_rfft=dft_volume,
        image_shape=volume_shape,
        rotation_matrices=rotation_matrices,
    )  # (..., h, w) rfft stack

    # FFT shift to original because ctf can only be applied to rfft or fftshift
    projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of 2D rfft
    return projections


def _rfft_slices_to_real_projections(
    fourier_slices: torch.Tensor,
) -> torch.Tensor:
    """Convert Fourier slices to real-space projections.

    Parameters
    ----------
    fourier_slices : torch.Tensor
        The Fourier slices to convert. Inverse Fourier transform is applied
        across the last two dimensions.


    Returns
    -------
    torch.Tensor
        The real-space projections.
    """
    fourier_slices = ifftshift_2d(fourier_slices, rfft=True)
    projections = torch.fft.irfftn(fourier_slices, dim=(-2, -1))
    projections = ifftshift_2d(projections, rfft=False)

    return projections


def get_rfft_slices_from_volume(
    volume: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    degrees: bool = True,
) -> torch.Tensor:
    """Helper function to get Fourier slices of a real-space volume.

    Parameters
    ----------
    volume : torch.Tensor
        The 3D volume to get Fourier slices from.
    phi : torch.Tensor
        The phi Euler angle.
    theta : torch.Tensor
        The theta Euler angle.
    psi : torch.Tensor
        The psi Euler angle.
    degrees : bool
        True if Euler angles are in degrees, False if in radians.

    Returns
    -------
    torch.Tensor
        The Fourier slices of the volume.

    """
    shape = volume.shape
    volume_rfft = fftshift_3d(volume, rfft=False)
    volume_rfft = torch.fft.fftn(volume_rfft, dim=(-3, -2, -1))
    volume_rfft = fftshift_3d(volume_rfft, rfft=True)

    # TODO: Keep all rotation computation in PyTorch
    psi_np = psi.cpu().numpy()
    theta_np = theta.cpu().numpy()
    phi_np = phi.cpu().numpy()

    angles = np.stack([phi_np, theta_np, psi_np], axis=-1)

    rot_np = Rotation.from_euler("zyz", angles, degrees=degrees)
    rot = torch.Tensor(rot_np.as_matrix())

    # Use torch_fourier_slice to take the Fourier slice
    fourier_slices = extract_central_slices_rfft_3d(
        volume_rfft=volume_rfft,
        image_shape=shape,
        rotation_matrices=rot,
    )

    # Invert contrast to match image
    fourier_slices = -fourier_slices

    return fourier_slices


def get_real_space_projections_from_volume(
    volume: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    degrees: bool = True,
) -> torch.Tensor:
    """Real-space projections of a 3D volume.

    Note that Euler angles are in 'zyz' convention.

    Parameters
    ----------
    volume : torch.Tensor
        The 3D volume to get projections from.
    phi : torch.Tensor
        The phi Euler angle.
    theta : torch.Tensor
        The theta Euler angle.
    psi : torch.Tensor
        The psi Euler angle.
    degrees : bool
        True if Euler angles are in degrees, False if in radians.

    Returns
    -------
    torch.Tensor
        The real-space projections.
    """
    fourier_slices = get_rfft_slices_from_volume(
        volume=volume,
        phi=phi,
        theta=theta,
        psi=psi,
        degrees=degrees,
    )
    projections = _rfft_slices_to_real_projections(fourier_slices)

    return projections


# TODO: Helper functions for applying filters to Fourier slices
