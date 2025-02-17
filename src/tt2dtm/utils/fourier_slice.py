"""Useful functions for extracting and filtering Fourier slices."""

import roma
import torch
from torch_fourier_slice import extract_central_slices_rfft_3d
from torch_fourier_slice.slice_extraction import extract_central_slices_rfft_3d
from torch_fourier_slice.dft_utils import (
    fftshift_3d, ifftshift_2d, 
    rfft_shape, fftfreq_to_dft_coordinates
)
from torch_grid_utils import fftfreq_grid
from torch_fourier_slice.grids.central_slice_fftfreq_grid import ( 
    central_slice_fftfreq_grid
)


def _sinc2(shape: tuple[int, ...], rfft: bool, fftshift: bool) -> torch.Tensor:
    """Helper function for creating a sinc^2 filter."""
    grid = fftfreq_grid(
        image_shape=shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
    )

    return torch.sinc(grid) ** 2


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

    # Use roma to keep angles on same device
    rot_matrix = roma.euler_to_rotmat("zyz", (phi, theta, psi), degrees=degrees)

    # Use torch_fourier_slice to take the Fourier slice
    fourier_slices = extract_central_slices_rfft_3d(
        volume_rfft=volume_rfft,
        image_shape=shape,
        rotation_matrices=rot_matrix,
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

from typing import Literal

import einops
import torch.nn.functional as F








def sample_image_3d_bicubic_trilinear(
    image: torch.Tensor,
    coordinates: torch.Tensor
) -> torch.Tensor:
    """Sample a 3D image using bicubic interpolation in XY and trilinear in Z.

    Parameters
    ----------
    image: torch.Tensor
        `(d, h, w)` image.
    coordinates: torch.Tensor
        `(..., 3)` array of coordinates at which `image` should be sampled.
        - Coordinates are ordered `zyx` and are positions in the `d`, `h`, and `w` dimensions respectively.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.

    Returns
    -------
    samples: torch.Tensor
        `(..., )` array of samples from `image`.
    """
    device = coordinates.device

    if len(image.shape) != 3:
        raise ValueError(f'image should have shape (d, h, w), got {image.shape}')

    # Setup for sampling with torch.nn.functional.grid_sample
    complex_input = torch.is_complex(image)
    coordinates, ps = einops.pack([coordinates], pattern='* zyx')
    n_samples = coordinates.shape[0]

    if complex_input:
        # Handle complex tensors by separating real and imaginary parts
        image = torch.view_as_real(image)
        image = einops.rearrange(image, 'd h w complex -> complex d h w')
        image = einops.repeat(image, 'complex d h w -> b complex d h w', b=n_samples)
    else:
        image = einops.repeat(image, 'd h w -> b 1 d h w', b=n_samples)

    # Rearrange coordinates
    coordinates = einops.rearrange(coordinates, 'b zyx -> b 1 1 1 zyx')

    # Extract Z and XY coordinates separately
    z_coords = einops.rearrange(coordinates[..., 0], 'b 1 1 1 -> b 1 1')
    xy_coords = einops.rearrange(coordinates[..., 1:], 'b 1 1 1 xy -> b 1 xy')

    # Sample bicubic in XY (treating each depth slice separately)
    bicubic_samples = []
    for i in range(image.shape[2]):  # Loop over depth (Z-axis)
        xy_slice = image[:, :, i, :, :]  # Extract XY slice at depth i
        xy_grid = array_to_grid_sample(xy_coords, array_shape=(image.shape[-2], image.shape[-1]))
        xy_grid = einops.rearrange(xy_grid, 'b 1 xy -> b 1 1 xy')
        bicubic_sample = F.grid_sample(
            xy_slice, xy_grid, mode="bicubic", padding_mode="border", align_corners=True
        )
        bicubic_samples.append(bicubic_sample)

    bicubic_samples = torch.stack(bicubic_samples, dim=2)  # Stack along depth

    # Interpolate in Z using trilinear
    z_grid = array_to_grid_sample(z_coords, array_shape=(image.shape[-3],))
    z_grid = einops.repeat(z_grid, 'b 1 1 -> b 1 1 1 3')
    z_grid = z_grid.clone()  # ensure contiguous memory
    z_grid[..., 1:] = 0  # set y and x coordinates to 0
    final_samples = F.grid_sample(
        bicubic_samples, z_grid, 
        mode="bilinear", padding_mode="border", align_corners=True
    )

    if complex_input:
        final_samples = einops.rearrange(final_samples, 'b complex 1 1 1 -> b complex')
        final_samples = torch.view_as_complex(final_samples.contiguous())  # (b, )
    else:
        final_samples = einops.rearrange(final_samples, 'b 1 1 1 1 -> b')

    # Set out-of-bounds samples to zero
    coordinates = einops.rearrange(coordinates, 'b 1 1 1 zyx -> b zyx')
    volume_shape = torch.as_tensor(image.shape[-3:]).to(device)

    inside = torch.logical_and(coordinates >= 0, coordinates <= volume_shape - 1)
    inside = torch.all(inside, dim=-1)  # (b,)
    final_samples[~inside] *= 0

    # Pack samples back into the expected shape and return
    [final_samples] = einops.unpack(final_samples, pattern='*', packed_shapes=ps)
    return final_samples  # (...)

from typing import Sequence

def array_to_grid_sample(
    array_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate grids for `torch.nn.functional.grid_sample` from array coordinates.

    These coordinates should be used with `align_corners=True` in
    `torch.nn.functional.grid_sample`.


    Parameters
    ----------
    array_coordinates: torch.Tensor
        `(..., d)` array of d-dimensional coordinates.
        Coordinates are in the range `[0, N-1]` for the `N` elements in each dimension.
    array_shape: Sequence[int]
        shape of the array being sampled at `array_coordinates`.
    """
    dtype, device = array_coordinates.dtype, array_coordinates.device
    array_shape = torch.as_tensor(array_shape, dtype=dtype, device=device)
    grid_sample_coordinates = (array_coordinates / (0.5 * array_shape - 0.5)) - 1
    grid_sample_coordinates = torch.flip(grid_sample_coordinates, dims=(-1,))
    return grid_sample_coordinates


def extract_central_slices_rfft_3d_bicubic_trilinear(
    volume_rfft: torch.Tensor,
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
):
    """Extract central slice from an fftshifted rfft."""
    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = central_slice_fftfreq_grid(
        volume_shape=image_shape,
        rfft=True,
        fftshift=True,
        device=volume_rfft.device,
    )  # (h, w, 3) zyx coords

    # keep track of some shapes
    stack_shape = tuple(rotation_matrices.shape[:-2])
    rfft_shape = freq_grid.shape[-3], freq_grid.shape[-2]
    output_shape = (*stack_shape, *rfft_shape)

    # get (b, 3, 1) array of zyx coordinates to rotate

    valid_coords = einops.rearrange(freq_grid, 'h w zyx -> (h w) zyx 1')

    # rotation matrices rotate xyz coordinates, make them rotate zyx coordinates
    # xyz:
    # [a b c] [x]    [ax + by + cz]
    # [d e f] [y]  = [dx + ey + fz]
    # [g h i] [z]    [gx + hy + iz]
    #
    # zyx:
    # [i h g] [z]    [gx + hy + iz]
    # [f e d] [y]  = [dx + ey + fz]
    # [c b a] [x]    [ax + by + cz]
    rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # add extra dim to rotation matrices for broadcasting
    rotation_matrices = einops.rearrange(rotation_matrices, '... i j -> ... 1 i j')

    # rotate all valid coordinates by each rotation matrix
    rotated_coords = rotation_matrices @ valid_coords  # (..., b, zyx, 1)

    # remove last dim of size 1
    rotated_coords = einops.rearrange(rotated_coords, '... b zyx 1 -> ... b zyx')

    # flip coordinates that ended up in redundant half transform after rotation
    conjugate_mask = rotated_coords[..., 2] < 0
    rotated_coords[conjugate_mask, ...] *= -1

    # convert frequencies to array coordinates in fftshifted DFT
    rotated_coords = fftfreq_to_dft_coordinates(
        frequencies=rotated_coords,
        image_shape=image_shape,
        rfft=True
    )
    samples = sample_image_3d_bicubic_trilinear(image=volume_rfft, coordinates=rotated_coords)  # (...) rfft

    # take complex conjugate of values from redundant half transform
    samples[conjugate_mask] = torch.conj(samples[conjugate_mask])

    # insert samples back into DFTs
    projection_image_dfts = torch.zeros(output_shape, device=volume_rfft.device, dtype=volume_rfft.dtype)
    freq_grid_mask = torch.ones(size=rfft_shape, dtype=torch.bool, device=volume_rfft.device)
    projection_image_dfts[..., freq_grid_mask] = samples

    return projection_image_dfts

# TODO: Helper functions for applying filters to Fourier slices










def sample_image_3d_tricubic_approx(image: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Approximate tricubic interpolation using multiple trilinear passes.

    Parameters
    ----------
    image: torch.Tensor
        `(d, h, w)` image volume.
    coordinates: torch.Tensor
        `(..., 3)` array of coordinates at which `image` should be sampled.
        - Ordered as `zyx`, representing `(d, h, w)`.
        - Coordinates range from `[0, N-1]`.

    Returns
    -------
    samples: torch.Tensor
        `(..., )` array of sampled values.
    """
    device = coordinates.device

    if len(image.shape) != 3:
        raise ValueError(f'Expected image shape (d, h, w), got {image.shape}')

    # Prepare sampling (packing shapes)
    complex_input = torch.is_complex(image)
    coordinates, ps = einops.pack([coordinates], pattern='* zyx')
    n_samples = coordinates.shape[0]

    # Expand image for batch processing
    if complex_input is True:
        # cannot sample complex tensors directly with grid_sample
        # c.f. https://github.com/pytorch/pytorch/issues/67634
        # workaround: treat real and imaginary parts as separate channels
        image = torch.view_as_real(image)
        image = einops.rearrange(image, 'd h w complex -> complex d h w')
        image = einops.repeat(image, 'complex d h w -> b complex d h w', b=n_samples)
    else:
        image = einops.repeat(image, 'd h w -> b 1 d h w', b=n_samples)  # b c d h w
    coordinates = einops.rearrange(coordinates, 'b zyx -> b 1 1 1 zyx')  # Reshape for grid_sample

    # First pass: Trilinear interpolation along X-axis
    grid_x = coordinates.clone()
    grid_x[..., 1:] = torch.round(grid_x[..., 1:])  # Keep Y, Z fixed
    interp_x = F.grid_sample(image, array_to_grid_sample(grid_x, image.shape[-3:]), mode='bilinear', align_corners=True)

    # Second pass: Trilinear interpolation along Y-axis
    grid_y = coordinates.clone()
    grid_y[..., 0] = torch.round(grid_y[..., 0])  # Keep X, Z fixed
    interp_y = F.grid_sample(interp_x, array_to_grid_sample(grid_y, image.shape[-3:]), mode='bilinear', align_corners=True)

    # Third pass: Trilinear interpolation along Z-axis
    grid_z = coordinates.clone()
    grid_z[..., :2] = torch.round(grid_z[..., :2])  # Keep X, Y fixed
    interp_z = F.grid_sample(interp_y, array_to_grid_sample(grid_z, image.shape[-3:]), mode='bilinear', align_corners=True)

    if complex_input is True:
        samples = einops.rearrange(interp_z, 'b complex 1 1 1 -> b complex')
        samples = torch.view_as_complex(samples.contiguous())  # (b, )
    else:
        samples = einops.rearrange(samples, 'b 1 1 1 1 -> b')

    # set samples from outside of volume to zero
    coordinates = einops.rearrange(coordinates, 'b 1 1 1 zyx -> b zyx')
    volume_shape = torch.as_tensor(image.shape[-3:]).to(device)

    inside = torch.logical_and(coordinates >= 0, coordinates <= volume_shape - 1)
    inside = torch.all(inside, dim=-1)  # (b, d, h, w)
    samples[~inside] *= 0

    # pack samples back into the expected shape and return
    [samples] = einops.unpack(samples, pattern='*', packed_shapes=ps)
    return samples  # (...)

def extract_central_slices_rfft_3d_tricubic_approx(
    volume_rfft: torch.Tensor,
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
):
    """Extract central slice from an fftshifted rfft."""
    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = central_slice_fftfreq_grid(
        volume_shape=image_shape,
        rfft=True,
        fftshift=True,
        device=volume_rfft.device,
    )  # (h, w, 3) zyx coords

    # keep track of some shapes
    stack_shape = tuple(rotation_matrices.shape[:-2])
    rfft_shape = freq_grid.shape[-3], freq_grid.shape[-2]
    output_shape = (*stack_shape, *rfft_shape)

    # get (b, 3, 1) array of zyx coordinates to rotate

    valid_coords = einops.rearrange(freq_grid, 'h w zyx -> (h w) zyx')
    valid_coords = einops.rearrange(valid_coords, 'b zyx -> b zyx 1')

    # rotation matrices rotate xyz coordinates, make them rotate zyx coordinates
    # xyz:
    # [a b c] [x]    [ax + by + cz]
    # [d e f] [y]  = [dx + ey + fz]
    # [g h i] [z]    [gx + hy + iz]
    #
    # zyx:
    # [i h g] [z]    [gx + hy + iz]
    # [f e d] [y]  = [dx + ey + fz]
    # [c b a] [x]    [ax + by + cz]
    rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # add extra dim to rotation matrices for broadcasting
    rotation_matrices = einops.rearrange(rotation_matrices, '... i j -> ... 1 i j')

    # rotate all valid coordinates by each rotation matrix
    rotated_coords = rotation_matrices @ valid_coords  # (..., b, zyx, 1)

    # remove last dim of size 1
    rotated_coords = einops.rearrange(rotated_coords, '... b zyx 1 -> ... b zyx')

    # flip coordinates that ended up in redundant half transform after rotation
    conjugate_mask = rotated_coords[..., 2] < 0
    rotated_coords[conjugate_mask, ...] *= -1

    # convert frequencies to array coordinates in fftshifted DFT
    rotated_coords = fftfreq_to_dft_coordinates(
        frequencies=rotated_coords,
        image_shape=image_shape,
        rfft=True
    )
    samples = sample_image_3d_tricubic_approx(image=volume_rfft, coordinates=rotated_coords)  # (...) rfft

    # take complex conjugate of values from redundant half transform
    samples[conjugate_mask] = torch.conj(samples[conjugate_mask])

    # insert samples back into DFTs
    projection_image_dfts = torch.zeros(output_shape, device=volume_rfft.device, dtype=volume_rfft.dtype)
    freq_grid_mask = torch.ones(size=rfft_shape, dtype=torch.bool, device=volume_rfft.device)
    projection_image_dfts[..., freq_grid_mask] = samples

    return projection_image_dfts








def _windowed_sinc(x: torch.Tensor, window_radius: int = 4) -> torch.Tensor:
    """Compute windowed sinc function.
    
    Uses Lanczos window (sinc itself as the window function).
    
    Parameters
    ----------
    x : torch.Tensor
        Input values to compute sinc for
    window_radius : int
        Radius of the window in pixels. Total kernel width will be 2*radius + 1
    """
    # Avoid division by zero
    x = x.clone()
    zero_mask = x == 0
    x[zero_mask] = 1.0
    
    # Compute sinc
    sinc = torch.sin(torch.pi * x) / (torch.pi * x)
    sinc[zero_mask] = 1.0
    
    # Apply Lanczos window
    window = torch.where(
        torch.abs(x) < window_radius,
        torch.sin(torch.pi * x / window_radius) / (torch.pi * x / window_radius),
        torch.zeros_like(x)
    )
    window[zero_mask] = 1.0
    
    return sinc * window

def sample_image_3d_sinc(
    image: torch.Tensor,
    coordinates: torch.Tensor,
    window_radius: int = 4,
) -> torch.Tensor:
    """Sample a 3D image using sinc interpolation.
    
    Parameters
    ----------
    image : torch.Tensor
        Input 3D image of shape (d, h, w)
    coordinates : torch.Tensor
        Coordinates to sample at, shape (..., 3) in zyx order
    window_radius : int
        Radius of the sinc window in pixels. Larger values give better
        frequency response but are slower and may introduce ringing.
    
    Returns
    -------
    torch.Tensor
        Interpolated values at the requested coordinates
    """
    device = coordinates.device
    dtype = image.dtype
    
    # Get integer coordinates and offsets
    coords_floor = torch.floor(coordinates).long()
    coords_frac = coordinates - coords_floor
    
    # Create sampling grid centered on each point
    grid_offsets = torch.arange(-window_radius, window_radius + 1, device=device)
    gz, gy, gx = torch.meshgrid(grid_offsets, grid_offsets, grid_offsets, indexing='ij')
    
    # Calculate distances for sinc
    dx = gx.unsqueeze(0) - coords_frac[..., 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    dy = gy.unsqueeze(0) - coords_frac[..., 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    dz = gz.unsqueeze(0) - coords_frac[..., 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    # Compute sinc weights
    weights_x = _windowed_sinc(dx)
    weights_y = _windowed_sinc(dy)
    weights_z = _windowed_sinc(dz)
    weights = weights_x * weights_y * weights_z
    
    # Normalize weights
    weights = weights / weights.sum(dim=(-3, -2, -1), keepdim=True)
    
    # Get sampling coordinates
    z_coords = coords_floor[..., 0, None, None, None] + gz
    y_coords = coords_floor[..., 1, None, None, None] + gy
    x_coords = coords_floor[..., 2, None, None, None] + gx
    
    # Clamp coordinates to valid image bounds
    z_coords = z_coords.clamp(0, image.shape[0] - 1)
    y_coords = y_coords.clamp(0, image.shape[1] - 1)
    x_coords = x_coords.clamp(0, image.shape[2] - 1)
    
    # Sample and weight
    if torch.is_complex(image):
        real_samples = image.real[z_coords, y_coords, x_coords]
        imag_samples = image.imag[z_coords, y_coords, x_coords]
        
        real_result = (real_samples * weights).sum(dim=(-3, -2, -1))
        imag_result = (imag_samples * weights).sum(dim=(-3, -2, -1))
        
        return torch.complex(real_result, imag_result)
    else:
        samples = image[z_coords, y_coords, x_coords]
        return (samples * weights).sum(dim=(-3, -2, -1))
    
def extract_central_slices_rfft_3d_sinc(
    volume_rfft: torch.Tensor,
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
):
    """Extract central slice from an fftshifted rfft using sinc interpolation."""
    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = central_slice_fftfreq_grid(
        volume_shape=image_shape,
        rfft=True,
        fftshift=True,
        device=volume_rfft.device,
    )  # (h, w, 3) zyx coords

    # keep track of some shapes
    stack_shape = tuple(rotation_matrices.shape[:-2])
    rfft_shape = freq_grid.shape[-3], freq_grid.shape[-2]
    output_shape = (*stack_shape, *rfft_shape)

    # get (b, 3, 1) array of zyx coordinates to rotate

    valid_coords = einops.rearrange(freq_grid, 'h w zyx -> (h w) zyx 1')

    # rotation matrices rotate xyz coordinates, make them rotate zyx coordinates
    # xyz:
    # [a b c] [x]    [ax + by + cz]
    # [d e f] [y]  = [dx + ey + fz]
    # [g h i] [z]    [gx + hy + iz]
    #
    # zyx:
    # [i h g] [z]    [gx + hy + iz]
    # [f e d] [y]  = [dx + ey + fz]
    # [c b a] [x]    [ax + by + cz]
    rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # add extra dim to rotation matrices for broadcasting
    rotation_matrices = einops.rearrange(rotation_matrices, '... i j -> ... 1 i j')

    # rotate all valid coordinates by each rotation matrix
    rotated_coords = rotation_matrices @ valid_coords  # (..., b, zyx, 1)

    # remove last dim of size 1
    rotated_coords = einops.rearrange(rotated_coords, '... b zyx 1 -> ... b zyx')

    # flip coordinates that ended up in redundant half transform after rotation
    conjugate_mask = rotated_coords[..., 2] < 0
    rotated_coords[conjugate_mask, ...] *= -1

    # convert frequencies to array coordinates in fftshifted DFT
    rotated_coords = fftfreq_to_dft_coordinates(
        frequencies=rotated_coords,
        image_shape=image_shape,
        rfft=True
    )
    samples = sample_image_3d_sinc(
        image=volume_rfft, 
        coordinates=rotated_coords,
        window_radius=4  # Adjust based on accuracy vs. speed needs
    )

    # take complex conjugate of values from redundant half transform
    samples[conjugate_mask] = torch.conj(samples[conjugate_mask])

    # insert samples back into DFTs
    projection_image_dfts = torch.zeros(output_shape, device=volume_rfft.device, dtype=volume_rfft.dtype)
    freq_grid_mask = torch.ones(size=rfft_shape, dtype=torch.bool, device=volume_rfft.device)
    projection_image_dfts[..., freq_grid_mask] = samples

    return projection_image_dfts