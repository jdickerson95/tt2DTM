"""Core cross-correlation methods for single and stacks of image/templates."""

from typing import Literal

import torch
import einops
from torch_fourier_slice.dft_utils import (
    fftshift_2d,
)

# Functions to apply mode

# TODO: Normalized cross correlation


def handle_correlation_mode(
    cross_correlation: torch.Tensor,
    out_shape: tuple[int, ...],
    mode: Literal["valid", "full"],
) -> torch.Tensor:
    """Handle cropping for cross correlation mode."""
    # Crop the result to the valid bounds
    if mode == "valid":
        slices = [slice(0, os) for os in out_shape]
        cross_correlation = cross_correlation[slices]
    elif mode == "full":
        pass
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return cross_correlation


def _handle_template_padding_dft(
    template_dft: torch.Tensor,
    image_shape: tuple[int, ...],
    dim: tuple[int, ...],
    rfft: bool,
    fftshift: bool,
) -> torch.Tensor:
    """Helper function for padding an unequal template DFT up to image size.

    Parameters
    ----------
    template_dft : torch.Tensor
        The template DFT to pad.
    image_shape : tuple[int, ...]
        The shape of the image.
    dim : tuple[int, ...]
        The dimensions to pad along using the Fourier transform.
    rfft : bool
        True if the DFT represents the rfft of the template.
    fftshift : bool
        True if the DFT was fftshifted.

    Returns
    -------
    torch.Tensor
        The padded template DFT.

    """
    # Convert negative dimensions to corresponding positive dimensions
    dim = tuple([d % len(image_shape) for d in dim])

    if fftshift:
        # TODO: Handle case where dim is not the last two dimensions
        template_dft = fftshift_2d(template_dft, rfft=rfft)

    # Fourier transform with padding up to image shape
    if rfft:
        template = torch.fft.irfftn(template_dft)
        template_dft = torch.fft.rfftn(template, s=image_shape, dim=dim)
    else:
        template = torch.fft.ifftn(template_dft)
        template_dft = torch.fft.fftn(template, s=image_shape, dim=dim)

    return template_dft


def cross_correlate(
    image: torch.Tensor,
    template: torch.Tensor,
    mode: Literal["valid", "full"] = "valid",
) -> torch.Tensor:
    """Cross correlate an real-space image and real-space template.

    Parameters
    ----------
    image : torch.Tensor
        The real-space image.
    template : torch.Tensor
        The real-space template, presumed to be the same size or smaller than
        the image.
    mode : Literal["valid", "full"], optional
        The mode of the cross correlation. Either 'valid' or 'full'. See
        [numpy.correlate](https://numpy.org/doc/2.1/reference/generated/numpy.convolve.html#numpy.convolve)
        for more details.

    Returns
    -------
    torch.Tensor
        The cross correlation of the image and template.
    """
    return batched_cross_correlate(
        image=image,
        template=template,
        dim=tuple(range(image.ndim)),
        mode=mode,
    )


def cross_correlate_dft(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    orig_image_shape: tuple[int, ...],
    orig_template_shape: tuple[int, ...],
    rfft: bool,
    fftshift: bool,
    mode: Literal["valid", "full"] = "valid",
) -> torch.Tensor:
    """Cross correlate an Fourier space image and Fourier space template.

    Note that this function performs the cross correlation along all dimensions
    of the image and template. Use `batched_cross_correlate_dft` for batched
    cross correlation along specific dimensions.

    Parameters
    ----------
    image_dft : torch.Tensor
        Discrete Fourier transform of the image.
    template_dft : torch.Tensor
        Discrete Fourier transform of the template.
    orig_image_shape : tuple[int, ...]
        The original shape of the real-space image.
    orig_template_shape : tuple[int, ...]
        The original shape of the real-space template.
    rfft : bool
        True if the dft represents the rfft of the image/template
        (torch.fft.rfftn). Applies to both the image and template.
    fftshift : bool
        True if the dft was fftshifted. Applies to both the image and template.
    mode : Literal["valid", "full"], optional
        The mode of the cross correlation. Either 'valid' or 'full'. See
        [numpy.correlate](https://numpy.org/doc/2.1/reference/generated/numpy.convolve.html#numpy.convolve)
        for more details.

    Returns
    -------
    torch.Tensor
        The cross correlation of the image and template.
    """
    return batched_cross_correlate_dft(
        image_dft=image_dft,
        template_dft=template_dft,
        orig_image_shape=image_dft.shape,
        orig_template_shape=template_dft.shape,
        rfft=rfft,
        fftshift=fftshift,
        dim=tuple(range(image_dft.ndim)),
        mode=mode,
    )


def batched_cross_correlate(
    image: torch.Tensor,
    template: torch.Tensor,
    dim: tuple[int, ...] = (-2, -1),
    mode: Literal["valid", "full"] = "valid",
) -> torch.Tensor:
    """Perform batched cross correlation along specific dimensions.

    The image and template must have the same number of dimensions, and the
    non cross-correlated dimensions of the template must be the same size as
    the image or broadcastable to the image shape.

    Parameters
    ----------
    image : torch.Tensor
        The real-space image.
    template : torch.Tensor
        The real-space template.
    dim : tuple[int, ...], optional
        Which dimensions the cross correlation should be performed along.
        Default is the last two dimensions.
    mode : Literal["valid", "full"], optional
        The mode of the cross correlation. Either 'valid' or 'full'. See
        [numpy.correlate](https://numpy.org/doc/2.1/reference/generated/numpy.convolve.html#numpy.convolve)
        for more details.

    Returns
    -------
    torch.Tensor
        The cross correlation of the image and template.
    """
    orig_image_shape = image.shape
    orig_template_shape = template.shape

    # TODO: Handle complex input (no rfft)
    image_rfft = torch.fft.rfftn(image, dim=dim)
    template_rfft = torch.fft.rfftn(template, s=orig_image_shape, dim=dim)

    cross_correlation = batched_cross_correlate_dft(
        image_dft=image_rfft,
        template_dft=template_rfft,
        orig_image_shape=orig_image_shape,
        orig_template_shape=orig_template_shape,
        rfft=True,
        fftshift=False,
        mode=mode,
        dim=dim,
    )

    return cross_correlation


def batched_cross_correlate_dft(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    orig_image_shape: tuple[int, ...],
    orig_template_shape: tuple[int, ...],
    rfft: bool,
    fftshift: bool,
    dim: tuple[int, ...] = (-2, -1),
    mode: Literal["valid", "full"] = "valid",
) -> torch.Tensor:
    """Batched cross correlation along specific dimensions for DFT inputs.

    The image and template must have the same number of dimensions, and the
    non cross-correlated dimensions of the template must be the same size as
    the image or broadcastable to the image shape.

    Parameters
    ----------
    image_dft : torch.Tensor
        Discrete Fourier transform of the image along the specified dimensions.
    template_dft : torch.Tensor
        Discrete Fourier transform of the template along the specified
        dimensions.
    orig_image_shape : tuple[int, ...]
        The original shape of the real-space image.
    orig_template_shape : tuple[int, ...]
        The original shape of the real-space template.
    rfft : bool
        True if the dft represents the rfft of the image/template
        (torch.fft.rfftn). Applies to both the image and template.
    fftshift : bool
        True if the dft was fftshifted. Applies to both the image and template.
    dim : tuple[int, ...], optional
        Which dimensions the cross correlation should be performed along.
        Default is the last two dimensions.
    mode : Literal["valid", "full"], optional
        The mode of the cross correlation. Either 'valid' or 'full'. See
        [numpy.correlate](https://numpy.org/doc/2.1/reference/generated/numpy.convolve.html#numpy.convolve)
        for more details.

    Returns
    -------
    torch.Tensor
        The cross correlation of the image and template.

    """
    ##################################
    ### Validate the method inputs ###
    ##################################
    if image_dft.ndim != template_dft.ndim:
        raise ValueError("Image and template must have the same number of dimensions.")

    dim_positive = tuple([d % len(orig_image_shape) for d in dim])

    assert torch.all(
        torch.isin(torch.tensor(dim), torch.arange(-image_dft.ndim, image_dft.ndim))
    )
    assert len(dim) == len(set(dim_positive)), "Found duplicate dimensions"

    dim = dim_positive

    ###############################################
    ### Do the cross correlation handling shape ###
    ###############################################
    image_shape = tuple([s for i, s in enumerate(orig_image_shape) if i in dim])
    template_shape = tuple([s for i, s in enumerate(orig_template_shape) if i in dim])
    out_shape = tuple(
        [
            (
                orig_image_shape[i] - orig_template_shape[i] + 1
                if i in dim
                else orig_image_shape[i]
            )
            for i in range(image_dft.ndim)
        ]
    )

    # Handle template that is smaller than the image
    if torch.any(torch.tensor(template_shape) < torch.tensor(image_shape)):
        template_dft = _handle_template_padding_dft(
            template_dft=template_dft,
            image_shape=image_shape,
            dim=dim,
            rfft=rfft,
            fftshift=fftshift,
        )

    # Perform the cross correlation
    cross_correlation = image_dft * template_dft.conj()

    if fftshift:
        # TODO: Handle case where dim is not the last two dimensions
        cross_correlation = fftshift_2d(cross_correlation, rfft=rfft)

    if rfft:
        cross_correlation = torch.fft.irfftn(cross_correlation, dim=dim)
    else:
        cross_correlation = torch.fft.ifftn(cross_correlation, dim=dim)

    # Handle the mode
    cross_correlation = handle_correlation_mode(
        cross_correlation=cross_correlation,
        out_shape=out_shape,
        mode=mode,
    )

    return cross_correlation

def simple_cross_correlation(
    projections: torch.Tensor,
    dft_micrographs_filtered: torch.Tensor,
) -> torch.Tensor:
    """
    Simple cross correlation.

    Parameters
    ----------
    projections : torch.Tensor
        The projections to cross correlate.
    dft_micrographs_filtered : torch.Tensor
        The DFT of the micrographs filtered by the template.

    Returns
    -------
    torch.Tensor
        The cross correlated projections.
    """
    # FFT shift to edge
    projections = torch.fft.fftshift(projections, dim=(-2, -1))
    # do FFT and 0 central pixel
    projections = torch.fft.rfftn(projections, dim=(-2, -1))
    # zero central pixel
    projections[:, 0, 0] = 0 + 0j
    # cross correlations
    dft_micrographs_filtered = einops.rearrange(dft_micrographs_filtered, "nMic h w -> nMic 1 1 1 h w")
    projections = projections.conj() * dft_micrographs_filtered
    
    # Inverse FFT this MIP
    projections = torch.fft.irfftn(projections, dim=(-2, -1))

    return projections

def simple_cross_correlation_single(
    projections: torch.Tensor,
    dft_micrographs_filtered: torch.Tensor,
) -> torch.Tensor:
    """
    Simple cross correlation.

    Parameters
    ----------
    projections : torch.Tensor
        The projections to cross correlate.
    dft_micrographs_filtered : torch.Tensor
        The DFT of the micrographs filtered by the template.

    Returns
    -------
    torch.Tensor
        The cross correlated projections.
    """
    # FFT shift to edge
    projections = torch.fft.fftshift(projections, dim=(-2, -1))
    # do FFT and 0 central pixel
    projections = torch.fft.rfftn(projections, dim=(-2, -1))
    # zero central pixel
    projections[:, 0, 0] = 0 + 0j
    # cross correlations
    dft_micrographs_filtered = einops.rearrange(dft_micrographs_filtered, "h w -> 1 1 1 h w")
    projections = projections.conj() * dft_micrographs_filtered
    
    # Inverse FFT this MIP
    projections = torch.fft.irfftn(projections, dim=(-2, -1))

    return projections

