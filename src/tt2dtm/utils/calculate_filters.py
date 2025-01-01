import einops
import pandas as pd
import torch
from torch_fourier_filter.bandpass import bandpass_filter
from torch_fourier_filter.whitening import whitening_filter

eps = 1e-10


def calculate_2d_template_filters(
    all_inputs: dict,
    micrograph_data: pd.DataFrame,
    dft_micrographs: torch.Tensor,
    micrograph_shape: tuple[int, int],
    template_shape: tuple[int, int, int],
):
    whiten_filter_template, bandpass_filter_template = None, None
    if all_inputs["filters"]["whitening_filter"]["enabled"]:
        whiten_filter_template = whitening_filter(
            image_dft=dft_micrographs,
            image_shape=micrograph_shape,
            output_shape=template_shape[-2:],
            rfft=True,
            fftshift=False,
            dimensions_output=2,
            smoothing=False,
            power_spec=True,
        )
    if all_inputs["filters"]["bandpass_filter"]["enabled"]:
        # convert res to fraction of px (range 0-0.5)
        pixel_size = float(micrograph_data["rlnMicrographPixelSize"][0])
        low_res = pixel_size / float(
            all_inputs["filters"]["bandpass_filter"]["lower_resolution"]
        )
        high_res = pixel_size / float(
            all_inputs["filters"]["bandpass_filter"]["upper_resolution"]
        )
        # High res max 0.5
        if high_res > 0.5:
            high_res = 0.5
        bandpass_filter_template = bandpass_filter(  # This needs to be made into 3d
            low=low_res,
            high=high_res,
            falloff=(high_res - low_res) / 50,
            image_shape=template_shape[-2:],
            rfft=True,
            fftshift=False,
        )
        # Make the bandpass filter length of whitening
        bandpass_filter_template = einops.repeat(
            bandpass_filter_template, "h w -> b h w", b=dft_micrographs.shape[0]
        )
    return whiten_filter_template, bandpass_filter_template


def calculate_micrograph_filters(
    all_inputs: dict,
    micrograph_data: pd.DataFrame,
    dft_micrographs: torch.Tensor,
    micrograph_shape: tuple[int, int],
):
    whiten_filter_micrograph, bandpass_filter_micrograph = None, None
    if all_inputs["filters"]["whitening_filter"]["enabled"]:
        whiten_filter_micrograph = whitening_filter(
            image_dft=dft_micrographs,
            image_shape=micrograph_shape,
            output_shape=micrograph_shape,
            rfft=True,
            fftshift=False,
            dimensions_output=2,
            smoothing=False,
            power_spec=True,
        )
    if all_inputs["filters"]["bandpass_filter"]["enabled"]:
        # convert res to fraction of px (range 0-0.5)
        pixel_size = float(micrograph_data["rlnMicrographPixelSize"][0])
        low_res = pixel_size / float(
            all_inputs["filters"]["bandpass_filter"]["lower_resolution"]
        )
        high_res = pixel_size / float(
            all_inputs["filters"]["bandpass_filter"]["upper_resolution"]
        )
        # High res max 0.5
        if high_res > 0.5:
            high_res = 0.5

        bandpass_filter_micrograph = bandpass_filter(
            low=low_res,
            high=high_res,
            falloff=(high_res - low_res) / 50,
            image_shape=micrograph_shape,
            rfft=True,
            fftshift=False,
        )
        # Make the bandpass filter length of micrographs
        bandpass_filter_micrograph = einops.repeat(
            bandpass_filter_micrograph, "h w -> b h w", b=dft_micrographs.shape[0]
        )
    return whiten_filter_micrograph, bandpass_filter_micrograph


def combine_filters(
    filter1: torch.Tensor,
    filter2: torch.Tensor,
) -> torch.Tensor:
    combined_filter = torch.ones(filter1.shape)
    if (filter1 is not None) and (filter2 is not None):
        combined_filter = filter1 * filter2
    elif filter1 is not None:
        combined_filter = filter1
    elif filter2 is not None:
        combined_filter = filter2
    return combined_filter


def get_Cs_range(
    pixel_size: float,
    pixel_size_range: float,
    pixel_size_step: float,
    Cs: float = 2.7,
) -> torch.Tensor:
    pixel_sizes = torch.arange(
        pixel_size - pixel_size_range / 2,
        pixel_size + pixel_size_range / 2 + eps,
        pixel_size_step,
    )
    Cs_values = Cs / torch.pow(pixel_sizes / pixel_size, 4)
    return Cs_values


def get_defocus_range(
    defocus_range: float,
    defocus_step: float,
) -> torch.Tensor:
    defocus_values = torch.arange(
        -defocus_range / 2, defocus_range / 2 + eps, defocus_step
    )
    defocus_values = defocus_values * 1e4  # convert to angstroms
    return defocus_values


def get_defocus_values(
    defoc_vals: torch.Tensor,
    defoc_range: torch.Tensor,
) -> torch.Tensor:
    defoc_range = einops.repeat(defoc_range, "n -> 1 n")
    defoc_vals = einops.rearrange(defoc_vals, "n -> n 1")
    defoc_vals = defoc_vals + defoc_range
    return defoc_vals


def get_max_box_size(
    micrograph_data: dict,
    max_Cs: float,
    max_defocus: float,
) -> torch.Tensor:
    pass
