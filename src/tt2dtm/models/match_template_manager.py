"""Root-level model for serialization and validation of 2DTM parameters."""

import os
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from pydantic import ConfigDict, field_validator
from torch_grid_utils import fftfreq_grid

from tt2dtm.backend_project import core_match_template_project
from tt2dtm.models.computational_config import ComputationalConfig
from tt2dtm.models.correlation_filters import PreprocessingFilters
from tt2dtm.models.defocus_search_config import DefocusSearchConfig
from tt2dtm.models.match_template_result import MatchTemplateResult
from tt2dtm.models.optics_group import OpticsGroup
from tt2dtm.models.orientation_search_config import OrientationSearchConfig
from tt2dtm.models.pixel_size_search_config import PixelSizeSearchConfig
from tt2dtm.models.types import BaseModel2DTM, ExcludedTensor
from tt2dtm.utils.data_io import load_mrc_image, load_mrc_volume
from tt2dtm.utils.pre_processing import (
    calculate_bandpass_filter,
    calculate_ctf_filter_stack,
    calculate_searched_orientations,
    calculate_whitening_filter_template,
    do_image_preprocessing,
)


def select_gpu_devices(gpu_ids: int | list[int]) -> list[torch.device]:
    """Convert requested GPU IDs to torch device objects.

    Parameters
    ----------
    gpu_ids : int | list[int]
        GPU ID(s) to use for computation.

    Returns
    -------
    list[torch.device]
    """
    if isinstance(gpu_ids, int):
        if gpu_ids < -1:  # -2 or lower means CPU
            return [torch.device("cpu")]
        gpu_ids = [gpu_ids]

    devices = [torch.device(f"cuda:{gpu_id}") for gpu_id in gpu_ids]

    return devices


class MatchTemplateManager(BaseModel2DTM):
    """Model holding parameters necessary for running full orientation 2DTM.

    Attributes
    ----------
    micrograph_path : str
        Path to the micrograph .mrc file.
    template_volume_path : str
        Path to the template volume .mrc file.
    micrograph : ExcludedTensor
        Image to run template matching on. Not serialized.
    template_volume : ExcludedTensor
        Template volume to match against. Not serialized.
    optics_group : OpticsGroup
        Optics group parameters for the imaging system on the microscope.
    defocus_search_config : DefocusSearchConfig
        Parameters for searching over defocus values.
    pixel_size_search_config : PixelSizeSearchConfig
        Parameters for searching over pixel sizes.
    orientation_search_config : OrientationSearchConfig
        Parameters for searching over orientation angles.
    preprocessing_filters : PreprocessingFilters
        Configurations for the preprocessing filters to apply during
        correlation.
    match_template_result : MatchTemplateResult
        Result of the match template program stored as an instance of the
        `MatchTemplateResult` class.
    computational_config : ComputationalConfig
        Parameters for controlling computational resources.

    Methods
    -------
    TODO: Document these methods
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    # Serialized attributes
    micrograph_path: str
    template_volume_path: str
    optics_group: OpticsGroup
    defocus_search_config: DefocusSearchConfig
    orientation_search_config: OrientationSearchConfig
    pixel_size_search_config: PixelSizeSearchConfig
    preprocessing_filters: PreprocessingFilters
    match_template_result: MatchTemplateResult
    computational_config: ComputationalConfig

    # Non-serialized large array-like attributes
    micrograph: ExcludedTensor
    template_volume: ExcludedTensor

    ###########################
    ### Pydantic Validators ###
    ###########################

    @field_validator("micrograph_path")  # type: ignore
    def validate_micrograph_path(cls, v) -> str:
        """Ensure the micrograph file exists."""
        if not os.path.exists(v):
            raise ValueError(f"File '{v}' for micrograph does not exist.")

        return str(v)

    @field_validator("template_volume_path")  # type: ignore
    def validate_template_volume_path(cls, v) -> str:
        """Ensure the template volume file exists."""
        if not os.path.exists(v):
            raise ValueError(f"File '{v}' for template volume does not exist.")

        return str(v)

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Load the data from the MRC files
        self.micrograph = load_mrc_image(self.micrograph_path)
        self.template_volume = load_mrc_volume(self.template_volume_path)

    ############################################
    ### Functional (data processing) methods ###
    ############################################

    def make_backend_core_function_kwargs(self) -> dict[str, Any]:
        """Generates the keyword arguments for backend call from held parameters."""
        image = torch.from_numpy(self.micrograph)
        template = torch.from_numpy(self.template_volume)

        #####Pad the template#####
        pad_length = template.shape[-1] // 2
        template = F.pad(template, pad=[pad_length] * 6, mode="constant", value=0)

        # premultiply by sinc2
        #grid = fftfreq_grid(
        #    image_shape=template.shape,
        #    rfft=False,
        #    fftshift=True,
        #    norm=True,
        #    device=template.device,
        #)
        # template = template * torch.sinc(grid) ** 2

        template_shape = template.shape[-2:]

        whitening_filter = calculate_whitening_filter_template(
            image, template_shape, smoothing=False
        )
        bandpass_filter_template = calculate_bandpass_filter(
            low_pass_cutoff=0.00,
            high_pass_cutoff=0.5,
            falloff=0.02,
            image_shape=template_shape,
            rfft=True,
            fftshift=False,
            device=template.device,
        )
        whitening_filter *= bandpass_filter_template
        image_preprocessed_dft = do_image_preprocessing(image)

        defocus_values = self.defocus_search_config.defocus_values
        defocus_values = torch.tensor(defocus_values, dtype=torch.float32)
        ctf_filters = calculate_ctf_filter_stack(
            pixel_size=self.optics_group.pixel_size,
            template_shape=(template_shape[0], template_shape[0]),
            defocus_u=self.optics_group.defocus_u * 1e-4,  # A to um
            defocus_v=self.optics_group.defocus_v * 1e-4,  # A to um
            astigmatism_angle=self.optics_group.defocus_astigmatism_angle,
            defocus_min=self.defocus_search_config.defocus_min * 1e-4,  # A to um
            defocus_max=self.defocus_search_config.defocus_max * 1e-4,  # A to um
            defocus_step=self.defocus_search_config.defocus_step * 1e-4,  # A to um
            amplitude_contrast_ratio=self.optics_group.amplitude_contrast_ratio,
            spherical_aberration=self.optics_group.spherical_aberration,
            phase_shift=self.optics_group.phase_shift,
            voltage=self.optics_group.voltage,
            ctf_B_factor=self.optics_group.ctf_B_factor,
        )

        euler_angles = calculate_searched_orientations(
            in_plane_angular_step=self.orientation_search_config.in_plane_angular_step,
            out_of_plane_angular_step=self.orientation_search_config.out_of_plane_angular_step,
            phi_min=self.orientation_search_config.phi_min,
            phi_max=self.orientation_search_config.phi_max,
            theta_min=self.orientation_search_config.theta_min,
            theta_max=self.orientation_search_config.theta_max,
            psi_min=self.orientation_search_config.psi_min,
            psi_max=self.orientation_search_config.psi_max,
            template_symmetry=self.orientation_search_config.template_symmetry,
        )
        euler_angles = euler_angles.to(torch.float32)

        device_list = select_gpu_devices(self.computational_config.gpu_ids)

        template_dft = torch.fft.fftshift(template, dim=(-3, -2, -1))
        # template_dft = torch.fft.rfftn(template, dim=(-3, -2, -1))
        template_dft = torch.fft.rfftn(template_dft, dim=(-3, -2, -1))
        template_dft = torch.fft.fftshift(template_dft, dim=(-3, -2))  # skip rfft dim

        return {
            "image_dft": image_preprocessed_dft,
            "template_dft": template_dft,
            "ctf_filters": ctf_filters,
            "whitening_filter_template": whitening_filter,
            "euler_angles": euler_angles,
            "defocus_values": defocus_values,
            "device": device_list,
        }

    def make_backend_project_core_function_kwargs(self) -> dict[str, Any]:
        """Generates the keyword arguments for backend call from held parameters."""
        image = torch.from_numpy(self.micrograph)
        template = torch.from_numpy(self.template_volume)

        #####Pad the template#####
        # pad_length = template.shape[-1] // 2
        # template = F.pad(template, pad=[pad_length] * 6, mode='constant', value=0)
        template_shape = template.shape[-2:]

        whitening_filter = calculate_whitening_filter_template(
            image, template_shape, smoothing=False
        )
        bandpass_filter_template = calculate_bandpass_filter(
            low_pass_cutoff=0.00,
            high_pass_cutoff=0.5,
            falloff=0.02,
            image_shape=template_shape,
            rfft=True,
            fftshift=False,
            device=template.device,
        )
        whitening_filter *= bandpass_filter_template
        image_preprocessed_dft = do_image_preprocessing(image)

        defocus_values = self.defocus_search_config.defocus_values
        defocus_values = torch.tensor(defocus_values, dtype=torch.float32)
        ctf_filters = calculate_ctf_filter_stack(
            pixel_size=self.optics_group.pixel_size,
            template_shape=(template_shape[0], template_shape[0]),
            defocus_u=self.optics_group.defocus_u * 1e-4,  # A to um
            defocus_v=self.optics_group.defocus_v * 1e-4,  # A to um
            astigmatism_angle=self.optics_group.defocus_astigmatism_angle,
            defocus_min=self.defocus_search_config.defocus_min * 1e-4,  # A to um
            defocus_max=self.defocus_search_config.defocus_max * 1e-4,  # A to um
            defocus_step=self.defocus_search_config.defocus_step * 1e-4,  # A to um
            amplitude_contrast_ratio=self.optics_group.amplitude_contrast_ratio,
            spherical_aberration=self.optics_group.spherical_aberration,
            phase_shift=self.optics_group.phase_shift,
            voltage=self.optics_group.voltage,
            ctf_B_factor=self.optics_group.ctf_B_factor,
        )

        euler_angles = calculate_searched_orientations(
            in_plane_angular_step=self.orientation_search_config.in_plane_angular_step,
            out_of_plane_angular_step=self.orientation_search_config.out_of_plane_angular_step,
            phi_min=self.orientation_search_config.phi_min,
            phi_max=self.orientation_search_config.phi_max,
            theta_min=self.orientation_search_config.theta_min,
            theta_max=self.orientation_search_config.theta_max,
            psi_min=self.orientation_search_config.psi_min,
            psi_max=self.orientation_search_config.psi_max,
            template_symmetry=self.orientation_search_config.template_symmetry,
        )
        euler_angles = euler_angles.to(torch.float32)

        device_list = select_gpu_devices(self.computational_config.gpu_ids)

        template_dft = torch.fft.fftshift(template, dim=(-3, -2, -1))
        # template_dft = torch.fft.rfftn(template, dim=(-3, -2, -1))
        template_dft = torch.fft.rfftn(template_dft, dim=(-3, -2, -1))
        template_dft = torch.fft.fftshift(template_dft, dim=(-3, -2))  # skip rfft dim

        return {
            "image_dft": image_preprocessed_dft,
            "template_dft_shape": template_dft.shape[-2:],
            "ctf_filters": ctf_filters,
            "whitening_filter_template": whitening_filter,
            "euler_angles": euler_angles,
            "defocus_values": defocus_values,
            "device": device_list,
        }

    def run_match_template(self, projection_batch_size: int = 1) -> None:
        """Runs the base match template in pytorch.

        Parameters
        ----------
        projection_batch_size : int
            The number of projections to process in a single batch. Default is 1.

        Returns
        -------
        None
        """
        # core_kwargs = self.make_backend_core_function_kwargs()
        # results = core_match_template(
        #    **core_kwargs, projection_batch_size=projection_batch_size
        # )
        core_kwargs = self.make_backend_project_core_function_kwargs()

        results = core_match_template_project(
            **core_kwargs, projection_batch_size=projection_batch_size
        )

        # Place results into the `MatchTemplateResult` object and save it.
        self.match_template_result.mip = results["mip"]
        self.match_template_result.scaled_mip = results["scaled_mip"]

        self.match_template_result.correlation_average = results["correlation_sum"]
        self.match_template_result.correlation_variance = results[
            "correlation_squared_sum"
        ]
        self.match_template_result.orientation_psi = results["best_psi"]
        self.match_template_result.orientation_theta = results["best_theta"]
        self.match_template_result.orientation_phi = results["best_phi"]
        self.match_template_result.relative_defocus = results["best_defocus"]

        # TODO: Implement pixel size calculation
        self.match_template_result.pixel_size = torch.zeros(1, 1)
        self.match_template_result.total_projections = results["total_projections"]
        self.match_template_result.total_orientations = results["total_orientations"]
        self.match_template_result.total_defocus = results["total_defocus"]
        self.match_template_result.export_results()
