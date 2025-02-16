"""Root-level model for serialization and validation of 2DTM parameters."""

import os
from typing import Any, ClassVar, Literal, Optional

import mrcfile
import pandas as pd
import torch
from pydantic import ConfigDict, field_validator

from tt2dtm.backend import core_match_template
from tt2dtm.pydantic_models.computational_config import ComputationalConfig
from tt2dtm.pydantic_models.correlation_filters import PreprocessingFilters
from tt2dtm.pydantic_models.defocus_search import DefocusSearchConfig
from tt2dtm.pydantic_models.match_template_result import MatchTemplateResult
from tt2dtm.pydantic_models.optics_group import OpticsGroup
from tt2dtm.pydantic_models.orientation_search import OrientationSearchConfig
from tt2dtm.pydantic_models.pixel_size_search_config import PixelSizeSearchConfig
from tt2dtm.pydantic_models.types import BaseModel2DTM, ExcludedTensor
from tt2dtm.utils.data_io import load_mrc_image, load_mrc_volume
from tt2dtm.utils.pre_processing import (
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

    def __init__(self, skip_mrc_preloads: bool = False, **data: Any):
        super().__init__(**data)

        # Load the data from the MRC files
        if not skip_mrc_preloads:
            self.micrograph = load_mrc_image(self.micrograph_path)
            self.template_volume = load_mrc_volume(self.template_volume_path)

    ############################################
    ### Functional (data processing) methods ###
    ############################################

    def make_backend_core_function_kwargs(self) -> dict[str, Any]:
        """Generates the keyword arguments for backend call from held parameters."""
        if not isinstance(self.micrograph, torch.Tensor):
            image = torch.from_numpy(self.micrograph)
        else:
            image = self.micrograph

        if not isinstance(self.template_volume, torch.Tensor):
            template = torch.from_numpy(self.template_volume)
        else:
            template = self.template_volume

        template_shape = template.shape[-2:]

        whitening_filter = calculate_whitening_filter_template(image, template_shape)
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

    def run_match_template(
        self, orientation_batch_size: int = 1, do_result_export: bool = True
    ) -> None:
        """Runs the base match template in pytorch.

        Parameters
        ----------
        orientation_batch_size : int
            The number of projections to process in a single batch. Default is 1.
        do_result_export : bool
            If True, call the `MatchTemplateResult.export_results` method to save the
            results to disk directly after running the match template. Default is True.

        Returns
        -------
        None
        """
        core_kwargs = self.make_backend_core_function_kwargs()
        results = core_match_template(
            **core_kwargs, orientation_batch_size=orientation_batch_size
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

        if do_result_export:
            self.match_template_result.export_results()

    def results_to_dataframe(
        self,
        do_peak_shifting: bool = True,
        exclude_columns: Optional[list] = None,
        locate_peaks_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Converts the match template results to a DataFrame with additional info.

        Data included in this dataframe should be sufficient to do cross-correlation on
        the extracted peaks, that is, all the microscope parameters, defocus parameters,
        etc. are included in the dataframe. Run-specific filter information is *not*
        included in this dataframe; use the YAML configuration file to replicate a
        match_template run.

        Parameters
        ----------
        do_peak_shifting : bool, optional
            If True, columns for the image peak position are shifted by half a template
            width to correspond to the center of the particle. Default is True. This
            should generally be left as True unless you know what you are doing.
        exclude_columns : list, optional
            List of columns to exclude from the DataFrame. Default is None and no
            columns are excluded.
        locate_peaks_kwargs : dict, optional
            Keyword arguments to pass to the 'MatchTemplateResult.locate_peaks' method.
            Default is None and no additional keyword arguments are passed.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the match template results.
        """
        # Short circuit if no kwargs and peaks have already been located
        if locate_peaks_kwargs is None:
            if self.match_template_result.match_template_peaks is None:
                self.match_template_result.locate_peaks()
        else:
            self.match_template_result.locate_peaks(**locate_peaks_kwargs)

        df = self.match_template_result.peaks_to_dataframe()

        # DataFrame currently contains pixel coordinates for results. Coordinates in
        # image correspond with upper left corner of the template. Need to translate
        # coordinates by half template width to get to particle center in image.
        # NOTE: We are assuming the template is cubic
        nx = mrcfile.open(self.template_volume_path).header.nx
        if do_peak_shifting:
            df["img_pos_y"] = df["pos_y"] + nx // 2
            df["img_pos_x"] = df["pos_x"] + nx // 2
        else:
            df["img_pos_y"] = df["pos_y"]
            df["img_pos_x"] = df["pos_x"]

        # Also, the positions are in terms of pixels. Also add columns for particle
        # positions in terms of Angstroms.
        pixel_size = self.optics_group.pixel_size
        df["img_pos_y_angstrom"] = df["img_pos_y"] * pixel_size
        df["img_pos_x_angstrom"] = df["img_pos_x"] * pixel_size

        # Add absolute defocus values and other imaging parameters
        df["defocus_u"] = self.optics_group.defocus_u + df["defocus"]
        df["defocus_v"] = self.optics_group.defocus_v + df["defocus"]
        df["defocus_astigmatism_angle"] = self.optics_group.defocus_astigmatism_angle

        # Add paths to the micrograph and reference template
        df["reference_micrograph"] = self.micrograph_path
        df["reference_template"] = self.template_volume_path

        # Add paths to the output statistic files
        df["mip_path"] = self.match_template_result.mip_path
        df["scaled_mip_path"] = self.match_template_result.scaled_mip_path
        df["psi_path"] = self.match_template_result.orientation_psi_path
        df["theta_path"] = self.match_template_result.orientation_theta_path
        df["phi_path"] = self.match_template_result.orientation_phi_path
        df["defocus_path"] = self.match_template_result.relative_defocus_path
        df["correlation_average_path"] = (
            self.match_template_result.correlation_average_path
        )
        df["correlation_variance_path"] = (
            self.match_template_result.correlation_variance_path
        )

        df["pixel_size"] = pixel_size
        df["voltage"] = self.optics_group.voltage
        df["spherical_aberration"] = self.optics_group.spherical_aberration
        df["amplitude_contrast_ratio"] = self.optics_group.amplitude_contrast_ratio
        df["phase_shift"] = self.optics_group.phase_shift
        df["ctf_B_factor"] = self.optics_group.ctf_B_factor

        # Drop columns if requested
        if exclude_columns is not None:
            df = df.drop(columns=exclude_columns)

        return df

    def save_config(self, path: str, mode: Literal["yaml", "json"] = "yaml") -> None:
        """Save this Pydandic model to disk. Wrapper around the serialization methods.

        Parameters
        ----------
        path : str
            Path to save the configuration file.
        mode : Literal["yaml", "json"], optional
            Serialization format to use. Default is 'yaml'.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If an invalid serialization mode is provided.
        """
        if mode == "yaml":
            self.to_yaml(path)
        elif mode == "json":
            self.to_json(path)
        else:
            raise ValueError(f"Invalid serialization mode '{mode}'.")
