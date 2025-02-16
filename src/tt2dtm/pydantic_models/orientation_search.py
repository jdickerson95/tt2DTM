"""Serialization and validation of orientation search parameters for 2DTM."""

from typing import Annotated

from pydantic import Field, field_validator

from tt2dtm.pydantic_models.types import BaseModel2DTM

ALLOWED_ORIENTATION_SAMPLING_METHODS = ["Hopf Fibration"]


class OrientationSearchConfig(BaseModel2DTM):
    """Serialization and validation of orientation search parameters for 2DTM.

    The angles -- psi, theta, and phi -- represent Euler angles in the 'ZYZ'
    convention.

    TODO: Implement indexing to get the i-th or range of orientations in the
    search space (need to be ordered).

    Attributes
    ----------
    orientation_sampling_method : str
        Method for sampling orientations. Default is 'Hopf Fibration'.
        Currently only 'Hopf Fibration' is supported.
    template_symmetry : str
        Symmetry group of the template. Default is 'C1'.
        Currently only 'C1' is supported.
    psi_min : float
        Minimum value for the psi angle in degrees.
    psi_max : float
        Maximum value for the psi angle in degrees.
    theta_min : float
        Minimum value for the theta angle in degrees.
    theta_max : float
        Maximum value for the theta angle in degrees.
    phi_min : float
        Minimum value for the phi angle in degrees.
    phi_max : float
        Maximum value for the phi angle in degrees.
    in_plane_angular_step : float
        Angular step size for in-plane rotations in degrees. Must be greater
        than 0.
    out_of_plane_angular_step : float
        Angular step size for out-of-plane rotations in degrees. Must be
        greater than 0.
    """

    orientation_sampling_method: str = "Hopf Fibration"
    template_symmetry: str = "C1"
    psi_min: float = 0.0
    psi_max: float = 360.0
    theta_min: float = 0.0
    theta_max: float = 180.0
    phi_min: float = 0.0
    phi_max: float = 360.0
    in_plane_angular_step: Annotated[float, Field(..., gt=0.0)] = 1.5
    out_of_plane_angular_step: Annotated[float, Field(..., gt=0.0)] = 2.5

    @field_validator("orientation_sampling_method")
    def validate_orientation_sampling_method(cls, value):  # type: ignore
        """Validate from allowed orientation sampling methods."""
        if value not in ALLOWED_ORIENTATION_SAMPLING_METHODS:
            raise ValueError(
                f"Currently only supports the following sampling "
                f"method(s):\n\t {ALLOWED_ORIENTATION_SAMPLING_METHODS}"
            )

        return value


class RefineOrientationConfig(BaseModel2DTM):
    """Serialization and validation of orientation refinement parameters.

    Angles will be sampled from [-coarse_step, coarse_step] in increments of
    'fine_step' for the orientation refinement search.

    Attributes
    ----------
    orientation_sampling_method : str
        Method for sampling orientations. Default is 'Hopf Fibration'.
        Currently only 'Hopf Fibration' is supported.
    template_symmetry : str
        Symmetry group of the template. Default is 'C1'.
        Currently only 'C1' is supported.
    in_plane_angular_step_coarse : float
        Angular step size for in-plane rotations in degrees for previous, coarse search.
        This corresponds to the 'OrientationSearchConfig.in_plane_angular_step' value
        for the match template program. Must be greater than or equal to 0.
    in_plane_angular_step_fine : float
        Angular step size for in-plane rotations in degrees for current, fine search.
        Must be greater than or equal to 0.
    out_of_plane_angular_step_coarse : float
        Angular step size for out-of-plane rotations in degrees for previous, coarse
        search. This corresponds to the
        'OrientationSearchConfig.out_of_plane_angular_step' value for the match template
        program. Must be greater than or equal to 0.
    out_of_plane_angular_step_fine : float
        Angular step size for out-of-plane rotations in degrees for current, fine
        search. Must be greater than or equal to 0.

    """

    orientation_sampling_method: str = "Hopf Fibration"
    template_symmetry: str = "C1"
    in_plane_angular_step_coarse: Annotated[float, Field(..., ge=0.0)] = 1.5
    in_plane_angular_step_fine: Annotated[float, Field(..., ge=0.0)] = 0.1
    out_of_plane_angular_step_coarse: Annotated[float, Field(..., ge=0.0)] = 2.5
    out_of_plane_angular_step_fine: Annotated[float, Field(..., ge=0.0)] = 0.1

    @field_validator("orientation_sampling_method")
    def validate_orientation_sampling_method(cls, value):  # type: ignore
        """Validate from allowed orientation sampling methods."""
        if value not in ALLOWED_ORIENTATION_SAMPLING_METHODS:
            raise ValueError(
                f"Currently only supports the following sampling "
                f"method(s):\n\t {ALLOWED_ORIENTATION_SAMPLING_METHODS}"
            )

        return value
