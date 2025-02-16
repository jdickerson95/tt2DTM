"""Helper functions for pre-processing data for 2DTM."""

import torch
from torch_fourier_filter.ctf import calculate_ctf_2d
from torch_fourier_filter.whitening import whitening_filter
from torch_so3.hopf_angles import get_uniform_euler_angles


def calculate_whitening_filter_template(
    image: torch.Tensor,
    output_shape: tuple[int, int],
    smoothing: bool = False,
) -> torch.Tensor:
    """Calculation of the whitening filter for the template.

    Parameters
    ----------
    image : torch.Tensor
        The image to use as a reference (power spectrum calculated from here).
    output_shape : tuple[int, int]
        Desired output shape for the filter. Argument is in terms of real-space shape
    smoothing : bool, optional
        If True, apply smoothing to the filter, by default False.

    Returns
    -------
    torch.Tensor
        The whitening filter for the template.
    """
    image_dft = torch.fft.rfftn(image)

    return whitening_filter(
        image_dft=image_dft,
        rfft=True,
        fftshift=False,
        do_power_spectrum=True,
        output_shape=output_shape,
        output_rfft=True,
        output_fftshift=False,
    )


def calculate_ctf_filter_stack(
    pixel_size: float,
    template_shape: tuple[int, int],
    defocus_u: float,  # in um, *NOT* Angstrom
    defocus_v: float,  # in um, *NOT* Angstrom
    astigmatism_angle: float,
    defocus_min: float,  # in um, *NOT* Angstrom
    defocus_max: float,  # in um, *NOT* Angstrom
    defocus_step: float,  # in um, *NOT* Angstrom
    amplitude_contrast_ratio: float = 0.07,
    spherical_aberration: float = 2.7,
    phase_shift: float = 0.0,
    voltage: float = 300.0,
    ctf_B_factor: float = 60.0,
) -> torch.Tensor:
    """Calculate stack (batch) of CTF filters to apply to projections during 2DTM.

    NOTE: While the defocus in the YAML and other Pydantic models is in Angstrom, this
    function expects defocus in um! Beware to convert accordingly.

    Parameters
    ----------
    pixel_size : float
        The pixel size of the images in Angstrom.
    template_shape : tuple[int, int]
        Desired output shape for the filter, in real space. Note that RFFT here cause
        resulting filter to *not* be this passed tuple, but (h, w // 2 + 1).
    defocus_u : float
        The defocus in the u direction, in um.
    defocus_v : float
        The defocus in the v direction, in um.
    astigmatism_angle : float
        The angle of defocus astigmatism, in degrees.
    defocus_min : float
        The minimum relative defocus to consider, in um.
    defocus_max : float
        The maximum relative defocus to consider, in um.
    defocus_step : float
        The step size between defocus values, in um.
    amplitude_contrast_ratio : float, optional
        The amplitude contrast ratio, by default 0.07.
    spherical_aberration : float, optional
        The spherical aberration constant, in mm, by default 2.7.
    phase_shift : float, optional
        The phase shift constant, by default 0.0.
    voltage : float, optional
        The voltage of the microscope, in kV, by default 300.0.
    ctf_B_factor : float, optional
        The additional B factor for the CTF, by default 60.0.
    """
    ctf_filters = []

    for delta_df in torch.arange(defocus_min, defocus_max + 1e-8, defocus_step):
        defocus = (defocus_u + defocus_v) / 2 + delta_df
        astigmatism = abs(defocus_u - defocus_v) / 2

        ctf = calculate_ctf_2d(
            defocus=defocus,
            astigmatism=astigmatism,
            astigmatism_angle=astigmatism_angle,
            voltage=voltage,
            spherical_aberration=spherical_aberration,
            amplitude_contrast=amplitude_contrast_ratio,
            b_factor=ctf_B_factor,
            phase_shift=phase_shift,
            pixel_size=pixel_size,
            image_shape=template_shape,
            rfft=True,
            fftshift=False,
        )

        ctf_filters.append(ctf)

    return torch.stack(ctf_filters, dim=0).squeeze()


def do_image_preprocessing(image: torch.Tensor) -> torch.Tensor:
    """Pre-processes the input image before running the algorithm.

    1. RFFT
    2. Calculate whitening filter
    2. Zero central pixel
    3. Calculate whitening filter and do element-wise multiplication.
    4. Zero central pixel again (superfluous, but following cisTEM)
    5. Normalize (x /= sqrt(sum(abs(x)**2)); pixelwise)

    Parameters
    ----------
    image : torch.Tensor
        The input image to be pre-processed.

    Returns
    -------
    torch.Tensor
        The pre-processed image.

    """
    image_dft = torch.fft.rfftn(image)
    image_dft[0, 0] = 0 + 0j

    wf_image = whitening_filter(
        image_dft=image_dft,
        rfft=True,
        fftshift=False,
        do_power_spectrum=True,
    )
    image_dft *= wf_image
    image_dft[0, 0] = 0 + 0j  # superfluous, but following cisTEM

    # NOTE: Extra indexing happening with squared_sum so that Hermitian pairs are
    # counted, but we skip the first column of the RFFT which should not be duplicated.
    squared_image_dft = torch.abs(image_dft) ** 2
    squared_sum = squared_image_dft.sum() + squared_image_dft[:, 1:].sum()
    image_dft /= torch.sqrt(squared_sum)

    image_dft *= image.numel()  # Scale to variance 1 in real-space

    return image_dft


def calculate_searched_orientations(
    in_plane_angular_step: float,
    out_of_plane_angular_step: float,
    phi_min: float,
    phi_max: float,
    theta_min: float,
    theta_max: float,
    psi_min: float,
    psi_max: float,
    template_symmetry: str = "C1",
    # orientation_sampling_method: str
) -> torch.Tensor:
    """Helper function for calculating the searched orientations.

    Parameters
    ----------
    in_plane_angular_step : float
        The step size for in-plane angles.
    out_of_plane_angular_step : float
        The step size for out-of-plane angles.
    phi_min : float
        The minimum phi angle.
    phi_max : float
        The maximum phi angle.
    theta_min : float
        The minimum theta angle.
    theta_max : float
        The maximum theta angle.
    psi_min : float
        The minimum psi angle.
    psi_max : float
        The maximum psi angle.
    template_symmetry : str, optional
        The symmetry of the template, by default "C1".

    Returns
    -------
    torch.Tensor
        The searched orientations as Euler angles in 'zyz' convention.
    """
    if template_symmetry != "C1":
        raise NotImplementedError(
            "Template symmetry is implemented in package 'torch-fourier-filter', "
            "BUT we have not added the automatic conversions yet."
        )

    return get_uniform_euler_angles(
        in_plane_step=in_plane_angular_step,
        out_of_plane_step=out_of_plane_angular_step,
        phi_min=phi_min,
        phi_max=phi_max,
        theta_min=theta_min,
        theta_max=theta_max,
        psi_min=psi_min,
        psi_max=psi_max,
    )
