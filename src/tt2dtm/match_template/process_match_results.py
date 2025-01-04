"""Functions to process the match results."""
import torch
import einops
import pandas as pd
from tt2dtm.utils.calculate_filters import Cs_to_pixel_size
from tt2dtm.utils.io_handler import write_mrc_files_single, write_survival_histogram, write_mrc_files, write_survival_histogram_single

# Constants
CCG_NOISE_STDEV = 1.0
HISTOGRAM_NUM_POINTS = 512
HISTOGRAM_MIN = -12.5
HISTOGRAM_MAX = 22.5
HISTOGRAM_STEP = (HISTOGRAM_MAX - HISTOGRAM_MIN) / HISTOGRAM_NUM_POINTS

def get_mip_and_best(
    projections: torch.Tensor,
    defocus_values: torch.Tensor,
    Cs_vals: torch.Tensor,
    euler_angles: torch.Tensor,
    micrograph_data: pd.DataFrame,
    num_pixels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the maximum intensity projection and the best defocus, Cs, and Euler angles.

    Parameters
    ----------
    projections : torch.Tensor
        The projections to get the maximum intensity projection from.
    defocus_values : torch.Tensor
        The defocus values to get the best defocus from.
    Cs_vals : torch.Tensor
        The Cs values to get the best Cs from.
    euler_angles : torch.Tensor
        The Euler angles to get the best Euler angles from.
    micrograph_data : pd.DataFrame
        The micrograph data to get the best defocus, Cs, and Euler angles from.

    Returns
    -------
    best_defoc : torch.Tensor
        The best defocus.
    best_pixel_size : torch.Tensor
        The best pixel size.
    best_phi : torch.Tensor
        The best phi.
    best_theta : torch.Tensor
        The best theta.
    best_psi : torch.Tensor
        The best psi.
    maximum_intensiy_projections : torch.Tensor
        The maximum intensity projections.
    """
    # Flatten the dimensions you're reducing over
    flattened = einops.rearrange(projections, "nMic nDefoc nCs nAng h w -> nMic (nDefoc nCs nAng) h w")
    maximum_intensiy_projections, flat_indices = torch.max(flattened, dim=1)  # Reduces along the combined dimension (nDefoc*nCs*nAng)
    nDefoc, nCs, nAng = projections.shape[1:4]
    defoc_indices = (flat_indices // (nCs * nAng)) % nDefoc
    cs_indices = (flat_indices // nAng) % nCs
    ang_indices = flat_indices % nAng

    print("defoc indices", defoc_indices.shape)
    print("cs indices", cs_indices.shape)
    print("ang indices", ang_indices.shape)
    best_defoc = defocus_values[
        torch.arange(defocus_values.shape[0]).unsqueeze(-1).unsqueeze(-1),  # nMic dimension
        defoc_indices  # Contains indices for each spatial position
    ]  # Result shape: (nMic, h, w)
    best_cs = Cs_vals[cs_indices]
    best_angles = euler_angles[ang_indices]

    #convert Cs back to pixel size
    best_pixel_size = Cs_to_pixel_size(
        Cs_vals=best_cs,
        nominal_pixel_size=float(micrograph_data["rlnMicrographPixelSize"][0]),
        nominal_Cs=float(micrograph_data["rlnSphericalAberration"][0]) * 1E7
    )
    #get best phi, theta, psi
    best_phi = best_angles[..., 0]
    best_theta = best_angles[..., 1]
    best_psi = best_angles[..., 2]

    #multiply up MIP
    maximum_intensiy_projections = maximum_intensiy_projections * torch.sqrt(num_pixels)

    return best_defoc, best_pixel_size, best_phi, best_theta, best_psi, maximum_intensiy_projections

def get_mip_and_best_single_mic(
    projections: torch.Tensor,
    defocus_values: torch.Tensor,
    Cs_vals: torch.Tensor,
    euler_angles: torch.Tensor,
    micrograph_data: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the maximum intensity projection and the best defocus, Cs, and Euler angles.

    Parameters
    ----------
    projections : torch.Tensor
        The projections to get the maximum intensity projection from.
    defocus_values : torch.Tensor
        The defocus values to get the best defocus from.
    Cs_vals : torch.Tensor
        The Cs values to get the best Cs from.
    euler_angles : torch.Tensor
        The Euler angles to get the best Euler angles from.
    micrograph_data : pd.DataFrame
        The micrograph data to get the best defocus, Cs, and Euler angles from.

    Returns
    -------
    best_defoc : torch.Tensor
        The best defocus.
    best_pixel_size : torch.Tensor
        The best pixel size.
    best_phi : torch.Tensor
        The best phi.
    best_theta : torch.Tensor
        The best theta.
    best_psi : torch.Tensor
        The best psi.
    maximum_intensiy_projections : torch.Tensor
        The maximum intensity projections.
    """
    # Flatten the dimensions you're reducing over
    flattened = einops.rearrange(projections, "nDefoc nCs nAng h w -> (nDefoc nCs nAng) h w")
    maximum_intensiy_projections, flat_indices = torch.max(flattened, dim=0)  # Reduces along the combined dimension (nDefoc*nCs*nAng)
    nDefoc, nCs, nAng = projections.shape[0:3]
    defoc_indices = (flat_indices // (nCs * nAng)) % nDefoc
    cs_indices = (flat_indices // nAng) % nCs
    ang_indices = flat_indices % nAng

    print("defoc indices", defoc_indices.shape)
    print("cs indices", cs_indices.shape)
    print("ang indices", ang_indices.shape)
    best_defoc = defocus_values[defoc_indices]
    best_cs = Cs_vals[cs_indices]
    best_angles = euler_angles[ang_indices]

    #convert Cs back to pixel size
    best_pixel_size = Cs_to_pixel_size(
        Cs_vals=best_cs,
        nominal_pixel_size=float(micrograph_data["rlnMicrographPixelSize"]),
        nominal_Cs=float(micrograph_data["rlnSphericalAberration"])
    )
    #get best phi, theta, psi
    best_phi = best_angles[..., 0]
    best_theta = best_angles[..., 1]
    best_psi = best_angles[..., 2]

    # Convert batch results to float32 if needed
    maximum_intensiy_projections = maximum_intensiy_projections.to(torch.float32)
    best_defoc = best_defoc.to(torch.float32)
    best_pixel_size = best_pixel_size.to(torch.float32)
    best_phi = best_phi.to(torch.float32)
    best_theta = best_theta.to(torch.float32)
    best_psi = best_psi.to(torch.float32)

    return best_defoc, best_pixel_size, best_phi, best_theta, best_psi, maximum_intensiy_projections

def calc_sum_correlation(
    projections: torch.Tensor,
    num_pixels: torch.Tensor,
    total_correlation_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the sum correlation.

    Parameters
    ----------
    projections : torch.Tensor
        The projections to calculate the sum correlation from.
    num_pixels : torch.Tensor
        The number of pixels.
    total_correlation_positions : torch.Tensor
        The total correlation positions.

    Returns
    -------
    sum_correlation : torch.Tensor
        The sum correlation.
    sum_correlation_squared : torch.Tensor
        The sum correlation squared.
    """
    sum_correlation = einops.reduce(projections, "nMic nDefoc nCs nAng h w -> nMic h w", "sum")
    sum_correlation_squared = einops.reduce(projections ** 2, "nMic nDefoc nCs nAng h w -> nMic h w", "sum")
    sum_correlation = sum_correlation / total_correlation_positions
    sum_correlation_squared = sum_correlation_squared / total_correlation_positions - (sum_correlation ** 2)
    sum_correlation_squared = torch.sqrt(torch.clamp(sum_correlation_squared, min=0)) * torch.sqrt(num_pixels)
    sum_correlation = sum_correlation * torch.sqrt(num_pixels)

    return sum_correlation, sum_correlation_squared

def normalize_mip(
    maximum_intensiy_projections: torch.Tensor,
    sum_correlation: torch.Tensor,
    sum_correlation_squared: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize the maximum intensity projection.

    Parameters
    ----------
    maximum_intensiy_projections : torch.Tensor
        The maximum intensity projections.
    sum_correlation : torch.Tensor
        The sum correlation.
    sum_correlation_squared : torch.Tensor
        The sum correlation squared.

    Returns
    -------
    maximum_intensiy_projections_normalized : torch.Tensor
        The normalized maximum intensity projections.
    """
    maximum_intensiy_projections_normalized = maximum_intensiy_projections - sum_correlation
    maximum_intensiy_projections_normalized = torch.where(sum_correlation_squared == 0, 
                                             torch.zeros_like(maximum_intensiy_projections_normalized),
                                             maximum_intensiy_projections_normalized / sum_correlation_squared)

    return maximum_intensiy_projections_normalized


def get_expected_noise(
    num_pixels: torch.Tensor,
    total_correlation_positions: torch.Tensor,
):
    """Get the expected noise.

    Parameters
    ----------
    num_pixels : torch.Tensor
        The number of pixels.
    total_correlation_positions : torch.Tensor
        The total correlation positions.
    """
    #Get the expected noise
    erf_input = torch.tensor(2.0, dtype=torch.float64) / (torch.tensor(1.0, dtype=torch.float64) * num_pixels.to(torch.float64) * total_correlation_positions)
    #erfc_input = torch.tensor(1.0, dtype=torch.float64) - erf_input
    inverse_c_erf = torch.erfinv(torch.tensor(1.0, dtype=torch.float64) - erf_input)  
    expected_noise = (2**0.5) * CCG_NOISE_STDEV * inverse_c_erf

    return expected_noise

def calc_survival_histogram(
    projections: torch.Tensor,
    num_pixels: torch.Tensor,
    total_correlation_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Calculate the survival histogram.

    Parameters
    ----------
    projections : torch.Tensor
        The projections to calculate the survival histogram from.
    num_pixels : torch.Tensor
        The number of pixels.
    total_correlation_positions : torch.Tensor
        The total correlation positions.

    Returns
    -------
    survival_histogram : torch.Tensor
        The survival histogram.
    expected_survival_hist : torch.Tensor
        The expected survival histogram.
    temp_float : float
        The temp float.
    histogram_data : torch.Tensor
        The histogram data.
    """
    histogram_min_scaled = HISTOGRAM_MIN / num_pixels**0.5
    histogram_step_scaled = HISTOGRAM_STEP / num_pixels**0.5
    # Create histogram for each micrograph
    histogram_data = torch.zeros((projections.shape[0], HISTOGRAM_NUM_POINTS), device=projections.device)
    
    # Calculate bins using torch operations
    bins = ((projections - histogram_min_scaled) / histogram_step_scaled).long()
    # For each micrograph
    for mic_idx in range(projections.shape[0]):
        # Create a mask for valid bins (between 0 and histogram_number_of_points)
        valid_mask = (bins[mic_idx] >= 0) & (bins[mic_idx] < HISTOGRAM_NUM_POINTS)
        
        # Use torch.bincount to count occurrences of each bin
        # Only count values where the mask is True
        valid_bins = bins[mic_idx][valid_mask]
        histogram_data[mic_idx] = torch.bincount(
            valid_bins, 
            minlength=HISTOGRAM_NUM_POINTS
        )
    
    temp_float = HISTOGRAM_MIN + (HISTOGRAM_STEP /2.0) # start pos
    num_points = torch.arange(0, HISTOGRAM_NUM_POINTS)
    expected_survival_hist = (torch.erfc((temp_float + HISTOGRAM_STEP *num_points) / 2.0**0.5)/2.0) * (num_pixels * total_correlation_positions)
    # Calculate survival histogram (cumulative sum from right to left)
    survival_histogram = torch.flip(torch.cumsum(torch.flip(histogram_data, [1]), dim=1), [1])

    return survival_histogram, expected_survival_hist, temp_float, histogram_data

def calc_survival_histogram_single(
    projections: torch.Tensor,
    num_pixels: torch.Tensor,
    total_correlation_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Calculate the survival histogram.

    Parameters
    ----------
    projections : torch.Tensor
        The projections to calculate the survival histogram from.
    num_pixels : torch.Tensor
        The number of pixels.
    total_correlation_positions : torch.Tensor
        The total correlation positions.

    Returns
    -------
    survival_histogram : torch.Tensor
        The survival histogram.
    expected_survival_hist : torch.Tensor
        The expected survival histogram.
    temp_float : float
        The temp float.
    histogram_data : torch.Tensor
        The histogram data.
    """
    histogram_min_scaled = HISTOGRAM_MIN / num_pixels**0.5
    histogram_step_scaled = HISTOGRAM_STEP / num_pixels**0.5
    # Create histogram for each micrograph
    histogram_data = torch.zeros((HISTOGRAM_NUM_POINTS), device=projections.device)
    
    # Calculate bins using torch operations
    bins = ((projections - histogram_min_scaled) / histogram_step_scaled).long()

    # Create a mask for valid bins (between 0 and histogram_number_of_points)
    valid_mask = (bins >= 0) & (bins < HISTOGRAM_NUM_POINTS)
        
    # Use torch.bincount to count occurrences of each bin
    # Only count values where the mask is True
    valid_bins = bins[valid_mask]
    histogram_data = torch.bincount(
        valid_bins, 
        minlength=HISTOGRAM_NUM_POINTS
    )
    
    temp_float = HISTOGRAM_MIN + (HISTOGRAM_STEP /2.0) # start pos
    num_points = torch.arange(0, HISTOGRAM_NUM_POINTS)
    expected_survival_hist = (torch.erfc((temp_float + HISTOGRAM_STEP *num_points) / 2.0**0.5)/2.0) * (num_pixels * total_correlation_positions)
    # Calculate survival histogram (cumulative sum from right to left)
    survival_histogram = torch.flip(torch.cumsum(torch.flip(histogram_data, [0]), dim=0), [0])

    return survival_histogram, expected_survival_hist, temp_float, histogram_data

def process_match_results_all(
    all_inputs: dict,
    projections: torch.Tensor,
    defocus_values: torch.Tensor,
    Cs_vals: torch.Tensor,
    euler_angles: torch.Tensor,
    micrograph_data: pd.DataFrame,
) -> None:
    """
    Process the match results.

    Parameters
    ----------
    all_inputs : dict
        The all inputs dictionary.
    projections : torch.Tensor
        The projections to process.
    defocus_values : torch.Tensor
        The defocus values to process.
    Cs_vals : torch.Tensor
        The Cs values to process.
    euler_angles : torch.Tensor
        The Euler angles to process.
    micrograph_data : pd.DataFrame
        The micrograph data to process.

    Returns
    -------
    None
    """
    total_correlation_positions = torch.tensor(projections.shape[1] * projections.shape[2] * projections.shape[3], dtype=torch.float32)
    print("total correlation positions", total_correlation_positions)
    num_pixels = torch.tensor(projections.shape[-2] * projections.shape[-1], dtype=torch.float32)
    # Get the maximum intensity projection and the best defocus, Cs, and Euler angles
    best_defoc, best_pixel_size, best_phi, best_theta, best_psi, maximum_intensiy_projections = get_mip_and_best(
        projections=projections,
        defocus_values=defocus_values,
        Cs_vals=Cs_vals,
        euler_angles=euler_angles,
        micrograph_data=micrograph_data,
        num_pixels=num_pixels,
    )

    # Calculate the sum correlation
    sum_correlation, sum_correlation_squared = calc_sum_correlation(
        projections=projections,
        num_pixels=num_pixels,
        total_correlation_positions=total_correlation_positions,
    )

    print("sum correlation mean", einops.reduce(sum_correlation, "nMic h w -> nMic", "mean"))
    print("sum correlation_squared mean", einops.reduce(sum_correlation_squared, "nMic h w -> nMic", "mean"))

    # Normalize the maximum intensity projection
    maximum_intensiy_projections_normalized = normalize_mip(
        maximum_intensiy_projections=maximum_intensiy_projections,
        sum_correlation=sum_correlation,
        sum_correlation_squared=sum_correlation_squared,
    )

    print("maximum_intensiy_projections_normalized mean", einops.reduce(maximum_intensiy_projections_normalized, "nMic h w -> nMic", "mean"))

    #Output these files
    write_mrc_files(
        all_inputs=all_inputs,
        micrograph_data=micrograph_data,
        maximum_intensiy_projections=maximum_intensiy_projections,
        maximum_intensiy_projections_normalized=maximum_intensiy_projections_normalized,
        sum_correlation=sum_correlation,
        sum_correlation_squared=sum_correlation_squared,
        best_defoc=best_defoc,
        best_phi=best_phi,
        best_theta=best_theta,
        best_psi=best_psi,
        best_pixel_size=best_pixel_size,
    )

    # Calculate the survival histogram
    #Get the expected noise
    expected_noise = get_expected_noise(
        num_pixels=num_pixels,
        total_correlation_positions=total_correlation_positions,
    )
    survival_histogram, expected_survival_hist, temp_float, histogram_data = calc_survival_histogram(
        projections=maximum_intensiy_projections_normalized,
        num_pixels=num_pixels,
        total_correlation_positions=total_correlation_positions,
    )

    write_survival_histogram(
        all_inputs=all_inputs,
        survival_histogram=survival_histogram,
        nMicrographs=projections.shape[0],
        expected_noise=expected_noise,
        histogram_data=histogram_data,
        expected_survival_hist=expected_survival_hist,
        temp_float=temp_float,
        HISTOGRAM_STEP=HISTOGRAM_STEP,
        HISTOGRAM_NUM_POINTS=HISTOGRAM_NUM_POINTS,
    )

def process_match_results_single(
    all_inputs: dict,
    micrograph_number: int,
    micrograph_data: pd.DataFrame,
    maximum_intensiy_projection: torch.Tensor,
    sum_correlation: torch.Tensor,
    sum_correlation_squared: torch.Tensor,
    best_defoc: torch.Tensor,
    best_phi: torch.Tensor,
    best_theta: torch.Tensor,
    best_psi: torch.Tensor,
    best_pixel_size: torch.Tensor,
    num_pixels: torch.Tensor,
    total_correlation_positions: torch.Tensor,
) -> None:
    #finsish processing
    sum_correlation = sum_correlation / total_correlation_positions
    sum_correlation_squared = sum_correlation_squared / total_correlation_positions - (sum_correlation ** 2)
    sum_correlation_squared = torch.sqrt(torch.clamp(sum_correlation_squared, min=0)) * torch.sqrt(num_pixels)
    sum_correlation = sum_correlation * torch.sqrt(num_pixels)

    print("sum correlation mean", sum_correlation.mean())
    print("sum correlation std", sum_correlation.std())
    print("sum correlation_squared mean", sum_correlation_squared.mean())
    print("sum correlation_squared std", sum_correlation_squared.std())
    maximum_intensiy_projection_normalized = maximum_intensiy_projection - sum_correlation
    maximum_intensiy_projection_normalized = torch.where(sum_correlation_squared == 0, 
                                             torch.zeros_like(maximum_intensiy_projection_normalized),
                                             maximum_intensiy_projection_normalized / sum_correlation_squared)

    print("maximum_intensiy_projection_normalized mean", maximum_intensiy_projection_normalized.mean())
    print("maximum_intensiy_projection_normalized std", maximum_intensiy_projection_normalized.std())
    #Output these files
    write_mrc_files_single(
        all_inputs=all_inputs,
        micrograph_number=micrograph_number,
        micrograph_data=micrograph_data,
        maximum_intensiy_projections=maximum_intensiy_projection,
        maximum_intensiy_projections_normalized=maximum_intensiy_projection_normalized,
        sum_correlation=sum_correlation,
        sum_correlation_squared=sum_correlation_squared,
        best_defoc=best_defoc,
        best_phi=best_phi,
        best_theta=best_theta,
        best_psi=best_psi,
        best_pixel_size=best_pixel_size,
    )

    # Calculate the survival histogram
    #Get the expected noise
    expected_noise = get_expected_noise(
        num_pixels=num_pixels,
        total_correlation_positions=total_correlation_positions,
    )
    survival_histogram, expected_survival_hist, temp_float, histogram_data = calc_survival_histogram_single(
        projections=maximum_intensiy_projection_normalized,
        num_pixels=num_pixels,
        total_correlation_positions=total_correlation_positions,
    )

    write_survival_histogram_single(
        all_inputs=all_inputs,
        survival_histogram=survival_histogram,
        micrograph_number=micrograph_number,
        expected_noise=expected_noise,
        histogram_data=histogram_data,
        expected_survival_hist=expected_survival_hist,
        temp_float=temp_float,
        HISTOGRAM_STEP=HISTOGRAM_STEP,
        HISTOGRAM_NUM_POINTS=HISTOGRAM_NUM_POINTS,
    )








