# This will take many inputs
# mrc map, micrographs, ctf's, output dir, px size, etc.

# It will mainly follow torch 2dtm package stuff

import os

# Set the environment variable before importing PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import einops
import torch
from torch_fourier_filter.ctf import calculate_ctf_2d
from torch_fourier_filter.phase_randomize import phase_randomize

from tt2dtm.match_template.load_angles import euler_to_rotation_matrix, get_euler_angles
from tt2dtm.utils.calculate_filters import (
    calculate_2d_template_filters,
    calculate_micrograph_filters,
    combine_filters,
    get_Cs_range,
    get_defocus_range,
    get_defocus_values,
)
from tt2dtm.utils.cross_correlation_core import simple_cross_correlation, simple_cross_correlation_single
from tt2dtm.utils.fourier_slice import _sinc2, extract_fourier_slice, fft_volume
from tt2dtm.utils.image_operations import mean_zero_var_one_full_size, pad_to_shape_2d, pad_volume, edge_mean_reduction_2d
from tt2dtm.utils.io_handler import (
    load_mrc_map,
    load_mrc_micrographs,
    load_relion_starfile,
    read_inputs,
)
from tt2dtm.match_template.process_match_results import get_mip_and_best_single_mic, process_match_results_all, process_match_results_single
from tt2dtm.utils.memory_utils import calculate_batch_size, get_gpu_with_most_memory

def get_search_ranges(
    all_inputs: dict, 
    micrograph_data: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the search ranges for Cs, defocus_u, and defocus_v
    Parameters
    ----------  
    all_inputs: dict
        The input dictionary
    micrograph_data: dict
        The micrograph data dictionary

    Returns
    -------
    Cs_vals: torch.Tensor
        The search ranges for Cs
    defoc_u_vals: torch.Tensor
        The search ranges for defocus_u
    defoc_v_vals: torch.Tensor
        The search ranges for defocus_v
    """
    Cs_vals = torch.tensor(
        float(micrograph_data["rlnSphericalAberration"][0]), dtype=torch.float32
    )

    if all_inputs["extra_searches"]["pixel_size_search"]["enabled"]:
        Cs_vals = get_Cs_range(
            pixel_size=float(micrograph_data["rlnMicrographPixelSize"][0]),
            pixel_size_range=all_inputs["extra_searches"]["pixel_size_search"]["range"],
            pixel_size_step=all_inputs["extra_searches"]["pixel_size_search"][
                "step_size"
            ],
            Cs=Cs_vals,
        )
    print("Cs vals shape", Cs_vals.shape)
    # Calculate defocus range for each micrograph
    defoc_u_vals = torch.tensor(micrograph_data["rlnDefocusU"], dtype=torch.float32)
    defoc_v_vals = torch.tensor(micrograph_data["rlnDefocusV"], dtype=torch.float32)
    if all_inputs["extra_searches"]["defocus_search"]["enabled"]:
        defoc_range = get_defocus_range(
            defocus_range=float(
                all_inputs["extra_searches"]["defocus_search"]["range"]
            ),
            defocus_step=float(
                all_inputs["extra_searches"]["defocus_search"]["step_size"]
            ),
        )
        # get defoc range for each micrograph
        defoc_u_vals = get_defocus_values(
            defoc_vals=defoc_u_vals,
            defoc_range=defoc_range,
        )
        defoc_v_vals = get_defocus_values(
            defoc_vals=defoc_v_vals,
            defoc_range=defoc_range,
        )

    return Cs_vals, defoc_u_vals, defoc_v_vals

def get_ctf(
    defocus_values: torch.Tensor,
    defoc_u_vals: torch.Tensor,
    defoc_v_vals: torch.Tensor,
    micrograph_data: dict,
    Cs_vals: torch.Tensor,
    mrc_map_shape: tuple[int, int],
) -> torch.Tensor:
    astigmatism_values = (defoc_u_vals - defoc_v_vals) / 2
    astigmatism_angle = torch.tensor(
        micrograph_data["rlnDefocusAngle"], dtype=torch.float32
    )
    # broadcast with einops to match defocus values
    astigmatism_angle = einops.rearrange(astigmatism_angle, "n -> n 1")
    ctf_2d = calculate_ctf_2d(
        defocus=defocus_values * 1e-4,  # convert to microns
        astigmatism=astigmatism_values * 1e-4,  # convert to microns
        astigmatism_angle=astigmatism_angle,
        pixel_size=float(micrograph_data["rlnMicrographPixelSize"][0]),
        voltage=float(micrograph_data["rlnVoltage"][0]),
        spherical_aberration=Cs_vals,
        amplitude_contrast=float(micrograph_data["rlnAmplitudeContrast"][0]),
        b_factor=0,
        phase_shift=0,
        image_shape=mrc_map_shape[-2:],
        rfft=True,
        fftshift=False,
    )  # shape needs to be (n_micrographs, n_Cs, n_defoc, h, w)
    print("ctf shape", ctf_2d.shape)
    # if ctf only four dimensions, rearrange to 5 (add Cs dimension)
    if ctf_2d.ndim == 4:
        ctf_2d = einops.rearrange(ctf_2d, "nMic nDefoc h w -> nMic nDefoc 1 h w")
    print("ctf shape", ctf_2d.shape)
    return ctf_2d

def get_ctf_single(
    defocus_values: torch.Tensor,
    defoc_u_vals: torch.Tensor,
    defoc_v_vals: torch.Tensor,
    micrograph_data: dict,
    Cs_vals: torch.Tensor,
    mrc_map_shape: tuple[int, int],
) -> torch.Tensor:
    astigmatism_values = (defoc_u_vals - defoc_v_vals) / 2
    astigmatism_angle = torch.tensor(
        micrograph_data["rlnDefocusAngle"], dtype=torch.float32
    )
    ctf_2d = calculate_ctf_2d(
        defocus=defocus_values * 1e-4,  # convert to microns
        astigmatism=astigmatism_values * 1e-4,  # convert to microns
        astigmatism_angle=astigmatism_angle,
        pixel_size=float(micrograph_data["rlnMicrographPixelSize"]),
        voltage=float(micrograph_data["rlnVoltage"]),
        spherical_aberration=Cs_vals,
        amplitude_contrast=float(micrograph_data["rlnAmplitudeContrast"]),
        b_factor=0,
        phase_shift=0,
        image_shape=mrc_map_shape[-2:],
        rfft=True,
        fftshift=False,
    )  # shape needs to be (n_micrographs, n_Cs, n_defoc, h, w)
    print("ctf shape", ctf_2d.shape)
    #Remove first nMic dimension
    ctf_2d = ctf_2d[0]
    print("ctf shape", ctf_2d.shape)
    # if ctf only four dimensions, rearrange to 4 (add Cs dimension)
    if ctf_2d.ndim == 3:
        ctf_2d = einops.rearrange(ctf_2d, "nDefoc h w -> nDefoc 1 h w")
    print("ctf shape", ctf_2d.shape)
    return ctf_2d


def run_tm(
    input_yaml: str
) -> None:
    """
    Run the template matching pipeline

    Parameters
    ----------
    input_yaml: str
        The input yaml file
    """
    # read the input yaml file
    print("Reading input yaml")
    all_inputs = read_inputs(input_yaml)

    ####LOAD THE DATA####
    # Get the micrograph list from relion starfile (add other options later)
    micrograph_data = load_relion_starfile(
        all_inputs["input_files"]["micrograph_ctf_star"]
    )
    # Load the micrographs
    micrographs = load_mrc_micrographs(micrograph_data["rlnMicrographName"])
    # Load the mrc template map or pdb
    mrc_map = load_mrc_map(all_inputs["input_files"]["mrc_map"])
    # if load pdb call sim3d, calc output size before that to be as small as possible

    # pad micrographs and mrc map up to nearest radix of 2, 3, 5, 7, 11, 13

    # if pad true, double size of volume for slice extraction
    # and filter multiplication
    if all_inputs["filters"]["pad"]["enabled"]:
        mrc_map = pad_volume(mrc_map, pad_length=mrc_map.shape[-1] // 2)

    ####ANGLE OPERATIONS####
    # Load the angles, based on symmetry and ranges, steps, etc
    print("Calculating rotation matrices")
    euler_angles = get_euler_angles(all_inputs)
    rotation_matrices = euler_to_rotation_matrix(euler_angles)

    euler_angles = torch.stack(euler_angles, dim=0).squeeze(0)
    print(euler_angles[0].shape)
    print(len(euler_angles))

    ####CALC FILTERS####
    # dft the micrographs and keep them like this. zero mean
    dft_micrographs = torch.fft.rfftn(micrographs, dim=(-2, -1))

    # calc whitening and any other filters for micrograph
    whiten_micrograph, bandpass_micrograph = calculate_micrograph_filters(
        all_inputs=all_inputs,
        micrograph_data=micrograph_data,
        dft_micrographs=dft_micrographs,
        micrograph_shape=micrographs.shape[-2:],
    )
    # combine filters together
    combined_micrograph_filter = combine_filters(whiten_micrograph, bandpass_micrograph)

    # calc 2D whitening and any others for template
    # one for each micrograph/ multiply filters together
    whiten_template, bandpass_template = calculate_2d_template_filters(
        all_inputs=all_inputs,
        micrograph_data=micrograph_data,
        dft_micrographs=dft_micrographs,
        micrograph_shape=micrographs.shape[-2:],
        template_shape=mrc_map.shape,
    )
    combined_template_filter = combine_filters(whiten_template, bandpass_template)
    print("combined template filter shape", combined_template_filter.shape)



    ####MICROGRAPH OPERATIONS####
    #Make sure mean zero after calc whiten to avoid div by zero
    dft_micrographs[:, 0, 0] = 0 + 0j
    # Apply the filter to the micrographs and phase random if wanted
    dft_micrographs_filtered = dft_micrographs * combined_micrograph_filter
    if all_inputs["filters"]["phase_randomize"]["enabled"]:
        cuton = float(all_inputs["filters"]["phase_randomize"]["cuton_resolution"])
        dft_micrographs_filtered = phase_randomize(
            dft=dft_micrographs_filtered,
            image_shape=micrographs.shape[-2:],
            rfft=True,
            cuton=cuton,
            fftshift=False,
        )
    # zero central pixel
    dft_micrographs_filtered[:, 0, 0] = 0 + 0j
    # divide by sqrt sum of squares.
    dft_micrographs_filtered /= torch.sqrt(
        torch.sum(torch.abs(dft_micrographs_filtered) ** 2, dim=(-2, -1), keepdim=True)
    )
    print("dft micrographs filtered shape", dft_micrographs_filtered.shape)

    ####TEMPLATE STUFF####
    # Calculate range of Cs values of pixel size search
    # Cs_vals = torch.tensor(float(micrograph_data['rlnSphericalAberration'][0]))
    Cs_vals, defoc_u_vals, defoc_v_vals = get_search_ranges(
        all_inputs=all_inputs,
        micrograph_data=micrograph_data,
    )
    defocus_values = (defoc_u_vals + defoc_v_vals) / 2
    print("defocus shapes", defoc_u_vals.shape, defoc_v_vals.shape)

    # Calculate size needed for 2D projection based on Cs/CTF and nyquist
    # Can only do this if we have the pdb

    ctf_2d = get_ctf(
        defocus_values=defocus_values,
        defoc_u_vals=defoc_u_vals,
        defoc_v_vals=defoc_v_vals,
        micrograph_data=micrograph_data,
        Cs_vals=Cs_vals,
        mrc_map_shape=mrc_map.shape,
    )
    # multiply ctf by filters
    combined_template_filter = einops.rearrange(
        combined_template_filter, "nMic h w -> nMic 1 1 h w"
    )
    combined_template_filter = combined_template_filter * ctf_2d
    print("combined template filter shape", combined_template_filter.shape)


    #### Extract Fourier slice at angles ####
    # Multiply map by sinc2
    mrc_map = mrc_map * _sinc2(shape=mrc_map.shape, rfft=False, fftshift=True)

    dft_map = fft_volume(
        volume=mrc_map,
        fftshift=True,  # Needed for slice extraction
    )
    print("calc projections")
    projections = extract_fourier_slice(
        dft_volume=dft_map,
        rotation_matrices=rotation_matrices,
        volume_shape=mrc_map.shape,
    )  # shape (n_angles, h, w)
    print("done projections")
    print("projections shape", projections.shape)

    # Apply the combined filters to projections
    #the final shape will now be (n_micrographs, n_defoc, n_Cs, n_angles, h, w)
    projections = einops.rearrange(projections, "nAng h w -> 1 1 1 nAng h w")
    combined_template_filter = einops.rearrange(
        combined_template_filter, "nMic nDefoc nCs h w -> nMic nDefoc nCs 1 h w"
    )
    projections = projections * combined_template_filter
    print("projections shape", projections.shape)
    print("projections filtered")

    # Backwards FFT
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    #flip contrast
    projections = projections * -1
    #. Subtract mean of edge (mean 0) and set variance 1 (full size)
    projections = mean_zero_var_one_full_size(
        projections=projections,
        micrographs_shape=micrographs.shape[-2:],
    )
    # Pad projections with zeros to size of micrographs
    projections = pad_to_shape_2d(
        image=projections,
        image_shape=projections.shape[-2:],
        shape=micrographs.shape[-2:],
        pad_val=0,
    )
    print("projections padded shape", projections.shape)
    # cross correlation
    projections = simple_cross_correlation(
        projections=projections,
        dft_micrographs_filtered=dft_micrographs_filtered,
    )
    print(f"mean proj = {einops.reduce(projections, "nMic nDefoc nCs nAng h w -> nMic", "mean")}")
    
    process_match_results_all(
        all_inputs=all_inputs,
        projections=projections,
        defocus_values=defocus_values,
        Cs_vals=Cs_vals,
        euler_angles=euler_angles,
        micrograph_data=micrograph_data,
    )

def run_tm_cpu_batch(input_yaml: str):
    # read the input yaml file
    print("Reading input yaml")
    all_inputs = read_inputs(input_yaml)

    ####LOAD THE DATA####
    # Get the micrograph list from relion starfile (add other options later)
    micrograph_data = load_relion_starfile(
        all_inputs["input_files"]["micrograph_ctf_star"]
    )
    # Load the micrographs to cpu
    micrographs = load_mrc_micrographs(micrograph_data["rlnMicrographName"])
    # Load the mrc template map or pdb
    mrc_map = load_mrc_map(all_inputs["input_files"]["mrc_map"])
    # if load pdb call sim3d, calc output size before that to be as small as possible

    # if pad true, double size of volume for slice extraction
    # and filter multiplication
    if all_inputs["filters"]["pad"]["enabled"]:
        mrc_map = pad_volume(mrc_map, pad_length=mrc_map.shape[-1] // 2)

    ####ANGLE OPERATIONS####
    # Load the angles, based on symmetry and ranges, steps, etc
    print("Calculating rotation matrices")
    euler_angles = get_euler_angles(all_inputs)
    rotation_matrices = euler_to_rotation_matrix(euler_angles)

    euler_angles = torch.stack(euler_angles, dim=0).squeeze(0)
    print(euler_angles[0].shape)
    print(len(euler_angles))

    ### SEARCH RANGES ###
    Cs_vals, defoc_u_vals, defoc_v_vals = get_search_ranges(
        all_inputs=all_inputs,
        micrograph_data=micrograph_data,
    )
    defocus_values = (defoc_u_vals + defoc_v_vals) / 2
    print("defocus shapes", defoc_u_vals.shape, defoc_v_vals.shape)

    ####TEMPLATE STUFF####
    mrc_map = mrc_map * _sinc2(shape=mrc_map.shape, rfft=False, fftshift=True)

    dft_map = fft_volume(
        volume=mrc_map,
        fftshift=True,  # Needed for slice extraction
    )

    #Do one micrograph at a time

    for i, micrograph in enumerate(micrographs):
        # Do on CPU until get to ctf calc and fourier slice
        this_ctf = get_ctf_single(
            defocus_values=defocus_values[i],
            defoc_u_vals=defoc_u_vals[i],
            defoc_v_vals=defoc_v_vals[i],
            micrograph_data=micrograph_data.iloc[i],
            Cs_vals=Cs_vals.clone(), # prevent converting to A
            mrc_map_shape=mrc_map.shape,
        )
        print("this ctf shape", this_ctf.shape)
        ###Calc Filters###
        dft_micrograph = torch.fft.rfftn( micrograph, dim=(-2, -1))
        # calc whitening and any other filters for micrograph
        whiten_micrograph, bandpass_micrograph = calculate_micrograph_filters(
            all_inputs=all_inputs,
            micrograph_data=micrograph_data,
            dft_micrographs=dft_micrograph,
            micrograph_shape=micrograph.shape[-2:],
        )
         # combine filters together
        combined_micrograph_filter = combine_filters(whiten_micrograph, bandpass_micrograph)

        # calc 2D whitening and any others for template
        # one for each micrograph/ multiply filters together
        whiten_template, bandpass_template = calculate_2d_template_filters(
            all_inputs=all_inputs,
            micrograph_data=micrograph_data,
            dft_micrographs=dft_micrograph,
            micrograph_shape=micrograph.shape[-2:],
            template_shape=mrc_map.shape,
        )
        combined_template_filter = combine_filters(whiten_template, bandpass_template)
        print("combined template filter shape", combined_template_filter.shape)
        ###Micrograph operations###
        #Make sure mean zero after calc whiten to avoid div by zero
        dft_micrograph[0, 0] = 0 + 0j
        # Apply the filter to the micrographs and phase random if wanted
        dft_micrograph = dft_micrograph * combined_micrograph_filter
        if all_inputs["filters"]["phase_randomize"]["enabled"]:
            cuton = float(all_inputs["filters"]["phase_randomize"]["cuton_resolution"])
            dft_micrograph = phase_randomize(
                dft=dft_micrograph,
                image_shape=micrograph.shape[-2:],
                rfft=True,
                cuton=cuton,
                fftshift=False,
            )
        # zero central pixel
        dft_micrograph[0, 0] = 0 + 0j
        # divide by sqrt sum of squares.
        dft_micrograph /= torch.sqrt(
            torch.sum(torch.abs(dft_micrograph) ** 2, dim=(-2, -1), keepdim=True)
        )

        #convert dft_micrograph to complex64
        #dft_micrograph = dft_micrograph.to(torch.complex64)
        ###Template stuff###
        # multiply ctf by filters
        combined_template_filter = einops.rearrange(
            combined_template_filter, "h w -> 1 1 h w"
        )
        combined_template_filter = combined_template_filter * this_ctf

        #now batch the angles
        #batch_size = 10  # Can be tuned based on memory
        device = torch.device('cpu')

        batch_size = calculate_batch_size(
            micrograph_shape=micrograph.shape[-2:],
            defoc_vals=defocus_values[i],
            Cs_vals=Cs_vals,
            device=device,
        )
        if batch_size <= 1:
            batch_size = 2
        print("batch size", batch_size)
        num_angles = rotation_matrices.shape[0]
        num_batches = (num_angles + batch_size - 1) // batch_size

        sum_correlation = torch.zeros(micrograph.shape[-2:], dtype=torch.float64)
        sum_correlation_squared = torch.zeros(micrograph.shape[-2:], dtype=torch.float64)
        best_defoc = torch.zeros(micrograph.shape[-2:], dtype=torch.float32)
        best_pixel_size = torch.zeros(micrograph.shape[-2:], dtype=torch.float32)
        best_phi = torch.zeros(micrograph.shape[-2:], dtype=torch.float32)
        best_theta = torch.zeros(micrograph.shape[-2:], dtype=torch.float32)
        best_psi = torch.zeros(micrograph.shape[-2:], dtype=torch.float32)
        maximum_intensiy_projection = torch.zeros(micrograph.shape[-2:], dtype=torch.float32)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_angles)
            batch_matrices = rotation_matrices[start_idx:end_idx]
            batch_euler_angles = euler_angles[start_idx:end_idx]
            proj = extract_fourier_slice(
                dft_volume=dft_map,
                rotation_matrices=batch_matrices,
                volume_shape=mrc_map.shape,
            )
            # Rearrange and multiply with template filter
            proj = einops.rearrange(proj, "nAng h w -> 1 1 nAng h w")
            combined_template_filter = einops.rearrange(combined_template_filter, "nDefoc nCs h w -> nDefoc nCs 1 h w")
            proj = proj * combined_template_filter
                
            # Backwards FFT
            proj = torch.fft.irfftn(proj, dim=(-2, -1))
                
            # Flip contrast
            proj = proj * -1
                
            # Mean zero var one
            proj = mean_zero_var_one_full_size(
                projections=proj,
                micrographs_shape=micrograph.shape[-2:],
            )

            #convert proj to float32
            #proj = proj.to(torch.float32)
            # Pad projections
            proj = pad_to_shape_2d(
                image=proj,
                image_shape=proj.shape[-2:],
                shape=micrograph.shape[-2:],
                pad_val=0,
            )
                
            # Cross correlation
            proj = simple_cross_correlation_single(
                projections=proj,
                dft_micrographs_filtered=dft_micrograph,
            )
            print(f"mean proj = {proj.mean()}")
            #Need to process each batch to get best and max, update sum corr and histogram
            sum_correlation = sum_correlation + einops.reduce(proj, "nDefoc nCs nAng h w -> h w", "sum")
            sum_correlation_squared = sum_correlation_squared + einops.reduce(proj ** 2, "nDefoc nCs nAng h w -> h w", "sum")
            #Get best and max
            this_best_defoc, this_best_pixel_size, this_best_phi, this_best_theta, this_best_psi, this_maximum_intensiy_projection = get_mip_and_best_single_mic(
                projections=proj,
                defocus_values=defocus_values[i],
                Cs_vals=Cs_vals,
                euler_angles=batch_euler_angles,
                micrograph_data=micrograph_data.iloc[i],
            )
            # Update best values where this batch has higher correlation
            mask = this_maximum_intensiy_projection >= maximum_intensiy_projection
            maximum_intensiy_projection[mask] = this_maximum_intensiy_projection[mask]
            best_defoc[mask] = this_best_defoc[mask]
            best_pixel_size[mask] = this_best_pixel_size[mask]
            best_phi[mask] = this_best_phi[mask]
            best_theta[mask] = this_best_theta[mask]
            best_psi[mask] = this_best_psi[mask]

        #finsish processing
        num_pixels = torch.tensor(micrograph.shape[-2] * micrograph.shape[-1], dtype=torch.float32)
        total_correlation_positions = torch.tensor(defocus_values[i].shape[0] * Cs_vals.shape[0] * num_angles, dtype=torch.float32)
        print("total correlation positions", total_correlation_positions)
        maximum_intensiy_projection = maximum_intensiy_projection * torch.sqrt(num_pixels)

        process_match_results_single(
            all_inputs=all_inputs,
            micrograph_number=i,
            micrograph_data=micrograph_data.iloc[i],
            maximum_intensiy_projection=maximum_intensiy_projection,
            sum_correlation=sum_correlation,
            sum_correlation_squared=sum_correlation_squared,
            best_defoc=best_defoc,
            best_pixel_size=best_pixel_size,
            best_phi=best_phi,
            best_theta=best_theta,
            best_psi=best_psi,
            num_pixels=num_pixels,
            total_correlation_positions=total_correlation_positions,
        )

def run_tm_gpu_batch_1(input_yaml: str):
    # read the input yaml file
    print("Reading input yaml")
    all_inputs = read_inputs(input_yaml)

    ####LOAD THE DATA####
    # Get the micrograph list from relion starfile (add other options later)
    micrograph_data = load_relion_starfile(
        all_inputs["input_files"]["micrograph_ctf_star"]
    )
    # Load the micrographs to cpu
    micrographs = load_mrc_micrographs(micrograph_data["rlnMicrographName"])
    # Load the mrc template map or pdb
    mrc_map = load_mrc_map(all_inputs["input_files"]["mrc_map"])
    # if load pdb call sim3d, calc output size before that to be as small as possible

    # if pad true, double size of volume for slice extraction
    # and filter multiplication
    if all_inputs["filters"]["pad"]["enabled"]:
        mrc_map = pad_volume(mrc_map, pad_length=mrc_map.shape[-1] // 2)

    ####ANGLE OPERATIONS####
    # Load the angles, based on symmetry and ranges, steps, etc
    print("Calculating rotation matrices")
    euler_angles = get_euler_angles(all_inputs)
    rotation_matrices = euler_to_rotation_matrix(euler_angles)

    euler_angles = torch.stack(euler_angles, dim=0).squeeze(0)
    print(euler_angles[0].shape)
    print(len(euler_angles))

    ### SEARCH RANGES ###
    Cs_vals, defoc_u_vals, defoc_v_vals = get_search_ranges(
        all_inputs=all_inputs,
        micrograph_data=micrograph_data,
    )
    defocus_values = (defoc_u_vals + defoc_v_vals) / 2
    print("defocus shapes", defoc_u_vals.shape, defoc_v_vals.shape)

    ####TEMPLATE STUFF####
    mrc_map = mrc_map * _sinc2(shape=mrc_map.shape, rfft=False, fftshift=True)

    dft_map = fft_volume(
        volume=mrc_map,
        fftshift=True,  # Needed for slice extraction
    )

    #Do one micrograph at a time

    for i, micrograph in enumerate(micrographs):
        # Do on CPU until get to ctf calc and fourier slice
        this_ctf = get_ctf_single(
            defocus_values=defocus_values[i],
            defoc_u_vals=defoc_u_vals[i],
            defoc_v_vals=defoc_v_vals[i],
            micrograph_data=micrograph_data.iloc[i],
            Cs_vals=Cs_vals.clone(), # prevent converting to A
            mrc_map_shape=mrc_map.shape,
        )
        print("this ctf shape", this_ctf.shape)
        ###Calc Filters###
        dft_micrograph = torch.fft.rfftn( micrograph, dim=(-2, -1))
        # calc whitening and any other filters for micrograph
        whiten_micrograph, bandpass_micrograph = calculate_micrograph_filters(
            all_inputs=all_inputs,
            micrograph_data=micrograph_data,
            dft_micrographs=dft_micrograph,
            micrograph_shape=micrograph.shape[-2:],
        )
         # combine filters together
        combined_micrograph_filter = combine_filters(whiten_micrograph, bandpass_micrograph)

        # calc 2D whitening and any others for template
        # one for each micrograph/ multiply filters together
        whiten_template, bandpass_template = calculate_2d_template_filters(
            all_inputs=all_inputs,
            micrograph_data=micrograph_data,
            dft_micrographs=dft_micrograph,
            micrograph_shape=micrograph.shape[-2:],
            template_shape=mrc_map.shape,
        )
        combined_template_filter = combine_filters(whiten_template, bandpass_template)
        print("combined template filter shape", combined_template_filter.shape)
        ###Micrograph operations###
        #Make sure mean zero after calc whiten to avoid div by zero
        dft_micrograph[0, 0] = 0 + 0j
        # Apply the filter to the micrographs and phase random if wanted
        dft_micrograph = dft_micrograph * combined_micrograph_filter
        if all_inputs["filters"]["phase_randomize"]["enabled"]:
            cuton = float(all_inputs["filters"]["phase_randomize"]["cuton_resolution"])
            dft_micrograph = phase_randomize(
                dft=dft_micrograph,
                image_shape=micrograph.shape[-2:],
                rfft=True,
                cuton=cuton,
                fftshift=False,
            )
        # zero central pixel
        dft_micrograph[0, 0] = 0 + 0j
        # divide by sqrt sum of squares.
        dft_micrograph /= torch.sqrt(
            torch.sum(torch.abs(dft_micrograph) ** 2, dim=(-2, -1), keepdim=True)
        )

        #convert dft_micrograph to complex64
        #dft_micrograph = dft_micrograph.to(torch.complex64)
        ###Template stuff###
        # multiply ctf by filters
        combined_template_filter = einops.rearrange(
            combined_template_filter, "h w -> 1 1 h w"
        )
        combined_template_filter = combined_template_filter * this_ctf

        #now batch the angles
        #batch_size = 10  # Can be tuned based on memory

        device = get_gpu_with_most_memory()

        # = torch.device('cpu')

        # Move input tensors to GPU
        defocus_values_gpu = defocus_values[i].to(device)
        Cs_vals_gpu = Cs_vals.to(device)
        euler_angles_gpu = euler_angles.to(device)

        batch_size = calculate_batch_size(
            micrograph_shape=micrograph.shape[-2:],
            defoc_vals=defocus_values[i],
            Cs_vals=Cs_vals,
            device=device,
        )
        if batch_size <= 1:
            batch_size = 2
        print("batch size", batch_size)
        num_angles = rotation_matrices.shape[0]
        num_batches = (num_angles + batch_size - 1) // batch_size

        sum_correlation = torch.zeros(micrograph.shape[-2:], dtype=torch.float64, device=device)
        sum_correlation_squared = torch.zeros(micrograph.shape[-2:], dtype=torch.float64, device=device)
        best_defoc = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
        best_pixel_size = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
        best_phi = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
        best_theta = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
        best_psi = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
        maximum_intensiy_projection = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)

        combined_template_filter = einops.rearrange(combined_template_filter, "nDefoc nCs h w -> nDefoc nCs 1 h w").to(device)

        # Move dft_map to device
        dft_map = dft_map.to(device)
        mrc_map_shape = torch.tensor(mrc_map.shape, device=device)
        dft_micrograph = dft_micrograph.to(device)

        start_mic_time = time.time()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_angles)
            batch_matrices = rotation_matrices[start_idx:end_idx].to(device)
            batch_euler_angles = euler_angles[start_idx:end_idx].to(device)

            
            proj = extract_fourier_slice(
                dft_volume=dft_map,
                rotation_matrices=batch_matrices,
                volume_shape=mrc_map_shape,
            )
            # Rearrange and multiply with template filter
            proj = einops.rearrange(proj, "nAng h w -> 1 1 nAng h w")
            proj = proj * combined_template_filter

            #print memoery size of proj
            #print(f"elem size {proj.element_size()}")
            #print(f"numel {proj.numel() / (micrograph.shape[-2] * micrograph.shape[-1])} ")
            #print("proj size", (proj.numel() * proj.element_size()) / 1024**3)
            #print proj data type
            #print("proj data type", proj.dtype)
            #print(f"proj shape {proj.shape}")
                
            # Backwards FFT
            proj = torch.fft.irfftn(proj, dim=(-2, -1))

            #print memoery size of proj
            #print("proj size", (proj.numel() * proj.element_size()) / 1024**3)
            #print proj data type
            #print("proj data type", proj.dtype)
            #print(f"proj shape {proj.shape}")

            # Flip contrast
            proj = proj * -1
                
            # Mean zero var one            print("proj size", (proj.numel() * proj.element_size()) / 1024**3)
            proj = mean_zero_var_one_full_size(
                projections=proj,
                micrographs_shape=micrograph.shape[-2:],
            )

            #convert proj to float32
            #proj = proj.to(torch.float32)
            # Pad projections
            proj = pad_to_shape_2d(
                image=proj,
                image_shape=proj.shape[-2:],
                shape=micrograph.shape[-2:],
                pad_val=0,
            )

            #print memoery size of proj
            #print("proj size", (proj.numel() * proj.element_size()) / 1024**3)
            #print proj data type
            #print("proj data type", proj.dtype)
            #print(f"proj shape {proj.shape}")
                
            # Cross correlation
            '''
            proj = simple_cross_correlation_single(
                projections=proj,
                dft_micrographs_filtered=dft_micrograph.to(device),
                device=device,
            )
            '''


            def print_memory_usage(step):
                #torch.cuda.synchronize(device)  # Ensure all operations are complete on the specified device
                print(f"{step}: Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB")


            
            #Do this here instead have having in place operation memory issues

            #print_memory_usage("Before fftshift")
            proj = torch.fft.fftshift(proj, dim=(-2, -1))
            #print_memory_usage("After fftshift")
            proj = torch.fft.rfftn(proj, dim=(-2, -1))
            #print_memory_usage("After rfftn")
            proj[:, 0, 0] = 0 + 0j
            #print_memory_usage("After zeroing central pixel")
            proj = proj.conj()
            #print_memory_usage("After conj")
            proj = proj * dft_micrograph
            #print_memory_usage("After multiplying by dft_micrograph")
            proj = torch.fft.irfftn(proj, dim=(-2, -1))
            #print_memory_usage("After irfftn")

            

















            #print memoery size of proj
            #print("proj size", (proj.numel() * proj.element_size()) / 1024**3)
            #print proj data type
            #print("proj data type", proj.dtype)
            #print(f"proj shape {proj.shape}")

            #print(f"mean proj = {proj.mean()}")
            #Need to process each batch to get best and max, update sum corr and histogram
            sum_correlation = sum_correlation + einops.reduce(proj, "nDefoc nCs nAng h w -> h w", "sum")
            sum_correlation_squared = sum_correlation_squared + einops.reduce(proj ** 2, "nDefoc nCs nAng h w -> h w", "sum")
            #Get best and max
            this_best_defoc, this_best_pixel_size, this_best_phi, this_best_theta, this_best_psi, this_maximum_intensiy_projection = get_mip_and_best_single_mic(
                projections=proj,
                defocus_values=defocus_values_gpu,  # Use GPU tensor
                Cs_vals=Cs_vals_gpu,  # Use GPU tensor
                euler_angles=batch_euler_angles,  # Already on GPU
                micrograph_data=micrograph_data.iloc[i],
            )
            # Update best values where this batch has higher correlation
            mask = this_maximum_intensiy_projection >= maximum_intensiy_projection
            maximum_intensiy_projection[mask] = this_maximum_intensiy_projection[mask]
            best_defoc[mask] = this_best_defoc[mask]
            best_pixel_size[mask] = this_best_pixel_size[mask]
            best_phi[mask] = this_best_phi[mask]
            best_theta[mask] = this_best_theta[mask]
            best_psi[mask] = this_best_psi[mask]

        torch.cuda.synchronize()    


        end_mic_time = time.time()
        print(f"Time taken for micrograph {i}: {end_mic_time - start_mic_time:.2f} seconds")
        num_proj = defocus_values.shape[0] * Cs_vals.shape[0] * num_angles
        print(f"Time taken per 1000 projections: {(end_mic_time - start_mic_time) / num_proj * 1000:.2f} seconds")
        #Move everything back to cpu
        sum_correlation = sum_correlation.cpu()
        sum_correlation_squared = sum_correlation_squared.cpu()
        best_defoc = best_defoc.cpu()
        best_pixel_size = best_pixel_size.cpu()
        best_phi = best_phi.cpu()
        best_theta = best_theta.cpu()
        best_psi = best_psi.cpu()
        maximum_intensiy_projection = maximum_intensiy_projection.cpu()

        #finsish processing
        num_pixels = torch.tensor(micrograph.shape[-2] * micrograph.shape[-1], dtype=torch.float32)
        total_correlation_positions = torch.tensor(defocus_values[i].shape[0] * Cs_vals.shape[0] * num_angles, dtype=torch.float32)
        print("total correlation positions", total_correlation_positions)
        maximum_intensiy_projection = maximum_intensiy_projection * torch.sqrt(num_pixels)

        process_match_results_single(
            all_inputs=all_inputs,
            micrograph_number=i,
            micrograph_data=micrograph_data.iloc[i],
            maximum_intensiy_projection=maximum_intensiy_projection,
            sum_correlation=sum_correlation,
            sum_correlation_squared=sum_correlation_squared,
            best_defoc=best_defoc,
            best_pixel_size=best_pixel_size,
            best_phi=best_phi,
            best_theta=best_theta,
            best_psi=best_psi,
            num_pixels=num_pixels,
            total_correlation_positions=total_correlation_positions,
        )




def run_tm_gpu_batch_2(input_yaml: str):
    # 
    # read the input yaml file
    print("Reading input yaml")
    all_inputs = read_inputs(input_yaml)

    ####LOAD THE DATA####
    # Get the micrograph list from relion starfile (add other options later)
    micrograph_data = load_relion_starfile(
        all_inputs["input_files"]["micrograph_ctf_star"]
    )
    # Load the micrographs to cpu
    micrographs = load_mrc_micrographs(micrograph_data["rlnMicrographName"])
    # Load the mrc template map or pdb
    mrc_map = load_mrc_map(all_inputs["input_files"]["mrc_map"])
    # if load pdb call sim3d, calc output size before that to be as small as possible

    # if pad true, double size of volume for slice extraction
    # and filter multiplication
    if all_inputs["filters"]["pad"]["enabled"]:
        mrc_map = pad_volume(mrc_map, pad_length=mrc_map.shape[-1] // 2)

    ####ANGLE OPERATIONS####
    # Load the angles, based on symmetry and ranges, steps, etc
    print("Calculating rotation matrices")
    euler_angles = get_euler_angles(all_inputs)
    rotation_matrices = euler_to_rotation_matrix(euler_angles)

    euler_angles = torch.stack(euler_angles, dim=0).squeeze(0)
    print(euler_angles[0].shape)
    print(len(euler_angles))

    ### SEARCH RANGES ###
    Cs_vals, defoc_u_vals, defoc_v_vals = get_search_ranges(
        all_inputs=all_inputs,
        micrograph_data=micrograph_data,
    )
    defocus_values = (defoc_u_vals + defoc_v_vals) / 2
    print("defocus shapes", defoc_u_vals.shape, defoc_v_vals.shape)

    ####TEMPLATE STUFF####
    mrc_map = mrc_map * _sinc2(shape=mrc_map.shape, rfft=False, fftshift=True)

    dft_map = fft_volume(
        volume=mrc_map,
        fftshift=True,  # Needed for slice extraction
    )

    #Do one micrograph at a time

    for i, micrograph in enumerate(micrographs):
        # Do on CPU until get to ctf calc and fourier slice
        this_ctf = get_ctf_single(
            defocus_values=defocus_values[i],
            defoc_u_vals=defoc_u_vals[i],
            defoc_v_vals=defoc_v_vals[i],
            micrograph_data=micrograph_data.iloc[i],
            Cs_vals=Cs_vals.clone(), # prevent converting to A
            mrc_map_shape=mrc_map.shape,
        )
        print("this ctf shape", this_ctf.shape)
        ###Calc Filters###
        dft_micrograph = torch.fft.rfftn( micrograph, dim=(-2, -1))
        # calc whitening and any other filters for micrograph
        whiten_micrograph, bandpass_micrograph = calculate_micrograph_filters(
            all_inputs=all_inputs,
            micrograph_data=micrograph_data,
            dft_micrographs=dft_micrograph,
            micrograph_shape=micrograph.shape[-2:],
        )
         # combine filters together
        combined_micrograph_filter = combine_filters(whiten_micrograph, bandpass_micrograph)

        # calc 2D whitening and any others for template
        # one for each micrograph/ multiply filters together
        whiten_template, bandpass_template = calculate_2d_template_filters(
            all_inputs=all_inputs,
            micrograph_data=micrograph_data,
            dft_micrographs=dft_micrograph,
            micrograph_shape=micrograph.shape[-2:],
            template_shape=mrc_map.shape,
        )
        combined_template_filter = combine_filters(whiten_template, bandpass_template)
        print("combined template filter shape", combined_template_filter.shape)
        ###Micrograph operations###
        #Make sure mean zero after calc whiten to avoid div by zero
        dft_micrograph[0, 0] = 0 + 0j
        # Apply the filter to the micrographs and phase random if wanted
        dft_micrograph = dft_micrograph * combined_micrograph_filter
        if all_inputs["filters"]["phase_randomize"]["enabled"]:
            cuton = float(all_inputs["filters"]["phase_randomize"]["cuton_resolution"])
            dft_micrograph = phase_randomize(
                dft=dft_micrograph,
                image_shape=micrograph.shape[-2:],
                rfft=True,
                cuton=cuton,
                fftshift=False,
            )
        # zero central pixel
        dft_micrograph[0, 0] = 0 + 0j
        # divide by sqrt sum of squares.
        dft_micrograph /= torch.sqrt(
            torch.sum(torch.abs(dft_micrograph) ** 2, dim=(-2, -1), keepdim=True)
        )

        #convert dft_micrograph to complex64
        #dft_micrograph = dft_micrograph.to(torch.complex64)
        ###Template stuff###
        # multiply ctf by filters
        combined_template_filter = einops.rearrange(
            combined_template_filter, "h w -> 1 1 h w"
        )
        combined_template_filter = combined_template_filter * this_ctf

        #now batch the angles
        #batch_size = 10  # Can be tuned based on memory

        device = get_gpu_with_most_memory()
        # = torch.device('cpu')

        # Move input tensors to GPU
        defocus_values_gpu = defocus_values[i].to(device)
        Cs_vals_gpu = Cs_vals.to(device)
        euler_angles_gpu = euler_angles.to(device)

        nAng_batch_size = calculate_batch_size(
            micrograph_shape=micrograph.shape[-2:],
            defoc_vals=defocus_values[i],
            Cs_vals=Cs_vals,
            device=device,
        )

        # Number of streams for nDefoc * nCs
        nAng = rotation_matrices.shape[0]
        nDefoc = defocus_values[i].shape[0]
        nCs = Cs_vals.shape[0]
        num_streams = nDefoc * nCs

        streams = [torch.cuda.Stream() for _ in range(num_streams)]

        print("nAng batch size", nAng_batch_size)
        num_batches = (nAng + nAng_batch_size - 1) // nAng_batch_size

        # Create separate tensors for each stream
        stream_tensors = [{
            'sum_correlation': torch.zeros(micrograph.shape[-2:], dtype=torch.float64, device=device),
            'sum_correlation_squared': torch.zeros(micrograph.shape[-2:], dtype=torch.float64, device=device),
            'best_defoc': torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device),
            'best_pixel_size': torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device),
            'best_phi': torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device),
            'best_theta': torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device),
            'best_psi': torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device),
            'maximum_intensity_projection': torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
        } for _ in range(num_streams)]
        sum_correlation = torch.zeros(micrograph.shape[-2:], dtype=torch.float64, device=device)
        sum_correlation_squared = torch.zeros(micrograph.shape[-2:], dtype=torch.float64, device=device)
        best_defoc = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
        best_pixel_size = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
        best_phi = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
        best_theta = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
        best_psi = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
        maximum_intensiy_projection = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)

        combined_template_filter = einops.rearrange(combined_template_filter, "nDefoc nCs h w -> nDefoc nCs 1 h w").to(device)
        # Move dft_map to device
        dft_map = dft_map.to(device)
        mrc_map_shape = torch.tensor(mrc_map.shape, device=device)
        # Iterate over angle batches
        for batch_start in range(0, nAng, nAng_batch_size):
            batch_end = min(batch_start + nAng_batch_size, nAng)
            angle_batch_size = batch_end - batch_start

            # Prepare batch-specific data
            batch_matrices = rotation_matrices[batch_start:batch_end].to(device)
            batch_euler_angles = euler_angles[batch_start:batch_end].to(device)

            # Launch parallel streams over nDefoc * nCs
            for defoc_idx in range(nDefoc):
                for cs_idx in range(nCs):
                    stream_id = defoc_idx * nCs + cs_idx
                    with torch.cuda.stream(streams[stream_id]):

                        proj = extract_fourier_slice(
                            dft_volume=dft_map,
                            rotation_matrices=batch_matrices,
                            volume_shape=mrc_map_shape,
                        )
                        # Rearrange and multiply with template filter
                        proj = einops.rearrange(proj, "nAng h w -> 1 1 nAng h w")

                        proj = proj * combined_template_filter
                
                        # Backwards FFT
                        proj = torch.fft.irfftn(proj, dim=(-2, -1))
                
                        # Flip contrast
                        proj = proj * -1
                
                        # Mean zero var one
                        proj = mean_zero_var_one_full_size(
                            projections=proj,
                            micrographs_shape=micrograph.shape[-2:],
                        )

                        #convert proj to float32
                        #proj = proj.to(torch.float32)
                        # Pad projections
                        proj = pad_to_shape_2d(
                            image=proj,
                            image_shape=proj.shape[-2:],
                            shape=micrograph.shape[-2:],
                            pad_val=0,
                        )
                
                        # Cross correlation
                        proj = simple_cross_correlation_single(
                            projections=proj,
                            dft_micrographs_filtered=dft_micrograph.to(device),
                        )
                                    # ... existing processing code ...

                        # Use stream-specific tensors
                        stream_tensors[stream_id]['sum_correlation'] += einops.reduce(proj, "nDefoc nCs nAng h w -> h w", "sum")
                        stream_tensors[stream_id]['sum_correlation_squared'] += einops.reduce(proj**2, "nDefoc nCs nAng h w -> h w", "sum")

                        #Get best and max
                        this_best_defoc, this_best_pixel_size, this_best_phi, this_best_theta, this_best_psi, this_maximum_intensity_projection = get_mip_and_best_single_mic(
                            projections=proj,
                            defocus_values=defocus_values_gpu,
                            Cs_vals=Cs_vals_gpu,
                            euler_angles=batch_euler_angles,
                            micrograph_data=micrograph_data.iloc[i],
                        )

                        # Update stream-specific best values where this batch has higher correlation
                        mask = this_maximum_intensity_projection >= stream_tensors[stream_id]['maximum_intensity_projection']
                        stream_tensors[stream_id]['maximum_intensity_projection'][mask] = this_maximum_intensity_projection[mask]
                        stream_tensors[stream_id]['best_defoc'][mask] = this_best_defoc[mask]
                        stream_tensors[stream_id]['best_pixel_size'][mask] = this_best_pixel_size[mask]
                        stream_tensors[stream_id]['best_phi'][mask] = this_best_phi[mask]
                        stream_tensors[stream_id]['best_theta'][mask] = this_best_theta[mask]
                        stream_tensors[stream_id]['best_psi'][mask] = this_best_psi[mask]



            torch.cuda.synchronize()
            #Need to process each batch to get best and max, update sum corr and histogram
            # Combine results from all streams
            for stream_tensor in stream_tensors:
                mask = stream_tensor['maximum_intensity_projection'] >= maximum_intensiy_projection
                maximum_intensiy_projection[mask] = stream_tensor['maximum_intensity_projection'][mask]
                best_defoc[mask] = stream_tensor['best_defoc'][mask]
                best_pixel_size[mask] = stream_tensor['best_pixel_size'][mask]
                best_phi[mask] = stream_tensor['best_phi'][mask]
                best_theta[mask] = stream_tensor['best_theta'][mask]
                best_psi[mask] = stream_tensor['best_psi'][mask]
                sum_correlation = sum_correlation + stream_tensor['sum_correlation']
                sum_correlation_squared = sum_correlation_squared + stream_tensor['sum_correlation_squared']

        torch.cuda.synchronize()    

        #Move everything back to cpu
        sum_correlation = sum_correlation.cpu()
        sum_correlation_squared = sum_correlation_squared.cpu()
        best_defoc = best_defoc.cpu()
        best_pixel_size = best_pixel_size.cpu()
        best_phi = best_phi.cpu()
        best_theta = best_theta.cpu()
        best_psi = best_psi.cpu()
        maximum_intensiy_projection = maximum_intensiy_projection.cpu()

        #finsish processing
        num_pixels = torch.tensor(micrograph.shape[-2] * micrograph.shape[-1], dtype=torch.float32)
        total_correlation_positions = torch.tensor(defocus_values[i].shape[0] * Cs_vals.shape[0] * nAng, dtype=torch.float32)
        print("total correlation positions", total_correlation_positions)
        maximum_intensiy_projection = maximum_intensiy_projection * torch.sqrt(num_pixels)

        process_match_results_single(
            all_inputs=all_inputs,
            micrograph_number=i,
            micrograph_data=micrograph_data.iloc[i],
            maximum_intensiy_projection=maximum_intensiy_projection,
            sum_correlation=sum_correlation,
            sum_correlation_squared=sum_correlation_squared,
            best_defoc=best_defoc,
            best_pixel_size=best_pixel_size,
            best_phi=best_phi,
            best_theta=best_theta,
            best_psi=best_psi,
            num_pixels=num_pixels,
            total_correlation_positions=total_correlation_positions,
        )


if __name__ == "__main__":
    import time
    start_time = time.time()
    #run_tm_cpu_batch("/Users/josh/git/teamtomo/tt2DTM/data/inputs_batch.yaml")
    run_tm_gpu_batch_1("/home/jdickerson/git/teamtomo/tt2DTM/data/inputs_batch.yaml")
    end_time = time.time()
    print(f"Total runtime gpu attempt 1: {end_time - start_time:.2f} seconds")
    #start_time = time.time()
    #run_tm_gpu_batch_2("/home/jdickerson/git/teamtomo/tt2DTM/data/inputs_batch.yaml")
    #end_time = time.time()
    #print(f"Total runtime gpu attempt 2: {end_time - start_time:.2f} seconds")
    #run_tm("/Users/josh/git/teamtomo/tt2DTM/data/inputs.yaml")
    #run_tm("/Users/josh/git/teamtomo/tt2DTM/data/inputs.yaml")



'''
GPU code attempt1, very simple, just trying to get it to work

import torch
import einops

# Assume micrograph and other inputs are already on the GPU
device = torch.device('cuda')

# Initialize output tensors directly on GPU
sum_correlation = torch.zeros(micrograph.shape[-2:], dtype=torch.float64, device=device)
sum_correlation_squared = torch.zeros(micrograph.shape[-2:], dtype=torch.float64, device=device)
best_defoc = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
best_pixel_size = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
best_phi = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
best_theta = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
best_psi = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
maximum_intensity_projection = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)

for batch_idx in range(num_batches):
    # Calculate batch indices
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_angles)

    # Load the batch into GPU memory
    batch_matrices = rotation_matrices[start_idx:end_idx].to(device)
    batch_euler_angles = euler_angles[start_idx:end_idx].to(device)

    # Fourier slice extraction
    proj = extract_fourier_slice(
        dft_volume=dft_map.to(device),  # Ensure dft_map is on the GPU
        rotation_matrices=batch_matrices,
        volume_shape=mrc_map.shape,
    )
    proj = einops.rearrange(proj, "nAng h w -> 1 1 nAng h w")

    # Apply template filter
    combined_template_filter = einops.rearrange(
        combined_template_filter, "nDefoc nCs h w -> nDefoc nCs 1 h w"
    ).to(device)
    proj = proj * combined_template_filter

    # Inverse FFT (GPU-accelerated)
    proj = torch.fft.irfftn(proj, dim=(-2, -1))

    # Flip contrast
    proj *= -1

    # Normalize to zero mean and unit variance
    proj = mean_zero_var_one_full_size(
        projections=proj,
        micrographs_shape=micrograph.shape[-2:],
    )

    # Pad projections
    proj = pad_to_shape_2d(
        image=proj,
        image_shape=proj.shape[-2:],
        shape=micrograph.shape[-2:],
        pad_val=0,
    )

    # Cross-correlation
    proj = simple_cross_correlation_single(
        projections=proj,
        dft_micrographs_filtered=dft_micrograph.to(device),
    )

    # Update sums for correlation and squared correlation
    sum_correlation = sum_correlation + einops.reduce(proj, "nDefoc nCs nAng h w -> h w", "sum")
    sum_correlation_squared = sum_correlation_squared + einops.reduce(proj**2, "nDefoc nCs nAng h w -> h w", "sum")

    # Calculate best parameters
    (
        this_best_defoc,
        this_best_pixel_size,
        this_best_phi,
        this_best_theta,
        this_best_psi,
        this_maximum_intensity_projection,
    ) = get_mip_and_best_single_mic(
        projections=proj,
        defocus_values=defocus_values[i],
        Cs_vals=Cs_vals,
        euler_angles=batch_euler_angles,
        micrograph_data=micrograph_data.iloc[i],
    )

    # Update best values using a mask
    mask = this_maximum_intensity_projection >= maximum_intensity_projection
    maximum_intensity_projection[mask] = this_maximum_intensity_projection[mask]
    best_defoc[mask] = this_best_defoc[mask]
    best_pixel_size[mask] = this_best_pixel_size[mask]
    best_phi[mask] = this_best_phi[mask]
    best_theta[mask] = this_best_theta[mask]
    best_psi[mask] = this_best_psi[mask]

torch.cuda.synchronize()  # Ensure all GPU computations are completed

'''

'''
with CUDA streams

import torch
import einops

# GPU device
device = torch.device('cuda')

# Problem dimensions
nAng = 1000000
nDefoc = 20
nCs = 10

# Adjustable batch size for angles
nAng_batch = 10  # Start with 10 angles per batch; adjust based on memory profiling

# Number of streams for nDefoc * nCs
num_streams = nDefoc * nCs
streams = [torch.cuda.Stream() for _ in range(num_streams)]

# Pre-allocate tensors to store results for each stream
max_correlations = [None] * num_streams
max_indices = [None] * num_streams

# Output tensors
sum_correlation = torch.zeros(micrograph.shape[-2:], dtype=torch.float64, device=device)
sum_correlation_squared = torch.zeros(micrograph.shape[-2:], dtype=torch.float64, device=device)
best_defoc = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
best_pixel_size = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
best_phi = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
best_theta = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
best_psi = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)
maximum_intensity_projection = torch.zeros(micrograph.shape[-2:], dtype=torch.float32, device=device)

# Iterate over angle batches
for batch_start in range(0, nAng, nAng_batch):
    batch_end = min(batch_start + nAng_batch, nAng)
    angle_batch_size = batch_end - batch_start

    # Prepare batch-specific data
    batch_matrices = rotation_matrices[batch_start:batch_end].to(device)
    batch_euler_angles = euler_angles[batch_start:batch_end].to(device)

    # Launch parallel streams over nDefoc * nCs
    for defoc_idx in range(nDefoc):
        for cs_idx in range(nCs):
            stream_id = defoc_idx * nCs + cs_idx
            with torch.cuda.stream(streams[stream_id]):
                # Fourier slice extraction for this batch
                proj = extract_fourier_slice(
                    dft_volume=dft_map.to(device),
                    rotation_matrices=batch_matrices,
                    volume_shape=mrc_map.shape,
                )
                proj = einops.rearrange(proj, "nAng h w -> 1 1 nAng h w")

                # Apply specific template filter for this defoc and Cs
                template_filter = combined_template_filter[defoc_idx, cs_idx].unsqueeze(0).unsqueeze(0).to(device)
                proj *= template_filter

                # Inverse FFT
                proj = torch.fft.irfftn(proj, dim=(-2, -1))

                # Flip contrast and normalize
                proj *= -1
                proj = mean_zero_var_one_full_size(
                    projections=proj,
                    micrographs_shape=micrograph.shape[-2:],
                )
                proj = pad_to_shape_2d(
                    image=proj,
                    image_shape=proj.shape[-2:],
                    shape=micrograph.shape[-2:],
                    pad_val=0,
                )

                # Cross-correlation
                proj = simple_cross_correlation_single(
                    projections=proj,
                    dft_micrographs_filtered=dft_micrograph.to(device),
                )

                # Sum correlations for this stream
                sum_correlation += einops.reduce(proj, "1 1 nAng h w -> h w", "sum")
                sum_correlation_squared += einops.reduce(proj**2, "1 1 nAng h w -> h w", "sum")

                # Find maximum correlation for this stream
                max_corr, indices = proj.max(dim=(-2, -1))  # Max over spatial dimensions
                max_correlations[stream_id] = max_corr
                max_indices[stream_id] = indices

# Synchronize all streams
torch.cuda.synchronize()

# Combine results across all streams
overall_max_corr = torch.stack(max_correlations).max()
print("Overall Maximum Correlation:", overall_max_corr.item())

'''