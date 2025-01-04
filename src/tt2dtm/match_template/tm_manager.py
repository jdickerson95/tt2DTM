# This will take many inputs
# mrc map, micrographs, ctf's, output dir, px size, etc.

# It will mainly follow torch 2dtm package stuff
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
        batch_size = 10  # Can be tuned based on memory
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



def run_tm_gpu(input_yaml: str):
    #This is going to do the same as run_tm but...
    # 1) one micrograph at a time
    # 2) run on CPU and then multiple GPUs
    # 3) batched operation for efficiency
    # 4) Use float16 to reduce memory usage, but start without this

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

    #get ctf for all micrographs
    ctf_2d = get_ctf(
        defocus_values=defocus_values,
        defoc_u_vals=defoc_u_vals,
        defoc_v_vals=defoc_v_vals,
        micrograph_data=micrograph_data,
        Cs_vals=Cs_vals,
        mrc_map_shape=mrc_map.shape,
    )

    ####TEMPLATE STUFF####
    mrc_map = mrc_map * _sinc2(shape=mrc_map.shape, rfft=False, fftshift=True)

    dft_map = fft_volume(
        volume=mrc_map,
        fftshift=True,  # Needed for slice extraction
    )

    #Do one micrograph at a time

    for i, micrograph in enumerate(micrographs):
        # Do on CPU until get to ctf calc and fourier slice
        this_ctf = ctf_2d[i]
        print("this ctf shape", this_ctf.shape)
        ###Calc Filters###
        dft_micrograph = torch.fft.rfftn(micrograph, dim=(-2, -1))
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

        ###Template stuff###
        # multiply ctf by filters
        combined_template_filter = einops.rearrange(
            combined_template_filter, "h w -> 1 1 h w"
        )
        combined_template_filter = combined_template_filter * this_ctf

        ###Extract Fourier slice at angles###
                    # Move relevant data to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #num_gpus = torch.cuda.device_count()
        num_gpus = 4 #User should set this
        
        # Move volume to first GPU (can be shared across GPUs)
        dft_map = dft_map.to(device)
        
        # Calculate batch size based on available GPU memory
        batch_size = 2048  # Can be tuned based on GPU memory
        num_angles = rotation_matrices.shape[0]
        num_batches = (num_angles + batch_size - 1) // batch_size
        
        # Create streams for concurrent execution
        num_streams = 4  # Number of concurrent operations per GPU, set better later
        streams = [[torch.cuda.Stream() for _ in range(num_streams)] for _ in range(num_gpus)]

        all_projections = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_angles)
            
            # If multiple GPUs available, split batch across GPUs
            if num_gpus > 1:
                sub_batch_size = (end_idx - start_idx + num_gpus - 1) // num_gpus
                batch_projections = []
                
                for gpu_idx in range(num_gpus):
                    gpu_start = start_idx + gpu_idx * sub_batch_size
                    gpu_end = min(gpu_start + sub_batch_size, end_idx)
                    
                    if gpu_start >= gpu_end:
                        continue
                        
                    gpu_device = torch.device(f'cuda:{gpu_idx}')
                    batch_matrices = rotation_matrices[gpu_start:gpu_end].to(gpu_device)
                    
                    # Process sub-batch on specific GPU using multiple streams
                    stream_batch_size = (gpu_end - gpu_start + num_streams - 1) // num_streams
                    stream_results = []
                    
                    for stream_idx, stream in enumerate(streams[gpu_idx]):
                        stream_start = gpu_start + stream_idx * stream_batch_size
                        stream_end = min(stream_start + stream_batch_size, gpu_end)
                        
                        if stream_start >= stream_end:
                            continue
                            
                        with torch.cuda.stream(stream):
                            stream_matrices = batch_matrices[stream_start-gpu_start:stream_end-gpu_start]
                            proj = extract_fourier_slice(
                                dft_volume=dft_map.to(gpu_device),
                                rotation_matrices=stream_matrices,
                                volume_shape=mrc_map.shape,
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

                            stream_results.append(proj)
                    
                    # Synchronize streams and combine results
                    torch.cuda.current_stream().wait_stream(*streams[gpu_idx])
                    batch_projections.append(torch.cat(stream_results, dim=0))
                
                batch_result = torch.cat(batch_projections, dim=0)
                
            else:
                # Single GPU processing with multiple streams
                batch_matrices = rotation_matrices[start_idx:end_idx].to(device)
                stream_batch_size = (end_idx - start_idx + num_streams - 1) // num_streams
                stream_results = []
                
                for stream_idx, stream in enumerate(streams[0]):
                    stream_start = start_idx + stream_idx * stream_batch_size
                    stream_end = min(stream_start + stream_batch_size, end_idx)
                    
                    if stream_start >= stream_end:
                        continue
                        
                    with torch.cuda.stream(stream):
                        stream_matrices = batch_matrices[stream_start-start_idx:stream_end-start_idx]
                        proj = extract_fourier_slice(
                            dft_volume=dft_map,
                            rotation_matrices=stream_matrices,
                            volume_shape=mrc_map.shape,
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
                
                        # Pad projections
                        proj = pad_to_shape_2d(
                            image=proj,
                            image_shape=proj.shape[-2:],
                            shape=micrograph.shape[-2:],
                            pad_val=0,
                        )
                
                        # Cross correlation
                        proj = simple_cross_correlation(
                            projections=proj,
                            dft_micrographs_filtered=dft_micrograph,
                        )
                        stream_results.append(proj)
                
                # Synchronize streams and combine results
                torch.cuda.current_stream().wait_stream(*streams[0])
                batch_result = torch.cat(stream_results, dim=0)
                
            all_projections.append(batch_result)
        '''
        #This doesn't need to be done for each micrograph but is for now
        projections = extract_fourier_slice(
            dft_volume=dft_map,
            rotation_matrices=rotation_matrices,
            volume_shape=mrc_map.shape,
        )  # shape (n_angles, h, w)

        projections = einops.rearrange(projections, "nAng h w -> 1 1 nAng h w")
        combined_template_filter = einops.rearrange(
            combined_template_filter, "nDefoc nCs h w -> nDefoc nCs 1 h w"
        )
        projections = projections * combined_template_filter

        # Backwards FFT
        projections = torch.fft.irfftn(projections, dim=(-2, -1))
        #flip contrast
        projections = projections * -1
        #. Subtract mean of edge (mean 0) and set variance 1 (full size)
        projections = mean_zero_var_one_full_size(
            projections=projections,
            micrographs_shape=micrograph.shape[-2:],
        )
        # Pad projections with zeros to size of micrographs
        projections = pad_to_shape_2d(
            image=projections,
            image_shape=projections.shape[-2:],
            shape=micrograph.shape[-2:],
            pad_val=0,
        )
        # cross correlation
        projections = simple_cross_correlation(
            projections=projections,
            dft_micrographs_filtered=dft_micrograph,
        )
        '''

'''
def gpu_test():
    print("GPU test")
            # Move relevant data to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()
        
        # Move volume to first GPU (can be shared across GPUs)
        dft_map = dft_map.to(device)
        
        # Calculate batch size based on available GPU memory
        # Larger batch sizes generally better utilize GPU parallelization
        batch_size = 2048  # Can be tuned based on GPU memory
        num_angles = rotation_matrices.shape[0]
        num_batches = (num_angles + batch_size - 1) // batch_size
        
        # Create streams for concurrent execution
        num_streams = 4  # Number of concurrent operations per GPU
        streams = [[torch.cuda.Stream() for _ in range(num_streams)] for _ in range(num_gpus)]
        
        all_projections = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_angles)
            
            # If multiple GPUs available, split batch across GPUs
            if num_gpus > 1:
                sub_batch_size = (end_idx - start_idx + num_gpus - 1) // num_gpus
                batch_projections = []
                
                for gpu_idx in range(num_gpus):
                    gpu_start = start_idx + gpu_idx * sub_batch_size
                    gpu_end = min(gpu_start + sub_batch_size, end_idx)
                    
                    if gpu_start >= gpu_end:
                        continue
                        
                    gpu_device = torch.device(f'cuda:{gpu_idx}')
                    batch_matrices = rotation_matrices[gpu_start:gpu_end].to(gpu_device)
                    
                    # Process sub-batch on specific GPU using multiple streams
                    stream_batch_size = (gpu_end - gpu_start + num_streams - 1) // num_streams
                    stream_results = []
                    
                    for stream_idx, stream in enumerate(streams[gpu_idx]):
                        stream_start = gpu_start + stream_idx * stream_batch_size
                        stream_end = min(stream_start + stream_batch_size, gpu_end)
                        
                        if stream_start >= stream_end:
                            continue
                            
                        with torch.cuda.stream(stream):
                            stream_matrices = batch_matrices[stream_start-gpu_start:stream_end-gpu_start]
                            proj = extract_fourier_slice(
                                dft_volume=dft_map.to(gpu_device),
                                rotation_matrices=stream_matrices,
                                volume_shape=mrc_map.shape,
                            )
                            stream_results.append(proj)
                    
                    # Synchronize streams and combine results
                    torch.cuda.current_stream().wait_stream(*streams[gpu_idx])
                    batch_projections.append(torch.cat(stream_results, dim=0))
                
                batch_result = torch.cat(batch_projections, dim=0)
                
            else:
                # Single GPU processing with multiple streams
                batch_matrices = rotation_matrices[start_idx:end_idx].to(device)
                stream_batch_size = (end_idx - start_idx + num_streams - 1) // num_streams
                stream_results = []
                
                for stream_idx, stream in enumerate(streams[0]):
                    stream_start = start_idx + stream_idx * stream_batch_size
                    stream_end = min(stream_start + stream_batch_size, end_idx)
                    
                    if stream_start >= stream_end:
                        continue
                        
                    with torch.cuda.stream(stream):
                        stream_matrices = batch_matrices[stream_start-start_idx:stream_end-start_idx]
                        proj = extract_fourier_slice(
                            dft_volume=dft_map,
                            rotation_matrices=stream_matrices,
                            volume_shape=mrc_map.shape,
                        )
                        stream_results.append(proj)
                
                # Synchronize streams and combine results
                torch.cuda.current_stream().wait_stream(*streams[0])
                batch_result = torch.cat(stream_results, dim=0)
                
            all_projections.append(batch_result)
            
        # Combine all batches, keeping on GPU
        projections = torch.cat(all_projections, dim=0)





        2

                ###Extract Fourier slice at angles###
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move data to GPU if not already there
        dft_map = dft_map.to(device)
        combined_template_filter = combined_template_filter.to(device)
        dft_micrograph = dft_micrograph.to(device)
        
        # Process projections in batches
        batch_size = 1000  # Adjust based on GPU memory
        num_angles = rotation_matrices.shape[0]
        num_batches = (num_angles + batch_size - 1) // batch_size
        
        all_projections = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_angles)
            
            with torch.cuda.stream(torch.cuda.Stream()):
                # Extract Fourier slice for batch
                batch_matrices = rotation_matrices[start_idx:end_idx].to(device)
                batch_projections = extract_fourier_slice(
                    dft_volume=dft_map,
                    rotation_matrices=batch_matrices,
                    volume_shape=mrc_map.shape,
                )  # shape (batch_angles, h, w)
                
                # Rearrange and multiply with template filter
                batch_projections = einops.rearrange(batch_projections, "nAng h w -> 1 1 nAng h w")
                batch_projections = batch_projections * combined_template_filter
                
                # Backwards FFT
                batch_projections = torch.fft.irfftn(batch_projections, dim=(-2, -1))
                
                # Flip contrast
                batch_projections = batch_projections * -1
                
                # Mean zero var one
                batch_projections = mean_zero_var_one_full_size(
                    projections=batch_projections,
                    micrographs_shape=micrograph.shape[-2:],
                )
                
                # Pad projections
                batch_projections = pad_to_shape_2d(
                    image=batch_projections,
                    image_shape=batch_projections.shape[-2:],
                    shape=micrograph.shape[-2:],
                    pad_val=0,
                )
                
                # Cross correlation
                batch_projections = simple_cross_correlation(
                    projections=batch_projections,
                    dft_micrographs_filtered=dft_micrograph,
                )
                
                all_projections.append(batch_projections)
        
        # Combine all batches (still on GPU)
        projections = torch.cat(all_projections, dim=2)  # Concatenate along angles dimension
'''

if __name__ == "__main__":
    run_tm_cpu_batch("/Users/josh/git/teamtomo/tt2DTM/data/inputs_batch.yaml")
    run_tm("/Users/josh/git/teamtomo/tt2DTM/data/inputs.yaml")
