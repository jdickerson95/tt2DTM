# This will take many inputs
# mrc map, micrographs, ctf's, output dir, px size, etc.

# It will mainly follow torch 2dtm package stuff
import einops
import torch
from torch_fourier_filter.ctf import calculate_ctf_2d
from torch_fourier_filter.phase_randomize import phase_randomize

from tt2dtm.match_template.load_angles import euler_to_rotation_matrix, get_euler_angles, get_rotation_matrices
from tt2dtm.utils.calculate_filters import (
    Cs_to_pixel_size,
    calculate_2d_template_filters,
    calculate_micrograph_filters,
    combine_filters,
    get_Cs_range,
    get_defocus_range,
    get_defocus_values,
)
from tt2dtm.utils.fourier_slice import _sinc2, extract_fourier_slice, fft_volume
from tt2dtm.utils.image_operations import pad_to_shape_2d, pad_volume, edge_mean_reduction_2d
from tt2dtm.utils.io_handler import (
    load_mrc_map,
    load_mrc_micrographs,
    load_relion_starfile,
    read_inputs,
    tensor_to_mrc,
)


def run_tm(input_yaml: str):
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

        #Make sure mean zero after calc whiten to avoid div by zero
    dft_micrographs[:, 0, 0] = 0 + 0j

    ####MICROGRAPH OPERATIONS####
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

    print("defocus shapes", defoc_u_vals.shape, defoc_v_vals.shape)
    # Calculate size needed for 2D projection based on Cs/CTF.

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
    # Calculate CTFs and multiply them with the filters
    defocus_values = (defoc_u_vals + defoc_v_vals) / 2
    astigmatism_values = (defoc_u_vals - defoc_v_vals) / 2
    astigmatism_angle = torch.tensor(
        micrograph_data["rlnDefocusAngle"], dtype=torch.float32
    )
    astigmatism_angle = einops.rearrange(astigmatism_angle, "n -> n 1")
    # broadcast with einops to match defocus values

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
        image_shape=mrc_map.shape[-2:],
        rfft=True,
        fftshift=False,
    )  # shape needs to be (n_micrographs, n_Cs, n_defoc, h, w)
    print("ctf shape", ctf_2d.shape)
    # if ctf only four dimensions, rearrange to 5 (add Cs dimension)
    if ctf_2d.ndim == 4:
        ctf_2d = einops.rearrange(ctf_2d, "nMic nDefoc h w -> nMic nDefoc 1 h w")
    print("ctf shape", ctf_2d.shape)
    # multiply ctf by filters
    combined_template_filter = einops.rearrange(
        combined_template_filter, "nMic h w -> nMic 1 1 h w"
    )
    combined_template_filter = combined_template_filter * ctf_2d
    print("combined template filter shape", combined_template_filter.shape)


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
    #. Subtract mean of edge (mean 0)
    mean_edge = edge_mean_reduction_2d(projections)
    mean_edge = einops.rearrange(mean_edge, "... -> ... 1 1") # add h w
    projections = projections - mean_edge
    print("mean edge shape", mean_edge.shape)
    print("test")
    # set variance 1, but for the full projection once it is fully padded
    npix_template = projections.shape[-2] * projections.shape[-1]
    npix_micrograph = micrographs.shape[-2] * micrographs.shape[-1]
    mean_edge = edge_mean_reduction_2d(projections)
    mean_edge = einops.rearrange(mean_edge, "... -> ... 1 1") # add h w
    sum_squares = torch.sum(projections ** 2, dim=(-2, -1), keepdim=True)
    variance = sum_squares * npix_template / npix_micrograph - ((mean_edge * npix_template / npix_micrograph) ** 2)
    projections = projections / torch.sqrt(variance)
    # Pad projections with zeros to size of micrographs
    projections = pad_to_shape_2d(
        image=projections,
        image_shape=projections.shape[-2:],
        shape=micrographs.shape[-2:],
        pad_val=0,
    )
    print("projections padded shape", projections.shape)
    # FFT shift to edge
    projections = torch.fft.fftshift(projections, dim=(-2, -1))
    # do FFT and 0 central pixel
    projections = torch.fft.rfftn(projections, dim=(-2, -1))
    # zero central pixel
    projections[:, 0, 0] = 0 + 0j
    print("projections fft shape", projections.shape)
    # cross correlations
    dft_micrographs_filtered = einops.rearrange(dft_micrographs_filtered, "nMic h w -> nMic 1 1 1 h w")
    projections = projections.conj() * dft_micrographs_filtered
    
    print("projections shape", projections.shape)
    # Inverse FFT this MIP
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    
    #Get all the max values and indices
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

    print("best defoc", best_defoc.shape)
    print("best cs", best_cs.shape)
    print("best angles", best_angles.shape)
    print("max intensity projections", maximum_intensiy_projections.shape)
    #get best phi, theta, psi
    best_phi = best_angles[..., 0]
    best_theta = best_angles[..., 1]
    best_psi = best_angles[..., 2]
    print("best phi", best_phi.shape)
    print("best theta", best_theta.shape)
    print("best psi", best_psi.shape)
    # get max SNRs and stuff
    
    #sum all xcorr per micrograph
    sum_correlation = einops.reduce(projections, "nMic nDefoc nCs nAng h w -> nMic h w", "sum")
    sum_correlation_squared = einops.reduce(projections ** 2, "nMic nDefoc nCs nAng h w -> nMic h w", "sum")
    total_correlation_positions = projections.shape[1] * projections.shape[2] * projections.shape[3]
    num_pixels = torch.tensor(projections.shape[-2] * projections.shape[-1], dtype=torch.float32)
    sum_correlation = sum_correlation / total_correlation_positions
    sum_correlation_squared = sum_correlation_squared / total_correlation_positions - (sum_correlation ** 2)
    sum_correlation_squared = torch.sqrt(torch.clamp(sum_correlation_squared, min=0)) * torch.sqrt(num_pixels)
    sum_correlation = sum_correlation * torch.sqrt(num_pixels)
    maximum_intensiy_projections = maximum_intensiy_projections * torch.sqrt(num_pixels)
    #Get the expected noise
    CCG_NOISE_STDEV = 1.0
    erf_input = 2.0 / (1.0 * num_pixels * total_correlation_positions)
    inverse_c_erf = torch.erfinv(1 - erf_input)
    expected_noise = (2**0.5) * CCG_NOISE_STDEV * inverse_c_erf

    #normalize
    maximum_intensiy_projections_normalized = maximum_intensiy_projections - sum_correlation
    maximum_intensiy_projections_normalized = torch.where(sum_correlation_squared == 0, 
                                             torch.zeros_like(maximum_intensiy_projections_normalized),
                                             maximum_intensiy_projections_normalized / sum_correlation_squared)
    #Get the SNR0-
    #write to file
    #max intensity to mrc called mip
    output_dir = all_inputs["outputs"]["output_directory"]
    for i in range(maximum_intensiy_projections.shape[0]):
        print(f"Writing mip for micrograph {i+1}")
        tensor_to_mrc(
            output_filename=f"{output_dir}/mip_micrograph_{i+1}.mrc",
            final_array=maximum_intensiy_projections[i],
            pixel_spacing=float(micrograph_data["rlnMicrographPixelSize"][0])
        )
        print(f"Writing scaled mip for micrograph {i+1}")
        tensor_to_mrc(
            output_filename=f"{output_dir}/scaled_mip_micrograph_{i+1}.mrc",
            final_array=maximum_intensiy_projections_normalized[i],
            pixel_spacing=float(micrograph_data["rlnMicrographPixelSize"][0])
        )
        print(f"Writing sum_correlation for micrograph {i+1}")
        tensor_to_mrc(
            output_filename=f"{output_dir}/corr_avg_micrograph_{i+1}.mrc",
            final_array=sum_correlation[i],
            pixel_spacing=float(micrograph_data["rlnMicrographPixelSize"][0])
        )
        print(f"Writing sum_corr_sqaured for micrograph {i+1}")
        tensor_to_mrc(
            output_filename=f"{output_dir}/corr_std_micrograph_{i+1}.mrc",
            final_array=sum_correlation_squared[i],
            pixel_spacing=float(micrograph_data["rlnMicrographPixelSize"][0])
        )
        print(f"Writing best defoc for micrograph {i+1}")
        tensor_to_mrc(
            output_filename=f"{output_dir}/best_defoc_micrograph_{i+1}.mrc",
            final_array=best_defoc[i],
            pixel_spacing=float(micrograph_data["rlnMicrographPixelSize"][0])
        )
        print(f"Writing best angles for micrograph {i+1}")
        tensor_to_mrc(
            output_filename=f"{output_dir}/best_phi_micrograph_{i+1}.mrc",
            final_array=best_phi[i],
            pixel_spacing=float(micrograph_data["rlnMicrographPixelSize"][0])
        )
        tensor_to_mrc(
            output_filename=f"{output_dir}/best_theta_micrograph_{i+1}.mrc",
            final_array=best_theta[i],
            pixel_spacing=float(micrograph_data["rlnMicrographPixelSize"][0])
        )
        tensor_to_mrc(
            output_filename=f"{output_dir}/best_psi_micrograph_{i+1}.mrc",
            final_array=best_psi[i],
            pixel_spacing=float(micrograph_data["rlnMicrographPixelSize"][0])
        )
        print(f"Writing best pixel size for micrograph {i+1}")
        tensor_to_mrc(
            output_filename=f"{output_dir}/best_pixel_size_micrograph_{i+1}.mrc",
            final_array=best_pixel_size[i],
            pixel_spacing=float(micrograph_data["rlnMicrographPixelSize"][0])
        )
        print(f"Writing best Cs for micrograph {i+1}")
        tensor_to_mrc(
            output_filename=f"{output_dir}/best_Cs_micrograph_{i+1}.mrc",
            final_array=best_cs[i],
            pixel_spacing=float(micrograph_data["rlnMicrographPixelSize"][0])
        )
    #write the best defoc, angles, pixel size/Cs

    #write survival histogram when I get one

if __name__ == "__main__":
    run_tm("/Users/josh/git/teamtomo/tt2DTM/data/inputs.yaml")
