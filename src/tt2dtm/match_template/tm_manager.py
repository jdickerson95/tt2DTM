
# This will take many inputs
# mrc map, micrographs, ctf's, output dir, px size, etc.

#It will mainly follow torch 2dtm package stuff
import torch
import einops
from tt2dtm.utils.calculate_filters import( 
    calculate_micrograph_filters,
    calculate_2d_template_filters,
    combine_filters,
    get_Cs_range,
    get_defocus_range,
    get_defocus_values
)
from tt2dtm.utils.fourier_slice import extract_fourier_slice, fft_volume, multiply_by_sinc2
from tt2dtm.utils.image_operations import pad_volume
from tt2dtm.utils.io_handler import(
    load_relion_starfile,
    load_mrc_micrographs,
    load_mrc_map, 
    read_inputs
)
from tt2dtm.match_template.load_angles import get_rotation_matrices

from torch_fourier_filter.phase_randomize import phase_randomize



def run_tm(
    input_yaml: str
):
    #read the input yaml file
    print("Reading input yaml")
    all_inputs = read_inputs(input_yaml)
    
    ####LOAD THE DATA####
    #Get the micrograph list from relion starfile (add other options later)
    micrograph_data = load_relion_starfile(all_inputs['input_files']['micrograph_ctf_star'])
    #Load the micrographs
    micrographs = load_mrc_micrographs(micrograph_data['rlnMicrographName'])  
    #Load the mrc template map or pdb
    mrc_map = load_mrc_map(all_inputs['input_files']['mrc_map'])
    # if load pdb call sim3d, calc output size before that to be as small as possible

    #pad micrographs and mrc map up to nearest radix of 2, 3, 5, 7, 11, 13

    # if pad true, double size of volume for slice extraction
    # and filter multiplication
    if all_inputs['filters']['pad']['enabled']:
        mrc_map = pad_volume(mrc_map, pad_length=mrc_map.shape[-1] // 2)
    
    
    ####ANGLE OPERATIONS####
    #Load the angles, based on symmetry and ranges, steps, etc
    print("Calculating rotation matrices")
    rotation_matrices = get_rotation_matrices(all_inputs)
    print(rotation_matrices[0].shape)


    ####CALC FILTERS####
    #dft the micrographs and keep them like this. zero mean
    dft_micrographs = torch.fft.rfftn(micrographs, dim=(-2, -1))
    dft_micrographs[:,0,0] = 0 + 0j
    # calc whitening and any other filters for micrograph
    whiten_micrograph, bandpass_micrograph = calculate_micrograph_filters(
        all_inputs=all_inputs, 
        micrograph_data=micrograph_data, 
        dft_micrographs=dft_micrographs,
        micrograph_shape=micrographs.shape[-2:]
        )
    #combine filters together
    combined_micrograph_filter = combine_filters(whiten_micrograph, bandpass_micrograph)

    # calc 2D whitening and any others for template
    # one for each micrograph/ multiply filters together
    whiten_template, bandpass_template = calculate_2d_template_filters(
        all_inputs=all_inputs, 
        micrograph_data=micrograph_data, 
        dft_micrographs=dft_micrographs,
        micrograph_shape=micrographs.shape[-2:],
        template_shape=mrc_map.shape
        )
    combined_template_filter = combine_filters(whiten_template, bandpass_template)


    ####MICROGRAPH OPERATIONS####
    # Apply the filter to the micrographs and phase random if wanted
    dft_micrographs_filtered = dft_micrographs * combined_micrograph_filter
    if all_inputs['filters']['phase_randomize']['enabled']:
        cuton = float(all_inputs['filters']['phase_randomize']['cuton_resolution'])
        dft_micrographs_filtered = phase_randomize(
            dft=dft_micrographs_filtered,
            image_shape=micrographs.shape[-2:],
            rfft=True,
            cuton=cuton,
            fftshift=False,
        )
    # zero central pixel
    dft_micrographs_filtered[:,0,0] = 0 + 0j
    # divide by sqrt sum of squares. 
    dft_micrographs_filtered /= torch.sqrt(torch.sum(torch.abs(dft_micrographs_filtered)**2, dim=(-2, -1), keepdim=True))


    ####TEMPLATE STUFF####
    # Calculate range of Cs values of pixel size search
    Cs_vals = torch.tensor(float(micrograph_data['rlnSphericalAberration'][0]))
    if all_inputs['extra_searches']['pixel_size_search']['enabled']:
        Cs_vals = get_Cs_range(pixel_size=float(all_inputs['extra_searches']['pixel_size_search']['pixel_size']),
            pixel_size_range=float(all_inputs['extra_searches']['pixel_size_search']['range']),
            pixel_size_step=float(all_inputs['extra_searches']['pixel_size_search']['step_size']),
            Cs=float(micrograph_data['rlnSphericalAberration'][0]),
        )
    # Calculate defocus range for each micrograph
    defoc_u_vals = torch.tensor(float(micrograph_data['rlnDefocusU']))
    defoc_v_vals = torch.tensor(float(micrograph_data['rlnDefocusV']))
    if all_inputs['extra_searches']['defocus_search']['enabled']:
        defoc_range = get_defocus_range(
            defocus_range=float(all_inputs['extra_searches']['defocus_search']['range']),
            defocus_step=float(all_inputs['extra_searches']['defocus_search']['step_size']),
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


    # Calculate size needed for 2D projection based on Cs/CTF.

    #### Extract Fourier slice at angles ####
    # Multiply map by sinc2
    mrc_map = multiply_by_sinc2(map=mrc_map)
    dft_map = fft_volume(
        volume=mrc_map,
        fftshift=True, #Needed for slice extraction
    )
    projections = extract_fourier_slice(
        dft_volume=dft_map,
        rotation_matrices=rotation_matrices,
        volume_shape=mrc_map.shape,
    ) # shape (n_angles, h, w)

    # Calculate CTFs and multiply them with the filters
    # I have Cs shape nCs and defoc shape (nMic, nDefoc)
    
    # Apply the combined filters to projections
    # Backwards FFT. Subtract mean of edge (mean 0)
    # set variance 1, but for the full projection once it is fully padded
    # Pad projection with zeros to make a larger size. FFT shift to edge
    # do FFT and 0 central pixel

    #cross correlations

    #get max SNRs and best orientations and everything else. 

if __name__ == "__main__":
    run_tm('/Users/josh/git/teamtomo/tt2DTM/data/inputs.yaml')






