"""Deals with input/output operations."""

import mrcfile
import torch
import einops
import starfile
import sys
import yaml
import numpy as np
import pandas as pd


def read_inputs(
        file_path: str
) -> dict:
    """
    Reads and parses the inputs from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML data as a Python dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        sys.exit()
    except yaml.YAMLError as e:
        print(f"Error: Issue reading the YAML file. {e}")
        sys.exit()

def load_mrc_map(
        file_path: str
) -> torch.Tensor:
    """Load MRC file into a torch tensor."""

    with mrcfile.open(file_path) as mrc:
        return torch.tensor(mrc.data)
    
def load_mrc_micrographs(
        file_paths: list[str]
) -> torch.Tensor:
    """Load MRC micrographs into a torch tensor."""

    for i, file_path in enumerate(file_paths):
        with mrcfile.open(file_path) as mrc:
            if i == 0:
                mrc_data = torch.tensor(mrc.data, dtype=torch.float32)
                mrc_data = einops.rearrange(mrc_data, 'h w -> 1 h w')
            else:
                mrc_data = torch.cat((mrc_data, einops.rearrange(torch.tensor(mrc.data, dtype=torch.float32), 'h w -> 1 h w')), dim=0)
    return mrc_data


def load_relion_starfile(
        file_path: str # Path to relion ctffind output
) -> tuple  :
    """Load micrographs_ctf.star into dataframe"""

    star = starfile.read(file_path)
    # merge into one dataframe
    df = star['micrographs'].merge(star['optics'], on='rlnOpticsGroup')
    return df

def tensor_to_mrc(
    output_filename: str,
    final_array: torch.Tensor,
    pixel_spacing: float,
) -> None:
    """Write a 2D array to an MRC file.

    Parameters
    ----------
    output_filename : str
        Path to the output MRC file.
    final_array : torch.Tensor
        Array information to write to the MRC file.
    pixel_spacing :
        pixel spacing in angstroms.

    Returns
    -------
    None
    """
    if final_array.ndim > 2:
        final_array = final_array.squeeze()  # Remove singleton dimensions
    write_array = final_array.cpu().numpy().astype(np.float32)
    with mrcfile.new(output_filename, overwrite=True) as mrc:
        mrc.set_data(write_array)
        # Populate more of the metadata...
        # Header setup
        mrc.header.mode = 2  # Mode 2 is float32
        mrc.header.nx = write_array.shape[1]  # X dimension
        mrc.header.ny = write_array.shape[0]  # Y dimension
        mrc.header.mx = write_array.shape[1]  # X sampling (same as dimension)
        mrc.header.my = write_array.shape[0]  # Y sampling (same as dimension)

        # Cell dimensions
        cell_dimensions = [dim * pixel_spacing for dim in [write_array.shape[1], write_array.shape[0]]]
        mrc.header.cella.x = cell_dimensions[0]  # X dimension in angstroms
        mrc.header.cella.y = cell_dimensions[1]  # Y dimension in angstroms

        mrc.header.mapc = 1  # Columns correspond to x
        mrc.header.mapr = 2  # Rows correspond to y
        mrc.header.maps = 3  # Sections correspond to z

        mrc.header.dmin = write_array.min()  # Minimum density value
        mrc.header.dmax = write_array.max()  # Maximum density value
        mrc.header.dmean = write_array.mean()  # Mean density value

        # Additional metadata
        mrc.update_header_from_data()  # Automatically populates remaining fields
        mrc.header.rms = np.std(write_array)  # RMS deviation of the density values

def write_mrc_files(
    all_inputs: dict,
    micrograph_data: pd.DataFrame,
    maximum_intensiy_projections: torch.Tensor,
    maximum_intensiy_projections_normalized: torch.Tensor,
    sum_correlation: torch.Tensor,
    sum_correlation_squared: torch.Tensor,
    best_defoc: torch.Tensor,
    best_phi: torch.Tensor,
    best_theta: torch.Tensor,
    best_psi: torch.Tensor,
    best_pixel_size: torch.Tensor,
) -> None:
    """
    Write mrc files for all micrographs

    Parameters
    ----------
    all_inputs : dict
        All inputs from the YAML file.
    micrograph_data : pd.DataFrame
        Micrograph data from the starfile.
    maximum_intensiy_projections : torch.Tensor
        Maximum intensity projections.
    maximum_intensiy_projections_normalized : torch.Tensor
        Maximum intensity projections normalized.
    sum_correlation : torch.Tensor
        Sum of correlation.
    sum_correlation_squared : torch.Tensor
        Sum of correlation squared.
    best_defoc : torch.Tensor
        Best defocus.
    best_phi : torch.Tensor
        Best phi.
    best_theta : torch.Tensor
        Best theta.
    best_psi : torch.Tensor
        Best psi.
    best_pixel_size : torch.Tensor
        Best pixel size.

    Returns
    -------
    None
    """
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

def write_survival_histogram(
    all_inputs: dict,
    survival_histogram: torch.Tensor,
    nMicrographs: int,
    expected_noise: float,
    histogram_data: torch.Tensor,
    expected_survival_hist: torch.Tensor,
    temp_float: float,
    HISTOGRAM_STEP: float,
    HISTOGRAM_NUM_POINTS: int,
) -> None:
    """Write survival histogram to file."""
    output_dir = all_inputs["outputs"]["output_directory"]
    print("Writing survival histogram")
    for j in range(nMicrographs):
        with open(f"{output_dir}/survival_histogram_micrograph_{j+1}.txt", "w") as f:
            f.write(f"Expected threshold is {expected_noise}\n")
            f.write(f"SNR, histogram, survival histogram, random survival histogram\n")
            for i in range(HISTOGRAM_NUM_POINTS):
                f.write(f"{temp_float + HISTOGRAM_STEP *i}, {histogram_data[j, i]}, {survival_histogram[j, i]}, {expected_survival_hist[i]}\n")
# some functions here to convert parts of dict or df into tensors


