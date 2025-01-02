"""Deals with input/output operations."""

import mrcfile
import torch
import einops
import starfile
import sys
import yaml
import numpy as np


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

# some functions here to convert parts of dict or df into tensors


