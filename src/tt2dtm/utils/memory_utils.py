

import torch
import psutil

def calculate_batch_size(
        micrograph_shape: tuple, 
        defoc_vals: torch.Tensor,
        Cs_vals: torch.Tensor,
        device: torch.device,
        available_memory_gb: float = None
) -> int:
    """
    Calculate the maximum batch size that will fit in memory.
    
    Parameters
    ----------
    micrograph_shape: tuple
        Shape of the micrograph (h, w)
    defoc_vals: torch.Tensor
        Defocus values tensor to get number of defocus values
    Cs_vals: torch.Tensor
        Cs values tensor to get number of Cs values
    device: torch.device
        Device to run the calculations on
    available_memory_gb: float, optional
        Available GPU/CPU memory in GB. If None, will try to detect.
        
    Returns
    -------
    batch_size: int
        Maximum batch size that will fit in memory
    """
    # Try to detect available memory if not provided
    if available_memory_gb is None:
        if device.type == 'cuda':
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Leave 10% memory as buffer
            available_memory_gb *= 0.9
        else:
            available_memory_gb = psutil.virtual_memory().total / (1024**3)
            # Leave 20% memory as buffer for CPU
            available_memory_gb *= 0.8

    # Calculate memory needed per projection
    h, w = micrograph_shape
    bytes_per_float32 = 4
    
    # Memory for one projection (accounting for all defocus and Cs values)
    # Shape will be (n_defoc, n_Cs, 1, h, w)
    proj_size = defoc_vals.shape[0] * Cs_vals.shape[0] * h * w * bytes_per_float32
    
    # Convert to GB
    proj_size_gb = proj_size / (1024**3)
    
    # Calculate maximum batch size
    max_batch_size = int(available_memory_gb / proj_size_gb)
    
    # Ensure batch size is at least 1
    batch_size = max(1, max_batch_size)
    
    print(f"Calculated batch size: {batch_size} (using {batch_size * proj_size_gb:.2f}GB of {available_memory_gb:.2f}GB available)")
    
    return batch_size