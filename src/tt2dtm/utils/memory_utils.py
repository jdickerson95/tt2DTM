

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
            # available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            available_memory_gb = torch.cuda.mem_get_info(device)[0] / (1024**3)
            # Leave 10% memory as buffer
            available_memory_gb *= 0.9
        else:
            available_memory_gb = psutil.virtual_memory().total / (1024**3)
            # Leave 20% memory as buffer for CPU
            available_memory_gb *= 0.8

    # Calculate memory needed per projection
    h, w = micrograph_shape
    bytes_per_float32 = 4
    bytes_per_complex64 = 8
    
    # Memory for one projection (accounting for all defocus and Cs values)
    # Shape will be (n_defoc, n_Cs, 1, h, w)
    proj_size = defoc_vals.shape[0] * Cs_vals.shape[0] * h * w * bytes_per_complex64
    
    # Convert to GB
    proj_size_gb = proj_size / (1024**3)

    #Needs to have double free for operations
    proj_size_gb *= 2
    
    # Calculate maximum batch size
    max_batch_size = int(available_memory_gb / proj_size_gb)
    
    # Ensure batch size is at least 1
    batch_size = max(1, max_batch_size)
    
    print(f"Calculated batch size: {batch_size} (using {batch_size * proj_size_gb:.2f}GB of {available_memory_gb:.2f}GB available)")
    
    return batch_size


def get_gpu_with_most_memory() -> torch.device:
    """
    Returns the CUDA device with the most available memory.
    
    Returns
    -------
    device: torch.device
        CUDA device with the most available memory, or CPU if no CUDA devices are available
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    n_gpus = torch.cuda.device_count()
    if n_gpus == 1:
        return torch.device('cuda:0')
    
    # Check available memory on each GPU
    available_memory = []
    for i in range(n_gpus):
        torch.cuda.set_device(i)
        available_memory.append(torch.cuda.mem_get_info()[0])
    
    # Get the GPU with most available memory
    best_gpu = available_memory.index(max(available_memory))
    return torch.device(f'cuda:{best_gpu}')