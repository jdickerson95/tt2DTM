
import torch
import torch.nn.functional as F

def is_efficient_size(n, radices):
    """Check if a number can be factored using only the given radices."""
    for radix in radices:
        while n % radix == 0:
            n //= radix
    return n == 1  # If n is reduced to 1, it's efficient

def next_efficient_size(n, radices):
    """Find the next number >= n that is efficient."""
    while not is_efficient_size(n, radices):
        n += 1
    return n

def calculate_optimal_padding(x, y, radices=(2, 3, 5, 7, 11, 13)):
    """
    Calculate the most efficient sizes to pad x and y to.
    
    Args:
        x (int): Current size of the first dimension.
        y (int): Current size of the second dimension.
        radices (tuple): Supported radices for FFT computation.

    Returns:
        tuple: Optimal sizes for x and y.
    """
    optimal_x = next_efficient_size(x, radices)
    optimal_y = next_efficient_size(y, radices)
    return optimal_x, optimal_y

def pad_volume(
    mrc_map: torch.Tensor, 
    pad_length: int,
):
    return F.pad(mrc_map, pad=[pad_length] * 6, mode='constant', value=0)

def pad_to_shape_2d(
        image: torch.Tensor,
        image_shape: tuple[int,int],
        shape: tuple[int,int],
        pad_val: float,
):
    x_pad = shape[1] - image_shape[1]
    y_pad = shape[0] - image_shape[0]
    p2d = (x_pad//2, x_pad//2 + x_pad%2, y_pad//2, y_pad//2 + y_pad%2)
    padded_image = F.pad(image, p2d, "constant", pad_val)
    return padded_image

def edge_mean_reduction_2d(image: torch.Tensor):
    """
    Calculate mean of edge pixels for batches of 2D images with arbitrary batch dimensions.
    
    Args:
        image: torch.Tensor with shape (..., H, W) where ... represents any number of batch dimensions
        
    Returns:
        torch.Tensor with shape (...) containing edge pixel averages for each image
    """
    
    # Extract edges while preserving batch dimensions
    top_edge = image[..., 0, :]          # shape: (..., W)
    bottom_edge = image[..., -1, :]      # shape: (..., W)
    left_edge = image[..., :, 0]         # shape: (..., H)
    right_edge = image[..., :, -1]       # shape: (..., H)
    
    # Stack all edges along a new dimension
    edge_pixels = torch.stack([
        top_edge.flatten(-1),    # flatten last dim (W) to combine with H
        bottom_edge.flatten(-1),
        left_edge.flatten(-1),
        right_edge.flatten(-1)
    ], dim=-1)  # shape: (..., num_edge_pixels, 4)
    
    # Calculate mean across all edge pixels
    return torch.mean(edge_pixels, dim=(-2, -1))  # shape: (...)