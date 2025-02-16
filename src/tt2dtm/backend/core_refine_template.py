"""Backend functions related to correlating and refining particle stacks."""

import torch


def core_refine_template(
    particle_stack_dft: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    euler_angles: torch.Tensor,  # (3, N)
    defocus_values: torch.Tensor,  # (N,)
    projective_filters: torch.Tensor,  # (N, h, w)
    euler_angle_offsets: torch.Tensor,  # (3, k)
    defocus_offsets: torch.Tensor,  # (l,)
    batch_size: int = 1024,
    # TODO: additional arguments for cc --> z-score scaling
) -> None:
    """Core function to refine orientations and defoci of a set of particles.

    Parameters
    ----------
    particle_stack_dft : torch.Tensor
        The stack of particle real-Fourier transformed and un-fftshifted images.
        Shape of (N, H, W).
    template_dft : torch.Tensor
        The template volume to extract central slices from. Real-Fourier transformed
        and fftshifted.
    euler_angles : torch.Tensor
        The Euler angles for each particle in the stack. Shape of (3, N).
    defocus_values : torch.Tensor
        The defocus values for each particle in the stack. Shape of (N,).
        NOTE: Will likely also need kwargs to pass to the ctf filter function in here
        somewhere and support multiple distinct images.
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    euler_angle_offsets : torch.Tensor
        The Euler angle offsets to apply to each particle. Shape of (3, k).
    defocus_offsets : torch.Tensor
        The defocus offsets to search over for each particle. Shape of (l,).
    batch_size : int, optional
        The number of particle images to cross-correlate at once. Default is 1024.
        Larger sizes will consume more memory. If -1, then the entire stack will be
        cross-correlated at once.
    """
    # device = particle_stack_dft.device
    # num_particles, H, W = particle_stack_dft.shape
    # d, h, w = template_dft.shape
    # # account for RFFT
    # W = 2 * (W - 1)
    # w = 2 * (w - 1)

    # for i in range(0, num_particles, batch_size):
    #     # extract batch parameters
    #     for delta_df in defocus_offsets:
    #         # Recompute CTF filters for each particle's absolute defocus
    #         for delta_ea in euler_angle_offsets:
    #             # Rotate Euler angles by delta_ea
    #             # call cross_correlate_particle_stack and update best values
    #             pass

    raise NotImplementedError("This function is not yet implemented.")
