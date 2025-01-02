import einops
import torch
from eulerangles import euler2matrix
from torch_angular_search.angular_ranges import get_symmetry_ranges
from torch_angular_search.hopf_angles import get_uniform_euler_angles


def check_user_set_ranges(
    all_inputs: dict,
) -> bool:
    user_set = False
    if "phi_range" in all_inputs["angular_sampling"]:
        phi_min_temp = float(all_inputs["angular_sampling"]["phi_range"][0])
        phi_max_temp = float(all_inputs["angular_sampling"]["phi_range"][1])
        if phi_max_temp - phi_min_temp < 360.0:
            user_set = True
    if "theta_range" in all_inputs["angular_sampling"]:
        theta_min_temp = float(all_inputs["angular_sampling"]["theta_range"][0])
        theta_max_temp = float(all_inputs["angular_sampling"]["theta_range"][1])
        if theta_max_temp - theta_min_temp < 180.0:
            user_set = True
    if "psi_range" in all_inputs["angular_sampling"]:
        psi_min_temp = float(all_inputs["angular_sampling"]["psi_range"][0])
        psi_max_temp = float(all_inputs["angular_sampling"]["psi_range"][1])
        if psi_max_temp - psi_min_temp < 360.0:
            user_set = True
    return user_set


def get_user_ranges(
    all_inputs: dict,
) -> torch.Tensor:
    phi_min, phi_max = -180.0, 180.0
    theta_min, theta_max = 0.0, 180.0
    psi_min, psi_max = -180.0, 180.0
    if "phi_range" in all_inputs["angular_sampling"]:
        phi_min_temp = float(all_inputs["angular_sampling"]["phi_range"][0])
        phi_max_temp = float(all_inputs["angular_sampling"]["phi_range"][1])
        if phi_max_temp - phi_min_temp < 360.0:
            phi_min = phi_min_temp
            phi_max = phi_max_temp
    if "theta_range" in all_inputs["angular_sampling"]:
        theta_min_temp = float(all_inputs["angular_sampling"]["theta_range"][0])
        theta_max_temp = float(all_inputs["angular_sampling"]["theta_range"][1])
        if theta_max_temp - theta_min_temp < 180.0:
            theta_min = theta_min_temp
            theta_max = theta_max_temp
    if "psi_range" in all_inputs["angular_sampling"]:
        psi_min_temp = float(all_inputs["angular_sampling"]["psi_range"][0])
        psi_max_temp = float(all_inputs["angular_sampling"]["psi_range"][1])
        if psi_max_temp - psi_min_temp < 360.0:
            psi_min = psi_min_temp
            psi_max = psi_max_temp
    return torch.tensor(
        [[phi_min, phi_max], [theta_min, theta_max], [psi_min, psi_max]],
        dtype=torch.float64,
    )


def get_angular_ranges(
    all_inputs: dict,
):
    phi_min, phi_max = -180.0, 180.0
    theta_min, theta_max = 0.0, 180.0
    psi_min, psi_max = -180.0, 180.0
    user_set = check_user_set_ranges(all_inputs)
    if user_set:
        return get_user_ranges(all_inputs)
    else:
        angular_ranges = torch.tensor(
            [[phi_min, phi_max], [theta_min, theta_max], [psi_min, psi_max]],
            dtype=torch.float64,
        )
        if "symmetry" in all_inputs["angular_sampling"]:
            # Check for symmetry
            symmetry_order = all_inputs["angular_sampling"]["symmetry"][-1:]
            symmetry_group = all_inputs["angular_sampling"]["symmetry"][0]
            angular_ranges = get_symmetry_ranges(
                symmetry_group=symmetry_group, symmetry_order=symmetry_order
            )

        return angular_ranges


def get_euler_angles(
    all_inputs: dict,
):
    angular_ranges = get_angular_ranges(all_inputs)
    in_plane_step = float(all_inputs["angular_sampling"]["in_plane_step"])
    out_of_plane_step = float(all_inputs["angular_sampling"]["out_of_plane_step"])
    phi_range = einops.rearrange(
        angular_ranges[0], "n -> 1 n"
    )  # rearrange for function
    theta_range = einops.rearrange(angular_ranges[1], "n -> 1 n")
    psi_range = einops.rearrange(angular_ranges[2], "n -> 1 n")
    all_angles = get_uniform_euler_angles(
        in_plane_step=in_plane_step,
        out_of_plane_step=out_of_plane_step,
        phi_ranges=phi_range,
        theta_ranges=theta_range,
        psi_ranges=psi_range,
    )
    return all_angles


def euler_to_rotation_matrix(euler_angles: torch.Tensor):
    rotation_matrices = []
    for euler_angle_tensor in euler_angles:
        rotation_matrices.append(
            torch.from_numpy(
                euler2matrix(
                    euler_angle_tensor.detach().numpy(),
                    axes="zyz",
                    intrinsic=True,
                    right_handed_rotation=False,
                )
            ).float()
        )

    rotation_matrices = torch.stack(rotation_matrices, dim=0).squeeze(0)
    return rotation_matrices

def get_rotation_matrices(
    all_inputs: dict,
):
    euler_angles = get_euler_angles(all_inputs)
    rotation_matrices = []
    for euler_angle_tensor in euler_angles:
        rotation_matrices.append(
            torch.from_numpy(
                euler2matrix(
                    euler_angle_tensor.detach().numpy(),
                    axes="zyz",
                    intrinsic=True,
                    right_handed_rotation=False,
                )
            ).float()
        )

    rotation_matrices = torch.stack(rotation_matrices, dim=0).squeeze(0)
    return rotation_matrices
