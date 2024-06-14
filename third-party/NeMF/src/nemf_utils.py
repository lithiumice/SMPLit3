import os

import numpy as np
import plyfile
import torch
import torch.nn.functional as F
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import holden.BVH as BVH
from holden.Animation import Animation
from holden.Quaternions import Quaternions

SMPL_JOINTS = {
    'Pelvis': 0,
    'L_Hip': 1, 'L_Knee': 4, 'L_Ankle': 7, 'L_Foot': 10,
    'R_Hip': 2, 'R_Knee': 5, 'R_Ankle': 8, 'R_Foot': 11,
    'Spine1': 3, 'Spine2': 6, 'Spine3': 9, 'Neck': 12, 'Head': 15,
    'L_Collar': 13, 'L_Shoulder': 16, 'L_Elbow': 18, 'L_Wrist': 20, 'L_Hand': 22,
    'R_Collar': 14, 'R_Shoulder': 17, 'R_Elbow': 19, 'R_Wrist': 21, 'R_Hand': 23
}

FOOT_IDX = [SMPL_JOINTS['L_Ankle'], SMPL_JOINTS['R_Ankle'], SMPL_JOINTS['L_Foot'], SMPL_JOINTS['R_Foot']]

CONTACTS_IDX = [SMPL_JOINTS['L_Ankle'], SMPL_JOINTS['R_Ankle'],
                SMPL_JOINTS['L_Foot'], SMPL_JOINTS['R_Foot'],
                SMPL_JOINTS['L_Wrist'], SMPL_JOINTS['R_Wrist'],
                SMPL_JOINTS['L_Knee'], SMPL_JOINTS['R_Knee']]


def align_joints(tensor, smpl_to_bvh=True):
    """
    Args:
        tensor (T x J x D): The 3D torch tensor we need to process.

    Returns:
        The 3D tensor whose joint order is compatible for bvh export.
    """
    order = list(SMPL_JOINTS.values())
    result = tensor.clone() if isinstance(tensor, torch.Tensor) else tensor.copy()
    for i in range(len(order)):
        if smpl_to_bvh:
            result[:, i] = tensor[:, order[i]]
        else:
            result[:, order[i]] = tensor[:, i]

    return result


def slerp(quat, trans, key_times, times, mask=True):
    """
    Args:
        quat: (T x J x 4)
        trans: (T x 3)
    """
    if mask:
        quat = c2c(quat[key_times])
        trans = c2c(trans[key_times])
    else:
        quat = c2c(quat)
        trans = c2c(trans)

    quats = []
    for j in range(quat.shape[1]):
        key_rots = R.from_quat(quat[:, j])
        s = Slerp(key_times, key_rots)
        interp_rots = s(times)
        quats.append(interp_rots.as_quat())
    slerp_quat = np.stack(quats, axis=1)

    lerp_trans = np.zeros((len(times), 3))
    for i in range(3):
        lerp_trans[:, i] = np.interp(times, key_times, trans[:, i])

    return slerp_quat, lerp_trans


def compute_orient_angle(matrix, traj):
    """
    Args:
        matrix (N x T x 3 x 3): The 3D rotation matrix at the root joint
        traj (N x T x 3): The trajectory to align
    """
    forward = matrix[:, :, :, 2].clone()
    forward[:, :, 2] = 0
    forward = F.normalize(forward, dim=-1)  # normalized forward vector (N, T, 3)

    traj[:, :, 2] = 0  # make sure the trajectory is projected to the plane

    # first steps is forward diff
    init_tan = traj[:, 1:2] - traj[:, :1]
    # middle steps are second order
    middle_tan = (traj[:, 2:] - traj[:, 0:-2]) / 2
    # last step is backward diff
    final_tan = traj[:, -1:] - traj[:, -2:-1]

    tangent = torch.cat([init_tan, middle_tan, final_tan], dim=1)
    tangent = F.normalize(tangent, dim=-1)  # normalized tangent vector (N, T, 3)

    cos = torch.sum(forward * tangent, dim=-1)

    return cos


def compute_trajectory(velocity, up, origin, dt, up_axis='z'):
    """
    Args:
        velocity: (B, T, 3)
        up: (B, T)
        origin: (B, 3)
        up_axis: x, y, or z

    Returns:
        trajectory: (B, T, 3)
    """
    ordermap = {
        'x': 0,
        'y': 1,
        'z': 2,
    }
    v_axis = [x for x in ordermap.values() if x != ordermap[up_axis]]

    origin = origin.unsqueeze(1)  # (B, 3) => (B, 1, 3)
    trajectory = origin.repeat(1, up.shape[1], 1)  # (B, 1, 3) => (B, T, 3)

    for t in range(1, up.shape[1]):
        trajectory[:, t, v_axis[0]] = trajectory[:, t - 1, v_axis[0]] + velocity[:, t - 1, v_axis[0]] * dt
        trajectory[:, t, v_axis[1]] = trajectory[:, t - 1, v_axis[1]] + velocity[:, t - 1, v_axis[1]] * dt

    trajectory[:, :, ordermap[up_axis]] = up

    return trajectory


def build_canonical_frame(forward, up_axis='z'):
    """
    Args:
        forward: (..., 3)

    Returns:
        frame: (..., 3, 3)
    """
    forward[..., 'xyz'.index(up_axis)] = 0
    forward = F.normalize(forward, dim=-1)  # normalized forward vector

    up = torch.zeros_like(forward)
    up[..., 'xyz'.index(up_axis)] = 1  # normalized up vector
    right = torch.cross(up, forward)
    frame = torch.stack((right, up, forward), dim=-1)  # canonical frame

    return frame


def estimate_linear_velocity(data_seq, dt):
    '''
    Given some batched data sequences of T timesteps in the shape (B, T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    The first and last frames are with forward and backward first-order
    differences, respectively
    - h : step size
    '''
    # first steps is forward diff (t+1 - t) / dt
    init_vel = (data_seq[:, 1:2] - data_seq[:, :1]) / dt
    # middle steps are second order (t+1 - t-1) / 2dt
    middle_vel = (data_seq[:, 2:] - data_seq[:, 0:-2]) / (2 * dt)
    # last step is backward diff (t - t-1) / dt
    final_vel = (data_seq[:, -1:] - data_seq[:, -2:-1]) / dt

    vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    return vel_seq


def estimate_angular_velocity(rot_seq, dt):
    '''
    Given a batch of sequences of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (B, T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_linear_velocity(rot_seq, dt)
    R = rot_seq
    RT = R.transpose(-1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector by averaging symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = torch.stack([w_x, w_y, w_z], axis=-1)
    return w


def normalize(tensor, mean=None, std=None):
    """
    Args:
        tensor: (B, T, ...)

    Returns:
        normalized tensor with 0 mean and 1 standard deviation, std, mean
    """
    if mean is None or std is None:
        # std, mean = torch.std_mean(tensor, dim=0, unbiased=False, keepdim=True)
        std, mean = torch.std_mean(tensor, dim=(0, 1), unbiased=False, keepdim=True)
        std[std == 0.0] = 1.0

        return (tensor - mean) / std, mean, std

    return (tensor - mean) / std


def denormalize(tensor, mean, std):
    """
    Args:
        tensor: B x T x D
        mean:
        std:
    """
    return tensor * std + mean


def export_bvh_animation(rotations, positions, offsets, parents, output_dir, prefix, joint_names, fps):
    """
    Args:
        rotations: quaternions of the shape (B, T, J, 4)
        positions: global translations of the shape (B, T, 3)
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in range(rotations.shape[0]):
        rotation = align_joints(rotations[i])
        position = positions[i]
        position = position.unsqueeze(1)
        anim = Animation(Quaternions(c2c(rotation)), c2c(position), None, offsets=offsets, parents=parents)
        BVH.save(os.path.join(output_dir, f'{prefix}_{i}.bvh'), anim, names=joint_names, frametime=1 / fps)


def export_ply_trajectory(points, color, ply_fname):
    v = []
    for p in points:
        v += [(p[0], p[1], p[2], color[0], color[1], color[2])]
    v = np.array(v, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el_v = plyfile.PlyElement.describe(v, 'vertex')

    e = np.empty(len(points) - 1, dtype=[('vertex1', 'i4'), ('vertex2', 'i4')])
    edge_data = np.array([[i, i + 1] for i in range(len(points) - 1)], dtype='i4')
    e['vertex1'] = edge_data[:, 0]
    e['vertex2'] = edge_data[:, 1]
    el_e = plyfile.PlyElement.describe(e, 'edge')

    plyfile.PlyData([el_v, el_e], text=True).write(ply_fname)
