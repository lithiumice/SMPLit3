import torch
import numpy as np
from scipy.interpolate import interp1d
import numpy as np
import torch
from .konia_transform import (
    quaternion_to_angle_axis,
    angle_axis_to_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    rotation_matrix_to_angle_axis,
    angle_axis_to_rotation_matrix,
)


from pytorch3d.transforms import matrix_to_quaternion as m2q
from pytorch3d.transforms import quaternion_to_matrix as q2m
from pytorch3d.transforms import matrix_to_axis_angle as m2a
from pytorch3d.transforms import axis_angle_to_matrix as a2m
from pytorch3d.transforms import matrix_to_rotation_6d as m2s
from pytorch3d.transforms import rotation_6d_to_matrix as s2m
from pytorch3d.transforms import euler_angles_to_matrix as e2m
from pytorch3d.transforms import matrix_to_euler_angles as m2e
from pytorch3d.transforms import axis_angle_to_quaternion as a2q
from pytorch3d.transforms import quaternion_to_axis_angle as q2a


def s2a(x):
    return m2a(s2m(x))


def a2s(x):
    return m2s(a2m(x))


def qnormalize(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    return q / torch.norm(q, dim=-1, keepdim=True)


def qbetween(v0, v1):
    """
    find the quaternion used to rotate v0 to v1
    """
    assert v0.shape[-1] == 3, "v0 must be of the shape (*, 3)"
    assert v1.shape[-1] == 3, "v1 must be of the shape (*, 3)"

    v = torch.cross(v0, v1)
    w = torch.sqrt(
        (v0**2).sum(dim=-1, keepdim=True) * (v1**2).sum(dim=-1, keepdim=True)
    ) + (v0 * v1).sum(dim=-1, keepdim=True)
    return qnormalize(torch.cat([w, v], dim=-1))


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


# @torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


# @torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


# @torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:].clone()
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 0:1].clone() * t + xyz.cross(t, dim=-1)).view(shape)


# @torch.jit.script
def quat_angle(a, eps: float = 1e-6):
    shape = a.shape
    a = a.reshape(-1, 4)
    s = 2 * (a[:, 0] ** 2) - 1
    s = s.clamp(-1 + eps, 1 - eps)
    s = s.acos()
    return s.view(shape[:-1])


# @torch.jit.script
def quat_angle_diff(quat1, quat2):
    return quat_angle(quat_mul(quat1, quat_conjugate(quat2)))


# @torch.jit.script
def torch_safe_atan2(y, x, eps: float = 1e-6):
    y = y.clone()
    y[(y.abs() < eps) & (x.abs() < eps)] += eps
    return torch.atan2(y, x)


# @torch.jit.script
def ypr_euler_from_quat(
    q, handle_singularity: bool = False, eps: float = 1e-6, singular_eps: float = 1e-6
):
    """
    convert quaternion to yaw-pitch-roll euler angles
    """
    yaw_atany = 2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2])
    yaw_atanx = 1 - 2 * (q[..., 2] * q[..., 2] + q[..., 3] * q[..., 3])
    roll_atany = 2 * (q[..., 0] * q[..., 1] + q[..., 2] * q[..., 3])
    roll_atanx = 1 - 2 * (q[..., 1] * q[..., 1] + q[..., 2] * q[..., 2])
    yaw = torch_safe_atan2(yaw_atany, yaw_atanx, eps)
    pitch = torch.asin(
        torch.clamp(
            2 * (q[..., 0] * q[..., 2] - q[..., 1] * q[..., 3]),
            min=-1 + eps,
            max=1 - eps,
        )
    )
    roll = torch_safe_atan2(roll_atany, roll_atanx, eps)

    if handle_singularity:
        """handle two special cases"""
        test = q[..., 0] * q[..., 2] - q[..., 1] * q[..., 3]
        # north pole, pitch ~= 90 degrees
        np_ind = test > 0.5 - singular_eps
        if torch.any(np_ind):
            # print('ypr_euler_from_quat singularity -- north pole!')
            roll[np_ind] = 0.0
            pitch[np_ind].clamp_max_(0.5 * np.pi)
            yaw_atany = q[..., 3][np_ind]
            yaw_atanx = q[..., 0][np_ind]
            yaw[np_ind] = 2 * torch_safe_atan2(yaw_atany, yaw_atanx, eps)
        # south pole, pitch ~= -90 degrees
        sp_ind = test < -0.5 + singular_eps
        if torch.any(sp_ind):
            # print('ypr_euler_from_quat singularity -- south pole!')
            roll[sp_ind] = 0.0
            pitch[sp_ind].clamp_min_(-0.5 * np.pi)
            yaw_atany = q[..., 3][sp_ind]
            yaw_atanx = q[..., 0][sp_ind]
            yaw[sp_ind] = 2 * torch_safe_atan2(yaw_atany, yaw_atanx, eps)

    return torch.stack([roll, pitch, yaw], dim=-1)


# @torch.jit.script
def quat_from_ypr_euler(angles):
    """
    convert yaw-pitch-roll euler angles to quaternion
    """
    half_ang = angles * 0.5
    sin = torch.sin(half_ang)
    cos = torch.cos(half_ang)
    q = torch.stack(
        [
            cos[..., 0] * cos[..., 1] * cos[..., 2]
            + sin[..., 0] * sin[..., 1] * sin[..., 2],
            sin[..., 0] * cos[..., 1] * cos[..., 2]
            - cos[..., 0] * sin[..., 1] * sin[..., 2],
            cos[..., 0] * sin[..., 1] * cos[..., 2]
            + sin[..., 0] * cos[..., 1] * sin[..., 2],
            cos[..., 0] * cos[..., 1] * sin[..., 2]
            - sin[..., 0] * sin[..., 1] * cos[..., 2],
        ],
        dim=-1,
    )
    return q


def quat_between_two_vec(v1, v2, eps: float = 1e-6):
    """
    quaternion for rotating v1 to v2
    """
    orig_shape = v1.shape
    v1 = v1.reshape(-1, 3)
    v2 = v2.reshape(-1, 3)
    dot = (v1 * v2).sum(-1)
    cross = torch.cross(v1, v2, dim=-1)
    out = torch.cat([(1 + dot).unsqueeze(-1), cross], dim=-1)
    # handle v1 & v2 with same direction
    sind = dot > 1 - eps
    out[sind] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=v1.device)
    # handle v1 & v2 with opposite direction
    nind = dot < -1 + eps
    if torch.any(nind):
        vx = torch.tensor([1.0, 0.0, 0.0], device=v1.device)
        vxdot = (v1 * vx).sum(-1).abs()
        nxind = nind & (vxdot < 1 - eps)
        if torch.any(nxind):
            out[nxind] = angle_axis_to_quaternion(
                normalize(torch.cross(vx.expand_as(v1[nxind]), v1[nxind], dim=-1))
                * np.pi
            )
        # handle v1 & v2 with opposite direction and they are parallel to x axis
        pind = nind & (vxdot >= 1 - eps)
        if torch.any(pind):
            vy = torch.tensor([0.0, 1.0, 0.0], device=v1.device)
            out[pind] = angle_axis_to_quaternion(
                normalize(torch.cross(vy.expand_as(v1[pind]), v1[pind], dim=-1)) * np.pi
            )
    # normalize and reshape
    out = normalize(out).view(orig_shape[:-1] + (4,))
    return out


# @torch.jit.script
def get_yaw(q, eps: float = 1e-6):
    yaw_atany = 2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2])
    yaw_atanx = 1 - 2 * (q[..., 2] * q[..., 2] + q[..., 3] * q[..., 3])
    yaw = torch_safe_atan2(yaw_atany, yaw_atanx, eps)
    return yaw


# @torch.jit.script
def get_yaw_q(q):
    yaw = get_yaw(q)
    angle_axis = torch.cat(
        [torch.zeros(yaw.shape + (2,), device=q.device), yaw.unsqueeze(-1)], dim=-1
    )
    heading_q = angle_axis_to_quaternion(angle_axis)
    return heading_q


# @torch.jit.script
def get_heading(q, eps: float = 1e-6):
    # 提取x、y分量
    heading_atany = q[..., 3]
    heading_atanx = q[..., 0]
    heading = 2 * torch_safe_atan2(heading_atany, heading_atanx, eps)
    return heading


def get_heading_q(q):
    # 只保留x分量
    q_new = q.clone()
    q_new[..., 1] = 0
    q_new[..., 2] = 0
    q_new = normalize(q_new)
    return q_new


# @torch.jit.script
def heading_to_vec(h_theta):
    v = torch.stack([torch.cos(h_theta), torch.sin(h_theta)], dim=-1)
    return v


# @torch.jit.script
def vec_to_heading(h_vec):
    h_theta = torch_safe_atan2(h_vec[..., 1], h_vec[..., 0])
    return h_theta


# @torch.jit.script
def heading_to_quat(h_theta):
    angle_axis = torch.cat(
        [
            torch.zeros(h_theta.shape + (2,), device=h_theta.device),
            h_theta.unsqueeze(-1),
        ],
        dim=-1,
    )
    heading_q = angle_axis_to_quaternion(angle_axis)
    return heading_q


def deheading_quat(q, heading_q=None):
    if heading_q is None:
        # 去掉x分量的旋转
        heading_q = get_heading_q(q)
    dq = quat_mul(quat_conjugate(heading_q), q)
    return dq


# @torch.jit.script
def rotmat_to_rot6d(mat):
    rot6d = torch.cat([mat[..., 0], mat[..., 1]], dim=-1)
    return rot6d


def rot6d_to_rotmat(rot6d):
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]
    b1 = normalize(a1)
    b2 = normalize(a2 - (b1 * a2).sum(-1, keepdims=True) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    mat = torch.stack([b1, b2, b3], dim=-1)
    return mat


def angle_axis_to_rot6d(aa):
    return rotmat_to_rot6d(angle_axis_to_rotation_matrix(aa))


def rot6d_to_angle_axis(rot6d):
    return rotation_matrix_to_angle_axis(rot6d_to_rotmat(rot6d))


def quat_to_rot6d(q):
    return rotmat_to_rot6d(quaternion_to_rotation_matrix(q))


def rot6d_to_quat(rot6d):
    return rotation_matrix_to_quaternion(rot6d_to_rotmat(rot6d))


def make_transform(rot, trans, rot_type=None):
    if rot_type == "axis_angle":
        rot = angle_axis_to_rotation_matrix(rot)
    elif rot_type == "6d":
        rot = rot6d_to_rotmat(rot)
    transform = torch.eye(4).to(trans.device).repeat(rot.shape[:-2] + (1, 1))
    transform[..., :3, :3] = rot
    transform[..., :3, 3] = trans
    return transform


def transform_trans(transform_mat, trans):
    trans = torch.cat((trans, torch.ones_like(trans[..., [0]])), dim=-1)[..., None, :]
    while len(transform_mat.shape) < len(trans.shape):
        transform_mat = transform_mat.unsqueeze(-3)
    trans_new = torch.matmul(trans, transform_mat.transpose(-2, -1))[..., 0, :3]
    return trans_new


def transform_rot(transform_mat, rot):
    rot_qmat = angle_axis_to_rotation_matrix(rot)
    while len(transform_mat.shape) < len(rot_qmat.shape):
        transform_mat = transform_mat.unsqueeze(-3)
    rot_qmat_new = torch.matmul(transform_mat[..., :3, :3], rot_qmat)
    rot_new = rotation_matrix_to_angle_axis(rot_qmat_new)
    return rot_new


def inverse_transform(transform_mat):
    transform_inv = torch.zeros_like(transform_mat)
    transform_inv[..., :3, :3] = transform_mat[..., :3, :3].transpose(-2, -1)
    transform_inv[..., :3, 3] = -torch.matmul(
        transform_mat[..., :3, 3].unsqueeze(-2), transform_mat[..., :3, :3]
    ).squeeze(-2)
    transform_inv[..., 3, 3] = 1.0
    return transform_inv


def batch_compute_similarity_transform_torch(S1, S2):
    """
    This function is borrowed from https://github.com/mkocabas/VIBE/blob/c0c3f77d587351c806e901221a9dc05d1ffade4b/lib/utils/eval_utils.py#L199

    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    if len(S1.shape) > 3:
        orig_shape = S1.shape
        S1 = S1.reshape(-1, *S1.shape[-2:])
        S2 = S2.reshape(-1, *S2.shape[-2:])
    else:
        orig_shape = None

    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    if orig_shape is not None:
        S1_hat = S1_hat.reshape(orig_shape)

    return S1_hat


def rot_2d(xy, theta):
    rot_x = xy[..., 0] * torch.cos(theta) - xy[..., 1] * torch.sin(theta)
    rot_y = xy[..., 0] * torch.sin(theta) + xy[..., 1] * torch.cos(theta)
    rot_xy = torch.stack([rot_x, rot_y], dim=-1)
    return rot_xy


def traj_global2local(trans, orient_q, base_orient=[0.5, 0.5, 0.5, 0.5]):
    base_orient = torch.tensor(base_orient, device=orient_q.device)
    xy, z = trans[..., :2], trans[..., 2]
    orient_q = quat_mul(orient_q, quat_conjugate(base_orient).expand_as(orient_q))
    eulers = ypr_euler_from_quat(orient_q)
    roll, pitch, yaw = eulers[..., 0], eulers[..., 1], eulers[..., 2]
    d_xy = xy[1:] - xy[:-1]
    d_yaw = yaw[1:] - yaw[:-1]
    d_xy_yawcoord = rot_2d(d_xy, -yaw[:-1])
    d_xy_yawcoord = torch.cat(
        [xy[[0]], d_xy_yawcoord]
    )  # first element is global trans xy
    d_yaw = torch.cat([yaw[[0]], d_yaw])  # first element is global yaw
    local_traj = torch.stack(
        [d_xy_yawcoord[..., 0], d_xy_yawcoord[..., 1], z, roll, pitch, d_yaw], dim=-1
    )
    return local_traj


def traj_local2global(local_traj, base_orient=[0.5, 0.5, 0.5, 0.5]):
    base_orient = torch.tensor(base_orient, device=local_traj.device)
    d_xy_yawcoord, z = local_traj[..., :2], local_traj[..., 2]
    roll, pitch, d_yaw = local_traj[..., 3], local_traj[..., 4], local_traj[..., 5]
    yaw = torch.cumsum(d_yaw, dim=0)
    d_xy = d_xy_yawcoord.clone()
    d_xy[1:] = rot_2d(d_xy_yawcoord[1:], yaw[:-1])
    xy = torch.cumsum(d_xy, dim=0)
    trans = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
    eulers = torch.stack([roll, pitch, yaw], dim=-1)
    orient_q = quat_from_ypr_euler(eulers)
    orient_q = quat_mul(orient_q, base_orient.expand_as(orient_q))
    return trans, orient_q


# 标准化到y+轴方向
def traj_global2local_heading(
    trans, orient_q, base_orient=[0.5, 0.5, 0.5, 0.5], local_orient_type="6d"
):
    assert trans.shape[0] > 1
    base_orient = torch.tensor(base_orient, device=orient_q.device)
    xy, z = trans[..., :2], trans[..., 2]

    # 转一个固定的角度
    orient_q = quat_mul(orient_q, quat_conjugate(base_orient).expand_as(orient_q))

    # 提取x、y分量
    heading = get_heading(orient_q)

    # 只保留x分量
    heading_q = get_heading_q(orient_q)

    # orient_q减去heading_q，去掉x分量
    local_q = deheading_quat(orient_q, heading_q)

    if local_orient_type == "6d":
        local_orient = quat_to_rot6d(local_q)
    else:
        local_orient = local_q[..., :3]

    # xy位移，二维向量
    # 注意！第一个维度必须是T
    d_xy = xy[1:] - xy[:-1]

    # 角度值
    d_heading = heading[1:] - heading[:-1]
    d_heading = torch.cat([heading[[0]], d_heading])  # first element is global heading

    # 角度值转2维向量
    d_heading_vec = heading_to_vec(d_heading)

    # d_xy转到head的坐标系
    d_xy_yawcoord = rot_2d(d_xy, -heading[:-1])
    d_xy_yawcoord = torch.cat(
        [xy[[0]], d_xy_yawcoord]
    )  # first element is global trans xy
    # d_xy_yawcoord,z,local_orient,d_heading_vec=2+1+6+2
    local_traj = torch.cat(
        [d_xy_yawcoord[..., :2], z.unsqueeze(-1), local_orient, d_heading_vec], dim=-1
    )  # dim: 3 + 6 + 2 = 11
    # import ipdb;ipdb.set_trace()
    return local_traj


# local_traj： B，11
def traj_local2global_heading(
    local_traj,
    base_orient=[0.5, 0.5, 0.5, 0.5],
    deheading_local=False,
    local_orient_type="6d",
    local_heading=True,
):
    base_orient = torch.tensor(base_orient, device=local_traj.device)

    # xy在head坐标系下xy的角度偏移
    d_xy_yawcoord, z = local_traj[..., :2], local_traj[..., 2]
    local_orient, d_heading_vec = local_traj[..., 3:-2], local_traj[..., -2:]

    # 角度二维偏移转角度值
    d_heading = vec_to_heading(d_heading_vec)

    # 累加得到绝对的（x）角度值
    if local_heading:
        heading = torch.cumsum(d_heading, dim=0)
    else:
        heading = d_heading
    heading_q = heading_to_quat(heading)

    # head坐标系转到世界坐标系
    d_xy = d_xy_yawcoord.clone()
    d_xy[1:] = rot_2d(d_xy_yawcoord[1:], heading[:-1])
    xy = torch.cumsum(d_xy, dim=0)

    trans = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
    if local_orient_type == "6d":
        local_q = rot6d_to_quat(local_orient)
        if deheading_local:
            local_q = deheading_quat(local_q)
    else:
        local_q = torch.cat(
            [local_orient, torch.zeros_like(local_orient[..., [0]])], dim=-1
        )
        local_q = normalize(local_q)

    # head坐标系转到世界坐标系
    orient_q = quat_mul(heading_q, local_q)

    # 加回base_orient
    orient_q = quat_mul(orient_q, base_orient.expand_as(orient_q))
    return trans, orient_q


# # <=======
# def bs_quat_conjugate(a):
#     shape = a.shape
#     return torch.cat((a[..., 0:1], -a[..., 1:]), dim=-1).view(shape)

# def bs_traj_global2local_heading(trans, orient_q, base_orient=[0.5, 0.5, 0.5, 0.5], local_orient_type='6d'):
#     base_orient = torch.tensor(base_orient, device=orient_q.device)
#     xy, z = trans[..., :2], trans[..., 2]

#     # 转一个固定的角度
#     orient_q = quat_mul(orient_q, bs_quat_conjugate(base_orient).expand_as(orient_q))

#     # 提取x、y分量
#     heading = get_heading(orient_q)

#     # 只保留x分量
#     heading_q = get_heading_q(orient_q)

#     # orient_q减去heading_q，去掉x分量
#     local_q = deheading_quat(orient_q, heading_q)

#     if local_orient_type == '6d':
#         local_orient = quat_to_rot6d(local_q)
#     else:
#         local_orient = local_q[..., :3]

#     # xy位移，二维向量
#     # 注意！第一个维度必须是T
#     d_xy = xy[1:] - xy[:-1]

#     # 角度值
#     d_heading = heading[1:] - heading[:-1]
#     d_heading = torch.cat([heading[[0]], d_heading])    # first element is global heading

#     # 角度值转2维向量
#     d_heading_vec = heading_to_vec(d_heading)

#     # d_xy转到head的坐标系
#     d_xy_yawcoord = rot_2d(d_xy, -heading[:-1])
#     d_xy_yawcoord = torch.cat([xy[[0]], d_xy_yawcoord])     # first element is global trans xy
#     # d_xy_yawcoord,z,local_orient,d_heading_vec=2+1+6+2
#     local_traj = torch.cat([d_xy_yawcoord[..., :2], z.unsqueeze(-1), local_orient, d_heading_vec], dim=-1)  # dim: 3 + 6 + 2 = 11
#     # import ipdb;ipdb.set_trace()
#     return local_traj


def get_init_heading_q(orient, base_orient=[0.5, 0.5, 0.5, 0.5]):
    orient_nobase = quat_mul(
        orient[0],
        quat_conjugate(torch.tensor(base_orient, device=orient.device)).expand_as(
            orient[0]
        ),
    )
    heading_q = get_heading_q(orient_nobase)
    return heading_q


def convert_traj_world2heading(
    orient, trans, base_orient=[0.5, 0.5, 0.5, 0.5], apply_base_orient_after=False
):
    # tmp=quat_to_rot6d(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    # rot6d_to_rotmat(tmp)
    # [[0., 0., 1.],
    # [1., 0., 0.],
    # [0., 1., 0.]])
    # 转一个固定的角度
    orient_nobase = quat_mul(
        orient,
        quat_conjugate(torch.tensor(base_orient, device=orient.device)).expand_as(
            orient
        ),
    )
    # 提取朝向的x、y分量
    heading_q = get_heading_q(orient_nobase[0])
    inv_heading_q = quat_conjugate(heading_q).expand_as(orient_nobase)
    # 去掉朝向的x y分量
    orient_heading = quat_mul(inv_heading_q, orient_nobase)
    trans_local = trans.clone()
    # 第一帧归零
    trans_local[..., :2] -= trans[0, ..., :2]
    # 转换到head坐标
    trans_heading = quat_apply(inv_heading_q, trans_local)
    if apply_base_orient_after:
        orient_heading = quat_mul(
            orient_heading,
            torch.tensor(base_orient, device=orient_heading.device).expand_as(
                orient_heading
            ),
        )
    return orient_heading, trans_heading


def convert_traj_heading2world(
    orient, trans, init_heading, init_trans, base_orient=[0.5, 0.5, 0.5, 0.5]
):
    init_heading = init_heading.expand_as(orient)
    trans_local = quat_apply(init_heading, trans)
    trans_world = trans_local.clone()
    trans_world[..., :2] += init_trans[..., :2]
    orient_nobase = quat_mul(init_heading, orient)
    orient_world = quat_mul(
        orient_nobase, torch.tensor(base_orient, device=orient.device).expand_as(orient)
    )
    return orient_world, trans_world


def interp_orient_q_sep_heading(
    orient_q_vis, vis_frames, base_orient=[0.5, 0.5, 0.5, 0.5]
):
    device = orient_q_vis.device
    base_orient = torch.tensor(base_orient, device=device)
    orient_q_vis_rb = quat_mul(
        orient_q_vis, quat_conjugate(base_orient).expand_as(orient_q_vis)
    )
    heading_q = get_heading_q(orient_q_vis_rb)
    heading_vec = heading_to_vec(get_heading(orient_q_vis_rb))
    local_orient = quat_to_rot6d(deheading_quat(orient_q_vis_rb, heading_q))
    max_len = vis_frames.shape[0]
    vis_ind = torch.where(vis_frames)[0].cpu().numpy()
    # heading_vec
    f = interp1d(
        vis_ind,
        heading_vec.cpu().numpy(),
        axis=0,
        assume_sorted=True,
        fill_value="extrapolate",
    )
    new_val = f(np.arange(max_len, dtype=np.float32))
    heading_vec_interp = torch.tensor(new_val, device=device, dtype=torch.float32)
    # local_orient
    f = interp1d(
        vis_ind,
        local_orient.cpu().numpy(),
        axis=0,
        assume_sorted=True,
        fill_value="extrapolate",
    )
    new_val = f(np.arange(max_len, dtype=np.float32))
    local_orient_interp = torch.tensor(new_val, device=device, dtype=torch.float32)
    # final
    heading_q_interp = heading_to_quat(vec_to_heading(heading_vec_interp))
    local_q_interp = rot6d_to_quat(local_orient_interp)
    orient_q_interp = quat_mul(heading_q_interp, local_q_interp)
    orient_q_interp = quat_mul(orient_q_interp, base_orient.expand_as(orient_q_interp))
    return orient_q_interp
