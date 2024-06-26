import pyvista
import torch
import numpy as np
import platform

# from pyvista.plotting.tools import parse_color
from vtk import vtkTransform
import pdb
from math import pi
import time
from .torch_transform import (
    quat_apply,
    quat_between_two_vec,
    quaternion_to_angle_axis,
    angle_axis_to_quaternion,
)
from .vis_pyvista import PyvistaVisualizer
from .smpl import SMPL, SMPL_MODEL_DIR
from .vis import make_checker_board_texture, get_color_palette


class SMPLActor:

    def __init__(self, pl, verts, faces, color="#FF8A82", visible=True):
        self.pl = pl
        self.verts = verts
        self.face = faces
        self.mesh = pyvista.PolyData(verts, faces)
        self.actor = self.pl.add_mesh(
            self.mesh, color=color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1
        )
        self.set_visibility(visible)

    def update_verts(self, new_verts):
        self.mesh.points[...] = new_verts
        self.mesh.compute_normals(inplace=True)

    def set_opacity(self, opacity):
        self.actor.GetProperty().SetOpacity(opacity)

    def set_visibility(self, flag):
        self.actor.SetVisibility(flag)

    def set_color(self, color):
        rgb_color = parse_color(color)
        rgb_color = pyvista.ploting.tools.Color(color)
        self.actor.GetProperty().SetColor(rgb_color)


class SkeletonActor:

    def __init__(
        self,
        pl,
        joint_parents,
        joint_color="green",
        bone_color="yellow",
        joint_radius=0.03,
        bone_radius=0.02,
        visible=True,
    ):
        self.pl = pl
        self.joint_parents = joint_parents
        self.joint_meshes = []
        self.joint_actors = []
        self.bone_meshes = []
        self.bone_actors = []
        self.bone_pairs = []
        for j, pa in enumerate(self.joint_parents):
            # joint
            joint_mesh = pyvista.Sphere(
                radius=joint_radius,
                center=(0, 0, 0),
                theta_resolution=10,
                phi_resolution=10,
            )
            # joint_actor = self.pl.add_mesh(joint_mesh, color=joint_color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
            joint_actor = self.pl.add_mesh(
                joint_mesh,
                color=joint_color,
                ambient=0.3,
                diffuse=0.5,
                specular=0.8,
                specular_power=5,
                smooth_shading=True,
            )
            self.joint_meshes.append(joint_mesh)
            self.joint_actors.append(joint_actor)
            # bone
            if pa >= 0:
                bone_mesh = pyvista.Cylinder(
                    radius=bone_radius,
                    center=(0, 0, 0),
                    direction=(0, 0, 1),
                    resolution=30,
                )
                # bone_actor = self.pl.add_mesh(bone_mesh, color=bone_color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
                bone_actor = self.pl.add_mesh(
                    bone_mesh,
                    color=bone_color,
                    ambient=0.3,
                    diffuse=0.5,
                    specular=0.8,
                    specular_power=5,
                    smooth_shading=True,
                )
                self.bone_meshes.append(bone_mesh)
                self.bone_actors.append(bone_actor)
                self.bone_pairs.append((j, pa))
        self.set_visibility(visible)

    def update_joints(self, jpos):
        # joint
        for actor, pos in zip(self.joint_actors, jpos):
            trans = vtkTransform()
            trans.Translate(*pos)
            actor.SetUserTransform(trans)
        # bone
        vec = []
        for actor, (j, pa) in zip(self.bone_actors, self.bone_pairs):
            vec.append((jpos[j] - jpos[pa]))
        vec = np.stack(vec)
        dist = np.linalg.norm(vec, axis=-1)
        vec = torch.tensor(vec / dist[..., None])
        aa = quaternion_to_angle_axis(
            quat_between_two_vec(torch.tensor([0.0, 0.0, 1.0]).expand_as(vec), vec)
        ).numpy()
        angle = np.linalg.norm(aa, axis=-1, keepdims=True)
        axis = aa / (angle + 1e-6)

        for actor, (j, pa), angle_i, axis_i, dist_i in zip(
            self.bone_actors, self.bone_pairs, angle, axis, dist
        ):
            trans = vtkTransform()
            trans.Translate(*(jpos[pa] + jpos[j]) * 0.5)
            trans.RotateWXYZ(np.rad2deg(angle_i), *axis_i)
            trans.Scale(1, 1, dist_i)
            actor.SetUserTransform(trans)

    def set_opacity(self, opacity):
        for actor in self.joint_actors:
            actor.GetProperty().SetOpacity(opacity)
        for actor in self.bone_actors:
            actor.GetProperty().SetOpacity(opacity)

    def set_visibility(self, flag):
        for actor in self.joint_actors:
            actor.SetVisibility(flag)
        for actor in self.bone_actors:
            actor.SetVisibility(flag)

    def set_color(self, color):
        rgb_color = parse_color(color)
        for actor in self.joint_actors:
            actor.GetProperty().SetColor(rgb_color)
        for actor in self.jbone_actors:
            actor.GetProperty().SetColor(rgb_color)


def get_transform(new_pos, new_dir):
    trans = vtkTransform()
    trans.Translate(new_pos)
    new_dir = torch.from_numpy(new_dir).float()
    aa = quaternion_to_angle_axis(
        quat_between_two_vec(torch.tensor([0.0, 0.0, 1.0]).expand_as(new_dir), new_dir)
    ).numpy()
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = aa / (angle + 1e-6)
    trans.RotateWXYZ(np.rad2deg(angle), *axis)
    return trans


class RacketActor:

    def __init__(self, pl, sport="tennis", debug=True):
        self.pl = pl
        self.sport = sport
        self.debug = debug
        if self.sport == "badminton":
            self.net_mesh = pyvista.Cylinder(
                center=(0, 0, 0), radius=0.25 / 2, height=0.01, direction=(0, 0, 1)
            )
            self.net_mesh.active_t_coords *= 1000
            tex = pyvista.numpy_to_texture(
                make_checker_board_texture("#FFFFFF", "#AAAAAA", width=10)
            )
            self.net_actor = self.pl.add_mesh(
                self.net_mesh,
                texture=tex,
                ambient=0.2,
                diffuse=0.8,
                opacity=0.1,
                smooth_shading=True,
            )

            self.head_mesh = pyvista.Tube(
                pointa=(0, 0, -0.005), pointb=(0, 0, 0.005), radius=0.25 / 2
            )
            self.head_actor = self.pl.add_mesh(
                self.head_mesh,
                color="black",
                ambient=0.3,
                diffuse=0.5,
                smooth_shading=True,
            )

            self.shaft_mesh = pyvista.Cylinder(
                center=(0, 0, 0), radius=0.005, height=0.25, direction=(0, 0, 1)
            )
            self.shaft_actor = self.pl.add_mesh(
                self.shaft_mesh,
                color="black",
                ambient=0.3,
                diffuse=0.5,
                smooth_shading=True,
            )

            self.handle_mesh = pyvista.Cylinder(
                center=(0, 0, 0), radius=0.0254 / 2, height=0.15, direction=(0, 0, 1)
            )
            self.handle_actor = self.pl.add_mesh(
                self.handle_mesh,
                color="#AAAAAA",
                ambient=0.3,
                diffuse=0.5,
                smooth_shading=True,
            )

            self.actors = [
                self.head_actor,
                self.net_actor,
                self.shaft_actor,
                self.handle_actor,
            ]
        elif self.sport == "tennis":
            self.net_mesh = pyvista.Cylinder(
                center=(0, 0, 0), radius=0.15, height=0.01, direction=(0, 0, 1)
            )
            self.net_mesh.active_t_coords *= 1000
            tex = pyvista.numpy_to_texture(
                make_checker_board_texture("#FFFFFF", "#AAAAAA", width=10)
            )
            self.net_actor = self.pl.add_mesh(
                self.net_mesh,
                texture=tex,
                ambient=0.2,
                diffuse=0.8,
                opacity=0.1,
                smooth_shading=True,
            )

            self.head_mesh = pyvista.Tube(
                pointa=(0, 0, -0.01), pointb=(0, 0, 0.01), radius=0.15
            )
            self.head_actor = self.pl.add_mesh(
                self.head_mesh,
                color="black",
                ambient=0.3,
                diffuse=0.5,
                smooth_shading=True,
            )

            self.shaft_left_mesh = pyvista.Cylinder(
                center=(0, 0, 0),
                radius=0.015 / 2,
                height=0.15 / np.cos(np.pi / 10),
                direction=(0, 0, 1),
            )
            self.shaft_left_actor = self.pl.add_mesh(
                self.shaft_left_mesh,
                color="black",
                ambient=0.3,
                diffuse=0.5,
                smooth_shading=True,
            )

            self.shaft_right_mesh = pyvista.Cylinder(
                center=(0, 0, 0),
                radius=0.015 / 2,
                height=0.15 / np.cos(np.pi / 10),
                direction=(0, 0, 1),
            )
            self.shaft_right_actor = self.pl.add_mesh(
                self.shaft_right_mesh,
                color="black",
                ambient=0.3,
                diffuse=0.5,
                smooth_shading=True,
            )

            self.handle_mesh = pyvista.Cylinder(
                center=(0, 0, 0), radius=0.03 / 2, height=0.2, direction=(0, 0, 1)
            )
            self.handle_actor = self.pl.add_mesh(
                self.handle_mesh,
                color="black",
                ambient=0.3,
                diffuse=0.5,
                smooth_shading=True,
            )

            self.actors = [
                self.head_actor,
                self.net_actor,
                self.shaft_left_actor,
                self.shaft_right_actor,
                self.handle_actor,
            ]

            if self.debug:
                self.normal_mesh = pyvista.Cylinder(
                    center=(0, 0, 0.1), radius=0.01, height=0.2, direction=(0, 0, 1)
                )
                self.normal_actor = self.pl.add_mesh(
                    self.normal_mesh, color="red", diffuse=1, smooth_shading=True
                )
                self.actors += [self.normal_actor]

    def update_racket(self, params):
        if self.sport == "badminton":
            self.head_actor.SetUserTransform(
                get_transform(
                    params["head_center"] + params["root"], params["racket_normal"]
                )
            )
            self.net_actor.SetUserTransform(
                get_transform(
                    params["head_center"] + params["root"], params["racket_normal"]
                )
            )
            self.shaft_actor.SetUserTransform(
                get_transform(
                    params["shaft_center"] + params["root"], params["racket_dir"]
                )
            )
            self.handle_actor.SetUserTransform(
                get_transform(
                    params["handle_center"] + params["root"], params["racket_dir"]
                )
            )
        elif self.sport == "tennis":
            self.head_actor.SetUserTransform(
                get_transform(
                    params["head_center"] + params["root"], params["racket_normal"]
                )
            )
            self.net_actor.SetUserTransform(
                get_transform(
                    params["head_center"] + params["root"], params["racket_normal"]
                )
            )
            self.shaft_left_actor.SetUserTransform(
                get_transform(
                    params["shaft_left_center"] + params["root"],
                    params["shaft_left_dir"],
                )
            )
            self.shaft_right_actor.SetUserTransform(
                get_transform(
                    params["shaft_right_center"] + params["root"],
                    params["shaft_right_dir"],
                )
            )
            self.handle_actor.SetUserTransform(
                get_transform(
                    params["handle_center"] + params["root"], params["racket_dir"]
                )
            )
            if self.debug:
                self.normal_actor.SetUserTransform(
                    get_transform(
                        params["head_center"] + params["root"], params["racket_normal"]
                    )
                )

    def set_visibility(self, flag):
        for actor in self.actors:
            actor.SetVisibility(flag)


class TargetRecoveryActor:

    def __init__(self, pl):
        self.pl = pl
        self.marker_mesh = pyvista.Disc(center=[0, 0, 0], inner=0.1, outer=0.2)
        self.actor = self.pl.add_mesh(
            self.marker_mesh, color="red", ambient=0.3, diffuse=0.5, smooth_shading=True
        )

    def update_target(self, pos):
        trans = vtkTransform()
        trans.Translate([pos[0], pos[1], 0.01])
        self.actor.SetUserTransform(trans)

    def set_visibility(self, flag):
        self.actor.SetVisibility(flag)


class TargetReactionActor:

    def __init__(self, pl):
        self.pl = pl
        self.outer_mesh = pyvista.Sphere(center=[0, 0, 0], radius=0.2)
        self.actor_outer = self.pl.add_mesh(
            self.outer_mesh,
            color="orange",
            ambient=0.3,
            diffuse=0.5,
            smooth_shading=True,
            opacity=0.5,
        )
        self.inner_mesh = pyvista.Sphere(center=[0, 0, 0], radius=0.2)
        self.actor_inner = self.pl.add_mesh(
            self.inner_mesh, color="red", ambient=0.3, diffuse=0.5, smooth_shading=True
        )

    def update_target(self, pos, time=None):
        trans = vtkTransform()
        trans.Translate([pos[0], pos[1], pos[2]])
        self.actor_outer.SetUserTransform(trans)
        trans = vtkTransform()
        trans.Translate([pos[0], pos[1], pos[2]])
        trans.Scale([time, time, time])
        self.actor_inner.SetUserTransform(trans)

    def set_visibility(self, flag):
        self.actor_inner.SetVisibility(flag)
        self.actor_outer.SetVisibility(flag)


class BallActor:

    def __init__(
        self,
        pl,
        sport="tennis",
        color="red",
        blur=False,
        num_exposure=30,
        real_shadow=False,
    ):
        self.pl = pl
        self.sport = sport
        self.blur = blur
        self.num_exposure = num_exposure
        self.real_shadow = real_shadow
        if sport == "tennis":
            ball_mesh = pyvista.Sphere(center=[0, 0, 0], radius=0.05)
            self.actor = self.pl.add_mesh(
                ball_mesh, color=color, ambient=0.3, diffuse=1, smooth_shading=True
            )

            # for motion blur
            if self.blur:
                self.actors = []
                for i in range(self.num_exposure):
                    ball_mesh = pyvista.Sphere(center=[0, 0, 0], radius=0.05)
                    self.actors += [
                        self.pl.add_mesh(
                            ball_mesh,
                            color=color,
                            ambient=0.3,
                            diffuse=0.8,
                            smooth_shading=True,
                            opacity=5.0 / num_exposure if not real_shadow else 1,
                        )
                    ]

            if not self.real_shadow:
                shadow_mesh = pyvista.Circle(radius=0.05)
                self.shadow_actor = self.pl.add_mesh(
                    shadow_mesh, color="#101010", diffuse=1, smooth_shading=True
                )

                if self.blur:
                    self.shadow_actors = []
                    for i in range(self.num_exposure):
                        shadow_mesh = pyvista.Circle(radius=0.05)
                        self.shadow_actors += [
                            self.pl.add_mesh(
                                shadow_mesh,
                                color="#101010",
                                diffuse=1,
                                smooth_shading=True,
                            )
                        ]
        else:
            NotImplemented
        self.set_visibility(False)

    def update_ball(self, params):
        if params is None:
            self.set_visibility(False)
            return

        if self.blur and params.get("pos_blur") is not None:
            for i in range(self.num_exposure):
                trans = vtkTransform()
                pos = params["pos_blur"][i]
                trans.Translate([pos[0], pos[1], pos[2]])
                self.actors[i].SetUserTransform(trans)
                self.actors[i].SetVisibility(True)

                if not self.real_shadow:
                    trans = vtkTransform()
                    trans.Translate([pos[0], pos[1], 0])
                    self.shadow_actors[i].SetUserTransform(trans)
                    self.shadow_actors[i].SetVisibility(True)
        else:
            trans = vtkTransform()
            pos = params["pos"]
            trans.Translate([pos[0], pos[1], pos[2]])
            self.actor.SetUserTransform(trans)
            self.actor.SetVisibility(True)

            # self.ang_vel_actor.SetUserTransform(get_transform(params['pos'].cpu().numpy(), params['ang_vel'].cpu().numpy()))
            # self.ang_vel_actor.SetVisibility(True)

            if not self.real_shadow:
                trans = vtkTransform()
                trans.Translate([pos[0], pos[1], 0.01])
                self.shadow_actor.SetUserTransform(trans)
                self.shadow_actor.SetVisibility(True)

    def set_visibility(self, flag):
        self.actor.SetVisibility(flag)
        # self.ang_vel_actor.SetVisibility(flag)
        if self.blur:
            for actor in self.actors:
                actor.SetVisibility(flag)
        if not self.real_shadow:
            self.shadow_actor.SetVisibility(flag)
            if self.blur:
                for actor in self.shadow_actors:
                    actor.SetVisibility(flag)


class TargetBounceActor:

    def __init__(self, pl):
        self.pl = pl
        self.marker_mesh = pyvista.Disc(center=[0, 0, 0], inner=0.1, outer=0.2)
        self.actor = self.pl.add_mesh(
            self.marker_mesh, color="red", ambient=0.3, diffuse=0.5, smooth_shading=True
        )
        self.set_visibility(False)

    def update_target(self, pos):
        trans = vtkTransform()
        trans.Translate([pos[0], pos[1], 0.01])
        self.actor.SetUserTransform(trans)
        self.set_visibility(True)

    def set_visibility(self, flag):
        self.actor.SetVisibility(flag)


class SportVisualizer(PyvistaVisualizer):

    def __init__(
        self,
        show_smpl=False,
        show_skeleton=True,
        show_racket=False,
        show_target=False,
        show_ball=False,
        show_ball_target=False,
        show_stats=True,
        track_first_actor=False,
        track_ball=False,
        enable_shadow=False,
        gender="male",
        correct_root_height=False,
        device=torch.device("cpu"),
        **kwargs
    ):

        super().__init__(**kwargs)
        self.show_smpl = show_smpl
        self.show_skeleton = show_skeleton
        self.show_racket = show_racket
        self.show_target = show_target
        self.show_ball = show_ball
        self.show_ball_target = show_ball_target
        self.show_stats = show_stats
        self.track_first_actor = track_first_actor
        self.track_ball = track_ball
        self.enable_shadow = enable_shadow
        self.correct_root_height = correct_root_height
        self.camera = "front"
        self.sport = "tennis"

        self.smpl = SMPL(SMPL_MODEL_DIR, create_transl=False, gender=gender).to(device)
        faces = self.smpl.faces.copy()
        self.smpl_faces = faces = np.hstack([np.ones_like(faces[:, [0]]) * 3, faces])
        self.smpl_joint_parents = self.smpl.parents.cpu().numpy()
        self.smpl_verts = None
        self.smpl_joints = None
        self.racket_params = None
        self.device = device

        self.forward = False

    def setup_animation(self, smpl_seq=None, racket_seq=None, ball_seq=None):
        self.smpl_seq = smpl_seq
        self.smpl_verts = None

        if "joint_rot" in smpl_seq:
            joint_rot = smpl_seq[
                "joint_rot"
            ]  # num_actor x num_frames x (num_joints x 3)
            trans = smpl_seq["trans"]  # num_actor x num_frames x 3

            self.smpl_motion = self.smpl(
                global_orient=joint_rot[..., :3].view(-1, 3),
                body_pose=joint_rot[..., 3:].view(-1, 69),
                betas=smpl_seq["betas"]
                .view(-1, 1, 10)
                .expand(-1, joint_rot.shape[1], 10)
                .reshape(-1, 10),
                root_trans=trans.view(-1, 3),
                return_full_pose=True,
                orig_joints=True,
            )

            self.smpl_verts = self.smpl_motion.vertices.reshape(
                *joint_rot.shape[:-1], -1, 3
            )
            if "joint_pos" not in smpl_seq:
                self.smpl_joints = self.smpl_motion.joints.reshape(
                    *joint_rot.shape[:-1], -1, 3
                )
                # set all 0 if joint rot is all 0 (invalid pose)
                num_actors, num_frames = self.smpl_joints.shape[:2]
                for i in range(num_actors):
                    for j in range(num_frames):
                        if joint_rot[i, j].sum() == 0:
                            self.smpl_joints[i, j, :, :] = 0

        if "joint_pos" in smpl_seq:
            joints = smpl_seq["joint_pos"]  # num_actor x num_frames x num_joints x 3
            trans = smpl_seq["trans"]  # num_actor x num_frames x 3

            # orient is None for hybrIK since joints already has global orentation
            orient = smpl_seq["orient"]

            joints_world = joints
            if orient is not None:
                joints_world = torch.cat(
                    [torch.zeros_like(joints[..., :3]), joints], dim=-1
                ).view(*joints.shape[:-1], -1, 3)
                orient_q = (
                    angle_axis_to_quaternion(orient)
                    .unsqueeze(-2)
                    .expand(joints.shape[:-1] + (4,))
                )
                joints_world = quat_apply(orient_q, joints_world)
            if trans is not None:
                joints_world = joints_world + trans.unsqueeze(-2)
            self.smpl_joints = joints_world

        if self.correct_root_height:
            diff_root_height = torch.min(self.smpl_joints[:, :, 10:12, 2], dim=2)[
                0
            ].view(*trans.shape[:2], 1)
            self.smpl_joints[:, :, :, 2] -= diff_root_height
            if self.smpl_verts is not None:
                self.smpl_verts[:, :, :, 2] -= diff_root_height
            trans[:, :, 2:] -= diff_root_height

        if racket_seq is not None:
            num_actors, num_frames = trans.shape[:2]
            for i in range(num_actors):
                for j in range(num_frames):
                    if racket_seq[i][j] is not None:
                        racket_seq[i][j]["root"] = trans[i, j].numpy()
            self.racket_params = racket_seq

        self.ball_params = ball_seq

        self.fr = 0
        self.num_fr = self.smpl_joints.shape[1]

    def init_camera(self, init_args):
        self.camera = init_args.get("camera", self.camera)
        self.sport = init_args.get("sport", self.sport)

        if self.sport == "tennis":
            if self.camera == "front":
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = (
                    [0, 0, 0] if self.enable_shadow else [0, -0.66, -1.78]
                )
                self.pl.camera.position = (
                    [0, -30, 5] if self.enable_shadow else [0, -25.5, 8.4]
                )
            elif self.camera == "front_right":
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = (
                    [2, 0, 0] if self.enable_shadow else [0, -0.66, -1.78]
                )
                self.pl.camera.position = (
                    [2, -30, 5] if self.enable_shadow else [0, -25.5, 8.4]
                )
            elif self.camera == "front_left":
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = (
                    [-2, 0, 0] if self.enable_shadow else [0, -0.66, -1.78]
                )
                self.pl.camera.position = (
                    [-2, -30, 5] if self.enable_shadow else [0, -25.5, 8.4]
                )
            elif self.camera == "back":
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, 0, 0]
                self.pl.camera.position = [0, 25, 5]
            elif self.camera == "top_both":
                self.pl.camera.up = (-1, 0, 0)
                self.pl.camera.focal_point = [0, 0, 0]
                self.pl.camera.position = [0, 0, 35]
            elif self.camera == "top_near":
                self.pl.camera.up = (-1, 0, 0)
                self.pl.camera.focal_point = [0, -12, 0]
                self.pl.camera.position = [0, -12, 20]
            elif self.camera == "top_far":
                self.pl.camera.up = (-1, 0, 0)
                self.pl.camera.focal_point = [0, 12, 0]
                self.pl.camera.position = [0, 12, 20]
            elif self.camera == "side_both":
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, 0, 0]
                self.pl.camera.position = [35, 0, 3]
            elif self.camera == "near_left":
                self.pl.camera.elevation = 0
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, -13, 0]
                self.pl.camera.position = [-12, -13, 3]
            elif self.camera == "near_right":
                self.pl.camera.elevation = 0
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, -13, 0]
                self.pl.camera.position = [12, -13, 3]
        elif self.sport == "badminton":
            if self.camera == "front":
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, 0, 0]
                self.pl.camera.position = [0, -13, 3]
            elif self.camera == "side_both":
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, 0, 0]
                self.pl.camera.position = [15, 0, 3]
            elif self.camera == "side_near":
                self.pl.camera.elevation = 0
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, -3.5, 0]
                self.pl.camera.position = [15, -3.5, 0]
            elif self.camera == "side_far":
                self.pl.camera.elevation = 0
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, 3.5, 0]
                self.pl.camera.position = [15, 3.5, 0]

    def init_scene(self, init_args):
        if init_args is None:
            init_args = dict()
        super().init_scene(init_args)

        # Init tennis court
        if init_args.get("sport") == "tennis" and not init_args.get("no_court", False):
            # Court
            wlh = (10.97, 11.89 * 2, 0.05)
            center = np.array([0, 0, -wlh[2] * 0.5])
            court_mesh = pyvista.Cube(center, *wlh)
            self.pl.add_mesh(
                court_mesh,
                color="#4A609D",
                ambient=0.2,
                diffuse=0.8,
                specular=0.2,
                specular_power=5,
                smooth_shading=True,
            )

            # Court lines (vertical)
            for x, l in zip(
                [-10.97 / 2, -8.23 / 2, 0, 8.23 / 2, 10.97 / 2],
                [23.77, 23.77, 12.8, 23.77, 23.77],
            ):
                wlh = (0.05, l, 0.05)
                center = np.array([x, 0, -wlh[2] * 0.5])
                court_line_mesh = pyvista.Cube(center, *wlh)
                court_line_mesh.points[:, 2] += 0.01
                self.pl.add_mesh(court_line_mesh, color="#FFFFFF", smooth_shading=True)

            # Court lines (horizontal)
            for y, w in zip(
                [-11.89, -6.4, 0, 6.4, 11.89], [10.97, 8.23, 10.97, 8.23, 10.97]
            ):
                wlh = (w, 0.05, 0.05)
                center = np.array([0, y, -wlh[2] * 0.5])
                court_line_mesh = pyvista.Cube(center, *wlh)
                court_line_mesh.points[:, 2] += 0.01
                self.pl.add_mesh(court_line_mesh, color="#FFFFFF", smooth_shading=True)

            # Post
            for x in [-0.91 - 10.97 / 2, 0.91 + 10.97 / 2]:
                wlh = (0.05, 0.05, 1.2)
                center = np.array([x, 0, wlh[2] * 0.5])
                post_mesh = pyvista.Cube(center, *wlh)
                self.pl.add_mesh(
                    post_mesh,
                    color="#BD7427",
                    ambient=0.2,
                    diffuse=0.8,
                    specular=0.8,
                    specular_power=5,
                    smooth_shading=True,
                )

            # Net
            wlh = (10.97 + 0.91 * 2, 0.01, 1.07)
            center = np.array([0, 0, 1.07 / 2])
            net_mesh = pyvista.Cube(center, *wlh)
            if not self.enable_shadow:
                net_mesh.active_t_coords *= 1000
                tex = pyvista.numpy_to_texture(
                    make_checker_board_texture("#FFFFFF", "#AAAAAA", width=10)
                )
                self.pl.add_mesh(
                    net_mesh,
                    texture=tex,
                    ambient=0.2,
                    diffuse=0.8,
                    opacity=0.1,
                    smooth_shading=True,
                )

            # Lighting
            if self.enable_shadow:
                for x, y in [(-5, -5), (5, -5), (-5, 5), (5, 5)]:
                    light = pyvista.Light(
                        position=(x, y, 10),
                        focal_point=(0, 0, 0),
                        color=[1.0, 1.0, 1.0, 1.0],  # Color temp. 5400 K
                        intensity=0.4,
                    )
                    self.pl.add_light(light)

        elif init_args.get("sport") == "badminton" and not init_args.get(
            "no_court", False
        ):
            # Court
            wlh = (6.1, 13.41, 0.05)
            center = np.array([0, 0, -wlh[2] * 0.5])
            court_mesh = pyvista.Cube(center, *wlh)
            self.pl.add_mesh(
                court_mesh,
                color="#4A609D",
                ambient=0.2,
                diffuse=0.8,
                specular=0.2,
                specular_power=5,
                smooth_shading=True,
            )

            # Court lines (vertical)
            for x in [-3.05, -2.6, 2.6, 3.05]:
                wlh = (0.05, 13.41, 0.05)
                center = np.array([x, 0, -wlh[2] * 0.5])
                court_line_mesh = pyvista.Cube(center, *wlh)
                court_line_mesh.points[:, 2] += 0.01
                self.pl.add_mesh(court_line_mesh, color="#FFFFFF", smooth_shading=True)
            for x, y, l in zip(
                [0, 0],
                [-(1.98 + 6.71) / 2, (1.98 + 6.71) / 2],
                [3.96 + 0.76, 3.96 + 0.76],
            ):
                wlh = (0.05, l, 0.05)
                center = np.array([x, y, -wlh[2] * 0.5])
                court_line_mesh = pyvista.Cube(center, *wlh)
                court_line_mesh.points[:, 2] += 0.01
                self.pl.add_mesh(court_line_mesh, color="#FFFFFF", smooth_shading=True)

            # Court lines (horizontal)
            for y in [-6.71, -3.96 - 1.98, -1.98, 1.98, 1.98 + 3.96, 6.71]:
                wlh = (6.1, 0.05, 0.05)
                center = np.array([0, y, -wlh[2] * 0.5])
                court_line_mesh = pyvista.Cube(center, *wlh)
                court_line_mesh.points[:, 2] += 0.01
                self.pl.add_mesh(court_line_mesh, color="#FFFFFF", smooth_shading=True)

            # Post
            for x in [-3.05, 3.05]:
                wlh = (0.05, 0.05, 1.55)
                center = np.array([x, 0, wlh[2] * 0.5])
                post_mesh = pyvista.Cube(center, *wlh)
                self.pl.add_mesh(
                    post_mesh,
                    color="#BD7427",
                    ambient=0.2,
                    diffuse=0.8,
                    specular=0.8,
                    specular_power=5,
                    smooth_shading=True,
                )

            # Net
            wlh = (6.1, 0.01, 0.79)
            center = np.array([0, 0, (0.76 + 1.55) / 2])
            net_mesh = pyvista.Cube(center, *wlh)
            net_mesh.active_t_coords *= 1000
            tex = pyvista.numpy_to_texture(
                make_checker_board_texture("#FFFFFF", "#AAAAAA", width=10)
            )
            self.pl.add_mesh(
                net_mesh,
                texture=tex,
                ambient=0.2,
                diffuse=0.8,
                opacity=0.1,
                smooth_shading=True,
            )

        if not init_args.get("no_court", False):
            # floor
            wlh = (100, 100, 0.05)
            center = np.array([0, 0, -wlh[2] * 0.5])
            floor_mesh = pyvista.Cube(center, *wlh)
            floor_mesh.points[:, 2] -= 0.01
            self.pl.add_mesh(
                floor_mesh,
                color="#769771",
                ambient=0.2,
                diffuse=0.8,
                specular=0.2,
                specular_power=5,
                smooth_shading=True,
            )
        else:
            wlh = (20.0, 40.0, 0.05)
            center = np.array([0, 0, -wlh[2] * 0.5])
            self.floor_mesh = pyvista.Cube(center, *wlh)
            self.floor_mesh.t_coords *= 10 / self.floor_mesh.t_coords.max()
            tex = pyvista.numpy_to_texture(
                make_checker_board_texture("#81C6EB", "#D4F1F7")
            )
            self.pl.add_mesh(
                self.floor_mesh,
                texture=tex,
                ambient=0.2,
                diffuse=0.8,
                specular=0.8,
                specular_power=5,
                smooth_shading=True,
            )

        smpl_seq, racket_seq, ball_seq = (
            init_args.get("smpl_seq"),
            init_args.get("racket_seq"),
            init_args.get("ball_seq"),
        )
        if smpl_seq is not None:
            self.setup_animation(smpl_seq, racket_seq, ball_seq)
        elif ball_seq is not None:
            self.ball_params = ball_seq
            self.fr = 0
            self.num_fr = len(ball_seq[0])
        self.num_actors = init_args["num_actors"]

        if self.show_smpl:
            if init_args.get("vis_mvae") and init_args.get("vis_pd_target"):
                colors_smpl = (
                    ["#ffca3a"] * (self.num_actors // 3)
                    + ["#9d0208"] * (self.num_actors // 3)
                    + ["green"] * (self.num_actors // 3)
                )
            elif init_args.get("vis_mvae") or init_args.get("vis_pd_target"):
                colors_smpl = ["#ffca3a"] * (self.num_actors // 2) + ["#9d0208"] * (
                    self.num_actors // 2
                )
            elif self.num_actors <= 2:
                colors_smpl = ["#ffca3a"] * self.num_actors
            else:
                colors_smpl = get_color_palette(self.num_actors, "Wistia")
                # colors_smpl = ['#ffca3a'] * self.num_actors
            # HACK: get vertices from fake smpl joint
            smpl_motion = self.smpl(
                global_orient=torch.zeros((1, 3)).float(),
                body_pose=torch.zeros((1, 69)).float(),
                betas=torch.zeros((1, 10)).float(),
                root_trans=torch.zeros((1, 3)).float(),
                return_full_pose=True,
                orig_joints=True,
            )
            vertices = smpl_motion.vertices.reshape(-1, 3).numpy()
            if init_args.get("debug_root"):
                # Odd actors are final result, even actors are old result
                self.smpl_actors = [
                    SMPLActor(
                        self.pl,
                        vertices,
                        self.smpl_faces,
                        color="#d00000" if i % 2 == 0 else "#ffca3a",
                    )
                    for i in range(self.num_actors)
                ]
            else:
                self.smpl_actors = [
                    SMPLActor(self.pl, vertices, self.smpl_faces, color=colors_smpl[a])
                    for a in range(self.num_actors)
                ]

        if self.show_skeleton:
            if not self.show_smpl:
                colors_skeleton = get_color_palette(self.num_actors, colormap="autumn")
            else:
                colors_skeleton = ["yellow"] * self.num_actors
            self.skeleton_actors = [
                SkeletonActor(
                    self.pl, self.smpl_joint_parents, bone_color=colors_skeleton[a]
                )
                for a in range(self.num_actors)
            ]

        if self.show_racket:
            self.racket_actors = [
                RacketActor(self.pl, init_args.get("sport"), debug=False)
                for _ in range(self.num_actors)
            ]

        if self.show_target:
            self.tar_reaction_actors = [
                TargetReactionActor(self.pl) for _ in range(self.num_actors)
            ]
            self.tar_recover_actors = [
                TargetRecoveryActor(self.pl) for _ in range(self.num_actors)
            ]

        if self.show_ball:
            if init_args.get("debug_ball"):
                if init_args.get("debug_ball") == "comparison":
                    self.ball_actors = [
                        BallActor(
                            self.pl,
                            init_args.get("sport"),
                            color="yellow" if a < self.num_actors // 2 else "red",
                            real_shadow=self.enable_shadow,
                        )
                        for a in range(self.num_actors)
                    ]
                else:
                    colors = get_color_palette(self.num_actors, "rainbow")
                    self.ball_actors = [
                        BallActor(
                            self.pl,
                            init_args.get("sport"),
                            color=colors[a],
                            real_shadow=self.enable_shadow,
                        )
                        for a in range(self.num_actors)
                    ]
            elif init_args.get("add_second_ball"):
                self.ball_actors = [
                    BallActor(
                        self.pl,
                        init_args.get("sport"),
                        blur=init_args.get("ball_blur"),
                        real_shadow=self.enable_shadow,
                    )
                    for _ in range(self.num_actors * 2)
                ]
            elif (
                self.num_actors <= 2
                or init_args.get("vis_mvae")
                or init_args.get("vis_pd_target")
            ):
                self.ball_actors = [
                    BallActor(
                        self.pl,
                        init_args.get("sport"),
                        blur=init_args.get("ball_blur"),
                        real_shadow=self.enable_shadow,
                    )
                    for _ in range(self.num_actors)
                ]
            else:
                self.ball_actors = [
                    BallActor(
                        self.pl,
                        init_args.get("sport"),
                        blur=init_args.get("ball_blur"),
                        color=colors_smpl[a],
                        real_shadow=self.enable_shadow,
                    )
                    for a in range(self.num_actors)
                ]

        if self.show_ball_target:
            self.ball_tar_actors = [
                TargetBounceActor(self.pl) for _ in range(self.num_actors)
            ]

        if self.show_stats:
            self.text_actor_tar = self.pl.add_text(
                "", position=(30, 1050), color="black", font_size=12
            )
            self.text_actor_reward = self.pl.add_text(
                "", position=(30, 1020), color="black", font_size=12
            )
            self.text_actor_residual = self.pl.add_text(
                "", position=(30, 990), color="black", font_size=12
            )
            self.text_actor_pose = self.pl.add_text(
                "", position=(30, 960), color="black", font_size=12
            )
            self.text_actor_pose_tar = self.pl.add_text(
                "", position=(30, 930), color="black", font_size=12
            )
            self.text_actor_racket = self.pl.add_text(
                "", position=(30, 900), color="black", font_size=12
            )
            self.text_actor_ball = self.pl.add_text(
                "", position=(30, 870), color="black", font_size=12
            )
            self.text_actor_contact = self.pl.add_text(
                "", position=(30, 840), color="black", font_size=12
            )

    def update_camera(self, interactive):
        if self.track_first_actor:
            root_pos = self.smpl_joints[0, self.fr, 0].cpu().numpy()
            if self.camera == "front":
                new_pos = (
                    [root_pos[0], -30, 5]
                    if self.enable_shadow
                    else [root_pos[0], -25, 5]
                )
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, 0, 0]
                self.pl.camera.position = new_pos

            elif self.camera == "near_left":
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [root_pos[0], root_pos[1], 1]
                self.pl.camera.position = [root_pos[0] - 5, root_pos[1], 1]

            elif self.camera == "near_right":
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [root_pos[0], root_pos[1], 1]
                self.pl.camera.position = [root_pos[0] + 5, root_pos[1], 1]

        if self.track_ball:
            self.pl.camera.up = (0, 0, 1)
            self.pl.camera.focal_point = self.ball_params[0][self.fr].cpu().numpy()
            self.pl.camera.position = self.pl.camera.focal_point + np.array([3, 0, 0])

    def update_scene(self):
        super().update_scene()

        if self.show_smpl and self.smpl_verts is not None:
            for i, actor in enumerate(self.smpl_actors):
                if self.smpl_joints[i, self.fr].sum() == 0:
                    actor.set_visibility(False)
                else:
                    actor.update_verts(self.smpl_verts[i, self.fr].cpu().numpy())
                    actor.set_visibility(True)
                    if self.enable_shadow:
                        actor.set_opacity(1.0)
                    elif self.show_skeleton:
                        actor.set_opacity(0.8)
                    else:
                        actor.set_opacity(1.0)

        if self.show_skeleton and self.smpl_joints is not None:
            for i, actor in enumerate(self.skeleton_actors):
                if self.smpl_joints[i, self.fr].sum() == 0:
                    actor.set_visibility(False)
                else:
                    actor.update_joints(self.smpl_joints[i, self.fr].cpu().numpy())
                    if self.enable_shadow:
                        actor.set_visibility(False)
                    else:
                        actor.set_visibility(True)
                        actor.set_opacity(0.2)

        if self.show_racket and self.racket_params is not None:
            for i, actor in enumerate(self.racket_actors):
                if self.smpl_joints[i, self.fr].sum() == 0:
                    actor.set_visibility(False)
                else:
                    actor.update_racket(self.racket_params[i][self.fr])
                    actor.set_visibility(True)

        if self.show_ball and self.ball_params is not None:
            for i, actor in enumerate(self.ball_actors):
                if i >= len(self.ball_params):
                    actor.set_visibility(False)
                else:
                    actor.update_ball(self.ball_params[i][self.fr])

    def setup_key_callback(self):
        super().setup_key_callback()

        def track_first_actor():
            self.track_first_actor = not self.track_first_actor

        def track_ball():
            self.track_ball = not self.track_ball
            if not self.track_ball:
                self.init_camera({"camera": "front"})

        def forward():
            self.forward = True

        def reset_camera_front():
            self.init_camera({"camera": "front"})

        def reset_camera_back():
            self.init_camera({"camera": "back"})

        def reset_camera_side_both():
            self.init_camera({"camera": "side_both"})

        def reset_camera_top_both():
            self.init_camera({"camera": "top_both"})

        def reset_camera_top_near():
            self.init_camera({"camera": "top_near"})

        def reset_camera_top_far():
            self.init_camera({"camera": "top_far"})

        def reset_camera_near_left():
            self.init_camera({"camera": "near_left"})

        def reset_camera_near_right():
            self.init_camera({"camera": "near_right"})

        self.pl.add_key_event("t", track_first_actor)
        self.pl.add_key_event("b", track_ball)
        self.pl.add_key_event("n", forward)
        self.pl.add_key_event("1", reset_camera_front)
        self.pl.add_key_event("2", reset_camera_back)
        self.pl.add_key_event("3", reset_camera_side_both)
        self.pl.add_key_event("4", reset_camera_near_left)
        self.pl.add_key_event("5", reset_camera_near_right)
        self.pl.add_key_event("6", reset_camera_top_both)
        self.pl.add_key_event("7", reset_camera_top_near)
        self.pl.add_key_event("8", reset_camera_top_far)

    def show_animation_online(
        self,
        window_size=(800, 800),
        init_args=None,
        enable_shadow=None,
        show_axes=True,
        off_screen=False,
        fps=30,
    ):
        self.fps = fps
        self.frame_mode = "fps"
        if off_screen:
            if platform.system() == "Linux":
                pyvista.start_xvfb()
        if enable_shadow is not None:
            self.enable_shadow = enable_shadow
        if self.enable_shadow:
            window_size = (1000, 1000)
            self.pl = pyvista.Plotter(
                window_size=window_size, off_screen=off_screen, lighting="none"
            )
        else:
            self.pl = pyvista.Plotter(window_size=window_size, off_screen=off_screen)
        self.init_camera(init_args)
        self.init_scene(init_args)
        self.setup_key_callback()
        if show_axes:
            self.pl.show_axes()
        self.pl.show(interactive_update=True)

    def update_scene_online(
        self,
        joint_pos=None,
        smpl_verts=None,
        racket_params=None,
        ball_params=None,
        ball_targets=None,
        tar_pos=None,
        tar_action=None,
        tar_time=None,
        stats=None,
    ):

        if self.show_smpl and smpl_verts is not None:
            for i, actor in enumerate(self.smpl_actors):
                actor.update_verts(smpl_verts[i].cpu().numpy())
                actor.set_visibility(True)
                if self.show_skeleton:
                    actor.set_opacity(0.8 if not self.enable_shadow else 1.0)
                else:
                    actor.set_opacity(1.0)

        if self.show_skeleton and joint_pos is not None:
            for i, actor in enumerate(self.skeleton_actors):
                actor.update_joints(joint_pos[i].cpu().numpy())
                if not self.enable_shadow:
                    actor.set_visibility(True)
                    actor.set_opacity(1.0)
                else:
                    actor.set_visibility(False)

        if self.show_racket and racket_params is not None:
            for i, actor in enumerate(self.racket_actors):
                actor.update_racket(racket_params[i])
                actor.set_visibility(True)

        if self.show_target and tar_pos is not None:
            for i in range(self.num_actors):
                rea_actor = self.tar_reaction_actors[i]
                rec_actor = self.tar_recover_actors[i]
                if tar_action is None:
                    rec_actor.update_target(tar_pos[i].cpu().numpy())
                    rec_actor.set_visibility(True)
                    rea_actor.set_visibility(False)
                else:
                    rec_actor.set_visibility(False)
                    if tar_action[i] == 1:
                        rea_actor.update_target(
                            tar_pos[i].cpu().numpy(), tar_time[i].cpu().numpy()
                        )
                        rea_actor.set_visibility(True)

        if self.show_ball and ball_params is not None:
            for i in range(min(self.num_actors, len(ball_params))):
                self.ball_actors[i].update_ball(ball_params[i])

        if self.show_ball_target and ball_targets is not None:
            for i in range(min(self.num_actors, len(ball_targets))):
                self.ball_tar_actors[i].update_target(ball_targets[i])

        if self.show_stats and stats is not None:
            self.text_actor_tar.SetInput(
                "Target time, action, phase, swing, recovery, atnet: {:02d}, {}, {:.2f}, {}, {}, {}".format(
                    stats["tar_time"].cpu().numpy(),
                    stats["tar_action"].cpu().numpy(),
                    stats["phase"].cpu().numpy(),
                    stats["swing_type"].cpu().numpy(),
                    stats["target_recovery"].cpu().numpy(),
                    stats["at_net"].cpu().numpy(),
                )
            )
            if stats["sub_reward_names"] is not None:
                self.text_actor_reward.SetInput(
                    "Reward: ({}) - {}".format(
                        stats["sub_reward_names"].replace("_reward", ""),
                        np.array2string(
                            stats["sub_rewards"].cpu().numpy(),
                            formatter={"all": lambda x: "{:>7.4f}".format(x)},
                            separator=",",
                        ),
                    )
                )
            if (
                stats.get("res_dof_actions") is not None
                and stats.get("mvae_actions_norm") is not None
            ):
                self.text_actor_residual.SetInput(
                    "VAE action norm, residual: {} {}".format(
                        np.array2string(
                            stats["mvae_actions_norm"].cpu().numpy(),
                            formatter={"all": lambda x: "{:>5.1f}".format(x)},
                            separator=",",
                        ),
                        np.array2string(
                            stats["res_dof_actions"].cpu().numpy(),
                            formatter={"all": lambda x: "{:>5.1f}".format(x * 180)},
                            separator=",",
                        ),
                    )
                )
            if stats.get("wrist_angle") is not None:
                self.text_actor_pose.SetInput(
                    "Physics Wrist, elbow, shoulder: {} {} {}".format(
                        # np.array2string(stats['wrist_angle_glb'].cpu().numpy(), formatter={'all': lambda x: '{:04.1f}'.format(x * 180 / pi)}, separator=','),
                        np.array2string(
                            stats["wrist_angle"].cpu().numpy(),
                            formatter={
                                "all": lambda x: "{:>5.1f}".format(x * 180 / pi)
                            },
                            separator=",",
                        ),
                        np.array2string(
                            stats["elbow_angle"].cpu().numpy(),
                            formatter={
                                "all": lambda x: "{:>5.1f}".format(x * 180 / pi)
                            },
                            separator=",",
                        ),
                        np.array2string(
                            stats["shoulder_angle"].cpu().numpy(),
                            formatter={
                                "all": lambda x: "{:>5.1f}".format(x * 180 / pi)
                            },
                            separator=",",
                        ),
                    )
                )
            if stats.get("wrist_angle_tar") is not None:
                self.text_actor_pose_tar.SetInput(
                    "Target Wrist, elbow, shoulder: {} {} {}".format(
                        # np.array2string(stats['wrist_angle_glb'].cpu().numpy(), formatter={'all': lambda x: '{:04.1f}'.format(x * 180 / pi)}, separator=','),
                        np.array2string(
                            stats["wrist_angle_tar"].cpu().numpy(),
                            formatter={
                                "all": lambda x: "{:>5.1f}".format(x * 180 / pi)
                            },
                            separator=",",
                        ),
                        np.array2string(
                            stats["elbow_angle_tar"].cpu().numpy(),
                            formatter={
                                "all": lambda x: "{:>5.1f}".format(x * 180 / pi)
                            },
                            separator=",",
                        ),
                        np.array2string(
                            stats["shoulder_angle_tar"].cpu().numpy(),
                            formatter={
                                "all": lambda x: "{:>5.1f}".format(x * 180 / pi)
                            },
                            separator=",",
                        ),
                    )
                )
            if stats.get("racket_pos") is not None:
                self.text_actor_racket.SetInput(
                    "Racket pos, vel, norm: {} {} {}".format(
                        np.array2string(
                            stats["racket_pos"].cpu().numpy(),
                            formatter={"all": lambda x: "{:>6.2f}".format(x)},
                            separator=",",
                        ),
                        np.array2string(
                            stats["racket_vel"].cpu().numpy(),
                            formatter={"all": lambda x: "{:>6.2f}".format(x)},
                            separator=",",
                        ),
                        np.array2string(
                            stats["racket_normal"].cpu().numpy(),
                            formatter={"all": lambda x: "{:>6.2f}".format(x)},
                            separator=",",
                        ),
                    )
                )
            if stats.get("ball_pos") is not None:
                self.text_actor_ball.SetInput(
                    "Ball pos, vel, ang_vel, vspin, bounce, target: {} {} {} {} {} {} {}".format(
                        np.array2string(
                            stats["ball_pos"].cpu().numpy(),
                            formatter={"all": lambda x: "{:>6.2f}".format(x)},
                            separator=",",
                        ),
                        np.array2string(
                            stats["ball_vel"].cpu().numpy(),
                            formatter={"all": lambda x: "{:>6.2f}".format(x)},
                            separator=",",
                        ),
                        np.array2string(
                            stats["ball_ang_vel"].cpu().numpy(),
                            formatter={"all": lambda x: "{:>6.2f}".format(x)},
                            separator=",",
                        ),
                        np.array2string(
                            stats["ball_vspin"].cpu().numpy(),
                            formatter={"all": lambda x: "{:>6.2f}".format(x)},
                            separator=",",
                        ),
                        np.array2string(
                            stats["est_ball_bounce"].cpu().numpy(),
                            formatter={"all": lambda x: "{:>6.2f}".format(x)},
                            separator=",",
                        ),
                        np.array2string(
                            stats["ball_target_pos"].cpu().numpy(),
                            formatter={"all": lambda x: "{:>6.2f}".format(x)},
                            separator=",",
                        ),
                        np.array2string(
                            stats["ball_target_spin"].cpu().numpy(),
                            formatter={"all": lambda x: "{}".format(x)},
                            separator=",",
                        ),
                    )
                )
            if stats.get("contact_force") is not None:
                self.text_actor_contact.SetInput(
                    "Contact force racket, ball: {} {}".format(
                        np.array2string(
                            stats["contact_force"][-2].cpu().numpy(),
                            formatter={"all": lambda x: "{:>5.1f}".format(x)},
                            separator=",",
                        ),
                        np.array2string(
                            stats["contact_force"][-1].cpu().numpy(),
                            formatter={"all": lambda x: "{:>5.1f}".format(x)},
                            separator=",",
                        ),
                    )
                )

        self.smpl_joints = joint_pos.unsqueeze(0)
        if ball_params[0] is not None:
            self.ball_params = [[ball_params[0]["pos"]]]
        self.fr = 0

    def render_online(self, interactive):
        last_render_time = time.time()
        if interactive:
            while True:
                self.render(interactive=True)
                if self.forward:
                    self.paused = True
                    self.forward = False
                    break
                if self.paused:
                    continue
                if time.time() - last_render_time >= (1 / self.fps - 0.002):
                    break
        else:
            self.render(interactive=False)
