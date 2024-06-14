# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import shutil
import time
import numpy as np
import os
from enum import Enum
from uuid import uuid4
from tqdm import tqdm
import glob
from scipy.stats import norm
import joblib
import numpy as np
import yaml

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples

import torch

from uhc.smpllib.smpl_local_robot import Robot
from embodied_pose.utils.torch_transform import heading_to_vec, angle_axis_to_rot6d
from embodied_pose.utils import torch_utils
from embodied_pose.utils.torch_transform import ypr_euler_from_quat, angle_axis_to_quaternion
from embodied_pose.utils.mjviewer import MjViewer
from embodied_pose.utils import torch_utils

from mujoco_py import load_model_from_path, MjSim
import glfw
import imageio



import torch
import sys
import os
import os.path as osp
import operator
import imageio
import operator
from datetime import datetime
from copy import deepcopy
import random
from lxml.etree import XMLParser, parse, Element
from copy import deepcopy

from uhc.smpllib.smpl_local_robot import Robot
from scipy.spatial.transform import Rotation as sRot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from uhc.smpllib.smpl_mujoco import SMPL_BONE_ORDER_NAMES as joint_names

TMP_SMPL_DIR = f"/tmp/smpl_humanoid_{uuid4()}"
ENABLE_MAX_COORD_OBS = True

def get_attr_val_from_sample(sample, offset, prop, attr):
    """Retrieves param value for the given prop and attr from the sample."""
    if sample is None:
        return None, 0
    if isinstance(prop, np.ndarray):
        smpl = sample[offset:offset+prop[attr].shape[0]]
        return smpl, offset+prop[attr].shape[0]
    else:
        return sample[offset], offset+1


class HumanoidSMPL():
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
         
        # super().__init__(cfg=self.cfg)
        ####
        cfg=self.cfg
        enable_camera_sensors=False
        self.gym = gymapi.acquire_gym()

        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)

        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        self.headless = cfg["headless"]
        self.headless = True
        # show_mj_viewer = not headless
        # headless = headless or not show_gym_viewer


        # double check!
        self.graphics_device_id = self.device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        # create envs, sim and viewer
        self.num_envs = cfg["env"]["numEnvs"]

        self.create_sim()
        self.gym.prepare_sim(self.sim)

        self.num_obs = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]

        self.control_freq_inv = cfg["env"].get("controlFrequencyInv", 1)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        self.original_props = {}
        self.dr_randomizations = {}
        self.first_randomization = True
        self.actor_params_generator = None
        self.extern_actor_params = {}
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        self.last_step = -1
        self.last_rand_step = -1

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "toggle_video_record")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        ###### Camera Sensors ######
        camera_props = gymapi.CameraProperties()
        camera_props.width = 2560
        camera_props.height = 1440
        camera_props.horizontal_fov = 90.0
        self.viewer_camera_handle = self.gym.create_camera_sensor(self.envs[0], camera_props);
        rendering_out = osp.join("out",  "recording")
        os.makedirs(rendering_out, exist_ok=True)
        self._video_path = osp.join(rendering_out, "video_%s.mp4")
        ####

        self.dt = self.control_freq_inv * sim_params.dt
        
        self._setup_tensors()
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        self._build_termination_heights()
        
        key_bodies = self.cfg["env"]["keyBodies"]
        contact_bodies = self.cfg["env"]["contactBodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)
        
        if self.viewer != None:
            # <----- init camera
            # self._init_camera()
            # def _init_camera(self):
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
            cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0)
            cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],self._cam_prev_char_pos[1],1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            # return
            # ------>


        if self.cfg['show_mujoco_viewer']:
            num_vis_humanoids = 3
            model_file = self.humanoid_files[0]
            print(f"Loading model file: {model_file}")
            vis_model_file = model_file[:-4] + '_vis.xml'
            self.num_vis_capsules = 100
            self.num_vis_spheres = 100
            
            ############
            def create_vis_model_xml(in_file, out_file, num_actor=2, num_vis_capsules=0, num_vis_spheres=0, num_vis_planes=0):
                parser = XMLParser(remove_blank_text=True)
                tree = parse(in_file, parser=parser)
                geom_capsule = Element('geom', attrib={'fromto': '0 0 -10000 0 0 -9999', 'size': '0.02', 'type': 'capsule'})
                geom_sphere = Element('geom', attrib={'pos': '0 0 -10000', 'size': '0.02', 'type': 'sphere'})
                geom_plane = Element('geom', attrib={'pos': '0 0 -10000', 'size': '0.15 0.15 0.005', 'type': 'box'})

                root = tree.getroot().find('worldbody')
                body = root.find('body')
                for body_node in body.findall('.//body'):
                    for joint_node in body_node.findall('joint')[1:]:
                        body_node.remove(joint_node)
                        body_node.insert(0, joint_node)

                for i in range(1, num_actor):
                    new_body = deepcopy(body)
                    new_body.attrib['childclass'] = f'actor{i}'
                    new_body.attrib['name'] = '%d_%s' % (i, new_body.attrib['name'])
                    for node in new_body.findall(".//body"):
                        node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
                    for node in new_body.findall(".//joint"):
                        node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
                    for node in new_body.findall(".//freejoint"):
                        node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
                    root.append(new_body)
                act_node = tree.find('actuator')
                act_node.getparent().remove(act_node)

                ind = 2
                for i in range(num_vis_capsules):
                    root.insert(ind, deepcopy(geom_capsule))
                    ind += 1
                for i in range(num_vis_spheres):
                    root.insert(ind, deepcopy(geom_sphere))
                    ind += 1
                for i in range(num_vis_planes):
                    root.insert(ind, deepcopy(geom_plane))
                    ind += 1
                
                tree.write(out_file, pretty_print=True)
            create_vis_model_xml(model_file, vis_model_file, num_vis_humanoids, 
                                 num_vis_capsules=self.num_vis_capsules, 
                                 num_vis_spheres=self.num_vis_spheres)
            #############

            self.mj_model = load_model_from_path(vis_model_file)
            self.mj_sim = MjSim(self.mj_model)
            self.mj_viewer = MjViewer(self.mj_sim)
            self.mj_data = self.mj_sim.data
            self.nq = self.mj_model.nq // num_vis_humanoids
            self.mj_viewer.render()
            self.mj_viewer.custom_key_callback = self.key_callback
            self.mj_viewer._hide_overlay = True
            glfw.restore_window(self.mj_viewer.window)
            glfw.set_window_size(self.mj_viewer.window, 1620, 1080)
        else:
            self.mj_viewer = None

        self.headless = headless
        self.cam_inited = False
        # self.max_episode_length = 3000
        self.vis_mode = self.args.vis_mode

        self.recording = False
        self.recording_npz = False
        self.cfg['env']['num_rec_frames'] = self._motion_lib._motion_num_frames.max()
        print(f"Recording only {self.cfg['env']['num_rec_frames']} frames")

        if self.cfg['env'].get('record', False):
            self.recording = True
            self.start_recording()

        if self.cfg['env'].get('record_npz', False):
            self.recording_npz = True
            self.start_recording_npz()
            self.record_npz_tqdm_bar = tqdm(range(self.cfg['env']['num_rec_frames']), desc = 'recording npz...')

        mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
        self.mujoco_2_smpl = [mujoco_joint_names.index(q) for q in joint_names if q in mujoco_joint_names]

            
        return

    def _setup_tensors(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()
        
        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0

        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]
        
        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]

    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):

        # set gravity based on up axis and return axis index
        def set_sim_params_up_axis(sim_params, axis):
            if axis == 'z':
                sim_params.up_axis = gymapi.UP_AXIS_Z
                sim_params.gravity.x = 0
                sim_params.gravity.y = 0
                sim_params.gravity.z = -9.81
                return 2
            return 1

        self.up_axis_idx = set_sim_params_up_axis(self.sim_params, 'z')

        if hasattr(self, 'sim'):
            self.gym.destroy_sim(self.sim)
            del self.sim
        self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        return

    def reset(self, env_ids=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
        return

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
            self._dof_obs_size = 72
            self._num_actions = 28

            if (ENABLE_MAX_COORD_OBS):
                self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
            else:
                self._num_obs = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
            
        elif (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 16, 17, 20, 21, 24, 27, 28, 31]
            self._dof_obs_size = 78
            self._num_actions = 31
            
            if (ENABLE_MAX_COORD_OBS):
                self._num_obs = 1 + 17 * (3 + 6 + 3 + 3) - 3
            else:
                self._num_obs = 13 + self._dof_obs_size + 31 + 3 * num_key_bodies
            
        elif (asset_file == "mjcf/ov_humanoid.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17]
            self._dof_offsets = [0, 3, 6, 9, 12, 15, 16, 19, 22, 23, 26, 27, 30, 33, 34, 37]
            self._dof_obs_size = 90
            self._num_obs = 13 + self._dof_obs_size + 37 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
            self._num_actions = 37
        elif (asset_file == "mjcf/ov_humanoid_sword_shield.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 15, 16, 17, 18, 19]
            self._dof_offsets = [0, 3, 6, 9, 12, 15, 16, 19, 22, 25, 26, 29, 30, 33, 36, 37, 40]
            self._dof_obs_size = 96
            self._num_obs = 13 + self._dof_obs_size + 40 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
            self._num_actions = 40
        else:   # TODO
            self.body_names = body_names = self.gym.get_asset_rigid_body_names(self.humanoid_asset)
            self.dof_names = dof_names = self.gym.get_asset_dof_names(self.humanoid_asset)
            self._dof_body_ids = []
            self._dof_offsets = []
            self._num_actions = self._num_dof = len(dof_names)
            cur_dof_index = 0
            dof_body_names = [x[:-2] for x in dof_names]
            for i, body in enumerate(body_names):
                if body != dof_body_names[cur_dof_index]:
                    continue
                self._dof_body_ids.append(i)
                self._dof_offsets.append(cur_dof_index)
                while body == dof_body_names[cur_dof_index]:
                    cur_dof_index += 1
                    if cur_dof_index >= len(dof_names):
                        break
            self._dof_offsets.append(len(dof_names))
            self._dof_obs_size = len(self._dof_body_ids) * 6

            if (ENABLE_MAX_COORD_OBS):
                self._num_obs = 1 + len(body_names) * (3 + 6 + 3 + 3) - 3
            else:
                self._num_obs = 13 + self._dof_obs_size + self._num_dof + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        return

    def _build_termination_heights(self):
        head_term_height = 0.3
        shield_term_height = 0.32

        termination_height = self.cfg["env"]["terminationHeight"]
        self._termination_heights = np.array([termination_height] * self.num_bodies)

        self._humanoid_head_id = head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "Head")
        self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            left_arm_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "left_lower_arm")
            self._termination_heights[left_arm_id] = max(shield_term_height, self._termination_heights[left_arm_id])
        
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return

    def _load_humanoid_assets(self):
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        #asset_options.fix_base_link = True

        return self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.humanoid_asset = humanoid_asset = self._load_humanoid_assets()
        self._setup_character_props(self.cfg["env"]["keyBodies"])

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        if self.cfg["env"]["asset"]["assetFileName"] in {"mjcf/smpl_humanoid_v1.xml"}:
            right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "R_Ankle")
            left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "L_Ankle")
        else:
            right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
            left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, humanoid_asset)
            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = 1
        
        start_pose = gymapi.Transform()
        char_h = 0.89
        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        self.humanoid_handles.append(humanoid_handle)

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(self._dof_offsets) - 1
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]

            if (dof_size == 3):
                curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                curr_low = np.max(np.abs(curr_low))
                curr_high = np.max(np.abs(curr_high))
                curr_scale = max([curr_low, curr_high])
                curr_scale = 1.2 * curr_scale
                curr_scale = min([curr_scale, np.pi])

                lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
                lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale
                
                #lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                #lim_high[dof_offset:(dof_offset + dof_size)] = np.pi


            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset[:] = 0.0
        self._pd_action_scale[:] = np.pi
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (ENABLE_MAX_COORD_OBS):
            if (env_ids is None):
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
            else:
                body_pos = self._rigid_body_pos[env_ids]
                body_rot = self._rigid_body_rot[env_ids]
                body_vel = self._rigid_body_vel[env_ids]
                body_ang_vel = self._rigid_body_ang_vel[env_ids]
        
            obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                    self._root_height_obs)

        else:
            if (env_ids is None):
                root_pos = self._rigid_body_pos[:, 0, :]
                root_rot = self._rigid_body_rot[:, 0, :]
                root_vel = self._rigid_body_vel[:, 0, :]
                root_ang_vel = self._rigid_body_ang_vel[:, 0, :]
                dof_pos = self._dof_pos
                dof_vel = self._dof_vel
                key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            else:
                root_pos = self._rigid_body_pos[env_ids][:, 0, :]
                root_rot = self._rigid_body_rot[env_ids][:, 0, :]
                root_vel = self._rigid_body_vel[env_ids][:, 0, :]
                root_ang_vel = self._rigid_body_ang_vel[env_ids][:, 0, :]
                dof_pos = self._dof_pos[env_ids]
                dof_vel = self._dof_vel[env_ids]
                key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
        
            obs = compute_humanoid_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                                dof_pos, dof_vel,
                                                key_body_pos, self._local_root_obs,
                                                self._root_height_obs, self._dof_obs_size,
                                                self._dof_offsets)
        return obs

    def _reset_actors(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        return

    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def step(self, actions):
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

    def get_states(self):
        return self.states_buf

    def get_actor_params_info(self, dr_params, env):
        """Returns a flat array of actor params, their names and ranges."""
        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name+'_'+str(prop_idx)+'_'+attr
                        lo_hi = attr_randomization_params['range']
                        distr = attr_randomization_params['distribution']
                        if 'uniform' not in distr:
                            lo_hi = (-1.0*float('Inf'), float('Inf'))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name+'_'+str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    def apply_randomizations(self, dr_params):
        # Apply randomizations only on resets, due to current PhysX limitations
        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])

                    self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda}

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

                    self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)}

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        for actor, actor_properties in dr_params["actor_params"].items():
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue
                    if prop_name == 'scale':
                        attr_randomization_params = prop_attrs
                        sample = generate_random_samples(attr_randomization_params, 1,
                                                         self.last_step, None)
                        og_scale = 1
                        if attr_randomization_params['operation'] == 'scaling':
                            new_scale = og_scale * sample
                        elif attr_randomization_params['operation'] == 'additive':
                            new_scale = og_scale + sample
                        self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], p, attr)
                                apply_random_samples(
                                    p, og_p, attr, attr_randomization_params,
                                    self.last_step, smpl)
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            smpl = None
                            if self.actor_params_generator is not None:
                                smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                    extern_sample, extern_offsets[env_id], prop, attr)
                            apply_random_samples(
                                prop, self.original_props[prop_name], attr,
                                attr_randomization_params, self.last_step, smpl)

                    setter = param_setters_map[prop_name]
                    default_args = param_setter_defaults_map[prop_name]
                    setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print('env_id', env_id,
                              'extern_offset', extern_offsets[env_id],
                              'vs extern_sample.shape', extern_sample.shape)
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False

    def _physics_step(self):
        self.render()
        for i in range(self.control_freq_inv):
            self.gym.simulate(self.sim)
        return


class HumanoidSMPLIM(HumanoidSMPL):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        self.cfg = cfg
        self.device = "cpu"
        self.model = None
        self.args = args = cfg['args']
        if device_type == "cuda" or device_type == "GPU":
            self.device = "cuda" + ":" + str(device_id)
        
        self.has_shape_obs = cfg["env"].get("has_shape_obs", False)
        self.has_self_collision = cfg["env"].get("has_self_collision", False)
        self.residual_force_scale = cfg["env"].get("residual_force_scale", 0.0)
        self.residual_torque_scale = cfg["env"].get("residual_torque_scale", self.residual_force_scale)
        self.kp_scale = cfg["env"].get("kp_scale", 1.0)
        self.kd_scale = cfg["env"].get("kd_scale", self.kp_scale)
        self.obs_type = cfg['env'].get('obs_type', 'joint_pos_and_angle')
        self.context_length = cfg['env'].get('context_length', 32)
        self.context_padding = cfg['env'].get('context_padding', 8)
        self.truncate_time = cfg['env'].get('truncate_time', True)
        self.pd_tar_lim = cfg['env'].get('pd_tar_lim', 0.5) * np.pi

        control_freq_inv = cfg["env"]["controlFrequencyInv"]
        self._motion_sync_dt = control_freq_inv * sim_params.dt
            
        if args.test:
            cfg["env"]["stateInit"] = 'Start'
            if 'test_motion_file' in cfg['env']:
                cfg['env']['motion_file'] = cfg['env']['test_motion_file']
        
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidSMPLIM.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        if ("enableHistObs" in cfg["env"]):
            self._enable_hist_obs = cfg["env"]["enableHistObs"]
        else:
            self._enable_hist_obs = False

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._sub_rewards = None
        self._sub_rewards_names = None

        self.ground_tolerance = cfg['env'].get('ground_tolerance', 0.0)
        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        body_weights = cfg['env'].get('body_pos_weights', dict())
        self.body_pos_weights = torch.ones(self.num_bodies, device=self.device)
        for val, bodies in body_weights.items():
            for body in bodies:
                ind = self.body_names.index(body)
                self.body_pos_weights[ind] = val

        return

    def register_model(self, model):
        self.model = model

    def post_epoch(self, epoch, cur_reward):
        # import ipdb; ipdb.set_trace()
        if self.cfg['rm_rfc_during_finetune']:
            if cur_reward > 0.85:
                self.residual_force_scale *= 0.5
                self.residual_torque_scale *= 0.5
                print(f'[INFO] reducing RFC to {self.residual_force_scale}')
                if self.residual_force_scale < 0.05: 
                    return True
                return False
            else:
                return False
        else:
            if cur_reward > 0.85: 
                return True 
            else: 
                return False
    
    def pre_epoch(self, epoch):
        return

    def pre_physics_step(self, actions):
        """
        self.actions = actions.to(self.device).clone()

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        
        """
        actions[self.reset_buf == 1] = 0

        self.actions = actions.to(self.device).clone()
        dof_actions = self.actions[:, :self._num_dof]

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(dof_actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
            self.pd_torque = (pd_tar - self._dof_pos) * self.stiffness
        else:
            forces = dof_actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        # residual forces
        if self.residual_force_scale > 0:
            res_force = self.actions[:, self._num_dof: self._num_dof + 3].clone() * self.residual_force_scale
            res_torque = self.actions[:, self._num_dof + 3: self._num_dof + 6].clone() * self.residual_torque_scale
            root_rot = remove_base_rot(self._rigid_body_rot[:, 0, :])
            root_heading_q = torch_utils.calc_heading_quat(root_rot)
            res_force = torch_utils.my_quat_rotate(root_heading_q, res_force)
            res_torque = torch_utils.my_quat_rotate(root_heading_q, res_torque)

            forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            forces[:, 0, :] = res_force
            torques[:, 0, :] = res_torque
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

        self._save_prev_target_motion_state()
        return

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)
        self.body_names = body_names = self.gym.get_asset_rigid_body_names(self.humanoid_asset)
        self.dof_names = dof_names = self.gym.get_asset_dof_names(self.humanoid_asset)
        self._dof_body_ids = []
        self._dof_offsets = []
        cur_dof_index = 0
        dof_body_names = [x[:-2] for x in dof_names]
        for i, body in enumerate(body_names):
            if body != dof_body_names[cur_dof_index]:
                continue
            self._dof_body_ids.append(i)
            self._dof_offsets.append(cur_dof_index)
            while body == dof_body_names[cur_dof_index]:
                cur_dof_index += 1
                if cur_dof_index >= len(dof_names):
                    break
        self._dof_offsets.append(len(dof_names))
        self._dof_obs_size = len(self._dof_body_ids) * 6

        self._num_actions = self._num_dof = len(dof_names)
        if self.residual_force_scale > 0:
            self._num_actions += 6
        
        num_bodies = len(body_names)
        shape_dict = {
            'body_pos': (num_bodies, 3),
            'body_pos_gt': (num_bodies, 3),
            'body_rot': (num_bodies, 4),
            'dof_pos': (self._num_dof,),
            'dof_pos_gt': (self._num_dof,),
            'dof_vel': (self._num_dof,),
            'body_vel': (num_bodies, 3),
            'body_ang_vel': (num_bodies, 3),
            'motion_bodies': (self._motion_lib._motion_bodies.shape[-1],),
            'joint_conf': (num_bodies,)
        }

        self.obs_names = ['body_pos', 'body_rot', 'dof_pos', 'dof_vel', 'body_vel', 'body_ang_vel', 'motion_bodies']
        self.obs_shapes = [shape_dict[x] for x in self.obs_names]
        self.obs_dims = [np.prod(x) for x in self.obs_shapes]

        self.context_names = ['body_pos', 'body_rot', 'dof_pos', 'body_pos_gt', 'dof_pos_gt']
        if 'transform_specs' in self.cfg['env']:
            self.context_names.append('joint_conf')

        self.context_shapes = [shape_dict[x] for x in self.context_names]
        self.context_dims = [np.prod(x) for x in self.context_shapes]

        self.is_env_dim_setup = False

        self._num_obs = sum(self.obs_dims)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()
        return

    def _build_termination_heights(self):
        head_term_height = self.cfg["env"]["terminationHeadHeight"]
        default_termination_height = self.cfg["env"]["terminationBodyHeight"]
        self._termination_heights = np.array([default_termination_height] * self.num_bodies)
        self._humanoid_head_id = head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "Head")
        self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return

    def _resample_amass_motions(self):
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self._setup_tensors()

    def _create_envs(self, num_envs, spacing, num_per_row):

        robot_cfg = {
            "mesh": True,
            "model": "smpl",
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
        }

        smpl_robot = Robot(
            robot_cfg,
            data_dir= "data/smpl",
        )

        if self.cfg['env'].get('sample_first_motions', False):
            self._reset_ref_motion_ids = torch.arange(self.num_envs, device=self._motion_lib._device) % self._motion_lib.num_motions()
        else:
            weights_from_lenth = self.cfg['env'].get('motion_weights_from_length', False)
            self._reset_ref_motion_ids = self._motion_lib.sample_motions(num_envs, weights_from_lenth=weights_from_lenth)
        if 'motion_id' in self.cfg['env']:
            self._reset_ref_motion_ids[:] = self.cfg['env']['motion_id']
        self._reset_ref_motion_bodies = self._motion_lib._motion_bodies[self._reset_ref_motion_ids].to(self.device)
        all_motion_bodies = self._motion_lib._motion_bodies.cpu()
        all_motion_body_scales = self._motion_lib._motion_body_scales.cpu().numpy() if hasattr(self._motion_lib, '_motion_body_scales') else np.ones(all_motion_bodies.shape[0])
        motion_ids = self._reset_ref_motion_ids.cpu().numpy()

        print('sampled motion ids:', self._reset_ref_motion_ids)

        unique_motion_ids = np.unique(motion_ids)

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = ""

        self.humanoid_masses = []
        self.humanoid_assets = dict()
        self.humanoid_files = dict()
        self.humanoid_rest_joints = dict()

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        t_start = time.time()
        res_acc = []

        def _create_smpl_humanoid_xml(humanoid_ids, queue, smpl_robot, motion_bodies, body_scales, pid):
            res = []
            for idx in humanoid_ids:
                model_xml_path = f"{TMP_SMPL_DIR}/smpl_humanoid_{idx}.xml"
                gender_beta = motion_bodies[idx]
                rest_joints = smpl_robot.load_from_skeleton(betas=gender_beta[None, 1:], gender=gender_beta[:1], scale=body_scales[idx], model_xml_path=model_xml_path)
                smpl_robot.write_xml(model_xml_path)
                res.append((idx, model_xml_path, rest_joints))

            if not queue is None:
                queue.put(res)
            else:
                return res

        for _, idx in enumerate(tqdm(unique_motion_ids)):
            res_acc += (_create_smpl_humanoid_xml([idx], None, smpl_robot, all_motion_bodies, all_motion_body_scales, 0))
        t_finish_gen = time.time()
        print(f"Finished generating {len(unique_motion_ids)} humanoids in {t_finish_gen - t_start:.3f}s!")

        for humanoid_config in res_acc:
            humanoid_idx, asset_file_real, rest_joints = humanoid_config
            humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file_real, asset_options)

            # create force sensors at the feet
            right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
            left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
            sensor_pose = gymapi.Transform()

            self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
            self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)
            self.humanoid_assets[humanoid_idx] = humanoid_asset
            self.humanoid_files[humanoid_idx] = asset_file_real
            self.humanoid_rest_joints[humanoid_idx] = rest_joints

        self.humanoid_asset = humanoid_asset = next(iter(self.humanoid_assets.values()))
        self._setup_character_props(self.cfg["env"]["keyBodies"])

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        self.smpl_rest_joints = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, self.humanoid_assets[motion_ids[i]])
            self.envs.append(env_ptr)
            self.smpl_rest_joints.append(self.humanoid_rest_joints[motion_ids[i]])
        self.smpl_rest_joints = torch.from_numpy(np.stack(self.smpl_rest_joints)).to(self.device)
        self.smpl_parents = smpl_robot.smpl_parser.parents.to(self.device)
        self.smpl_children = smpl_robot.smpl_parser.children_map.to(self.device)
        self.humanoid_masses = np.array(self.humanoid_masses)
        print("Humanoid weights:", np.array2string(self.humanoid_masses[:32], precision=2, separator=","))

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        # shutil.rmtree(TMP_SMPL_DIR, ignore_errors=True)
        print(f"Finished loading {num_envs} humanoids in {time.time() - t_finish_gen:.3f}s!")
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = 0
        if not self.has_self_collision:
            col_filter = 1

        start_pose = gymapi.Transform()
        char_h = 0.89
        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)
        humanoid_mass = np.sum([prop.mass for prop in self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)])
        self.humanoid_masses.append(humanoid_mass)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            pd_scale = humanoid_mass / self.cfg['env'].get('default_humanoid_mass', 90.0)
            dof_prop['stiffness'] *= pd_scale * self.kp_scale
            dof_prop['damping'] *= pd_scale * self.kd_scale
            self.stiffness = torch.from_numpy(dof_prop['stiffness']).to(self.device)
            self.damping = torch.from_numpy(dof_prop['damping']).to(self.device)

            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        self.humanoid_handles.append(humanoid_handle)

        return

    def _action_to_pd_targets(self, action):
        pd_tar = action
        pd_lower = self._dof_pos - self.pd_tar_lim
        pd_upper = self._dof_pos + self.pd_tar_lim
        pd_tar = torch.maximum(torch.minimum(pd_tar, pd_upper), pd_lower)
        return pd_tar

    def post_physics_step(self):
        self.progress_buf += 1
        self._rigid_body_states_simulated = True

        self._cur_ref_motion_times += self.dt
        self._set_target_motion_state()

        self._refresh_sim_tensors()
        self._compute_observations()

        # <--------- compute reward
        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        body_vel = self._rigid_body_vel
        body_ang_vel = self._rigid_body_ang_vel
        dof_pos = self._dof_pos
        dof_vel = self._dof_vel
        target_pos = self._prev_target_rb_pos
        target_rot = self._prev_target_rb_rot
        target_dof_pos = self._prev_target_dof_pos
        target_dof_vel = self._prev_target_dof_vel

        reward_specs = {'k_dof': 60, 'k_vel': 0.2, 'k_pos': 100, 'k_rot': 40, 'w_dof': 0.6, 'w_vel': 0.1, 'w_pos': 0.2, 'w_rot': 0.1}
        cfg_reward_specs = self.cfg['env'].get('reward_specs', dict())
        reward_specs.update(cfg_reward_specs)

        self.rew_buf[:], self._sub_rewards, self._sub_rewards_names = compute_humanoid_reward(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, target_dof_pos, target_dof_vel, body_vel, body_ang_vel, self._dof_obs_size, self._dof_offsets, self.body_pos_weights, reward_specs)
        reset_mask = self.reset_buf == 1
        if torch.any(reset_mask):
            self.rew_buf[reset_mask] = 0
            self._sub_rewards[reset_mask] = 0
        # -------->


        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf
        self.extras["sub_rewards"] = self._sub_rewards
        self.extras["sub_rewards_names"] = self._sub_rewards_names

        # debug viz
        if self.viewer and self.debug_viz:
            # self._update_debug_viz()
            self.gym.clear_lines(self.viewer)
        
        return

    def _load_motion(self, motion_file):
        gpu_motion_lib = self.cfg['env'].get('gpu_motion_lib', True)
        device = self.device if gpu_motion_lib else 'cpu'
         
        if os.path.isdir(motion_file):
            self.motion_lib_files = sorted(glob.glob(f'{motion_file}/*.pth'))
            print(f'self.motion_lib_files: {self.motion_lib_files}')
            motion_file_range = self.cfg['env'].get('motion_file_range', None)
            if motion_file_range is not None:
                self.motion_lib_files = self.motion_lib_files[motion_file_range[0]:motion_file_range[1]]
            motion_libs = [torch.load(f, map_location=device) for f in self.motion_lib_files]
            self._motion_lib = motion_libs[0]
            self._motion_lib.merge_multiple_motion_libs(motion_libs[1:])
            print(f'Loading motion files to {device}:')
            for f in self.motion_lib_files:
                print(f)
        else:
            # self.motion_lib_files = [motion_file]
            # self._motion_lib = torch.load(motion_file, map_location=device)
            if self.args.train_low_imitator:
                pass
            else:
                ##########################
                from embodied_pose.utils.motion_lib import MotionLib
                from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
                from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES as joint_names
                from uhc.smpllib.smpl_local_robot import Robot
                from scipy.spatial.transform import Rotation as sRot

                info = joblib.load('data/misc/smpl_body_info.pkl')
                robot_cfg = {
                    "mesh": True,
                    "model": "smpl",
                    "body_params": {},
                    "joint_params": {},
                    "geom_params": {},
                    "actuator_params": {},
                }

                model_xml_path = f"/tmp/smpl/smpl_mesh_humanoid_v1_convert.xml"
                smpl_local_robot = Robot(robot_cfg,data_dir= "data/smpl",model_xml_path=model_xml_path)
                mujoco_joint_names = [
                    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
                    'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
                    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
                    'R_Elbow', 'R_Wrist', 'R_Hand'
                ]
                smpl_2_mujoco = [joint_names.index(q) for q in mujoco_joint_names if q in joint_names]
                motion_lib_input_dict = dict()
                smpl_data_entry=np.load(motion_file)
                
                # import ipdb;ipdb.set_trace()
                if self.args.cvt_npz_to_amass:
                    sys.path.insert(0, '/apdcephfs/private_wallyliang/PLANT/Thirdparty/Pose_to_SMPL/fit/tools')
                    from merge import cvt_npz_data_to_amss_axis
                    smpl_data_entry = dict(smpl_data_entry)
                    smpl_data_entry = cvt_npz_data_to_amss_axis(smpl_data_entry)
                    
                # import ipdb;ipdb.set_trace()
                pose_aa = smpl_data_entry['poses'].copy()[:,:22].reshape(-1,66)
                trans = smpl_data_entry['trans'].copy()
                beta = smpl_data_entry['betas'][:10].copy()
                gender = smpl_data_entry['gender']
                fps = 30.0

                if isinstance(gender, np.ndarray):
                    gender = gender.item()
                if isinstance(gender, bytes):
                    gender = gender.decode("utf-8")
                if gender == "neutral":
                    gender_number = [0]
                    smpl_parser = smpl_local_robot.smpl_parser_n
                elif gender == "male":
                    gender_number = [1]
                    smpl_parser = smpl_local_robot.smpl_parser_m
                elif gender == "female":
                    gender_number = [2]
                    smpl_parser = smpl_local_robot.smpl_parser_f
                else:
                    import ipdb
                    ipdb.set_trace()
                    raise Exception("Gender Not Supported!!")
                
                batch_size = pose_aa.shape[0]
                pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)  # TODO: need to extract correct handle rotations instead of zero
                pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)[..., smpl_2_mujoco, :]
                smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None, ]), gender=gender_number)
                # TODO: check betas
                smpl_local_robot.write_xml()
                skeleton_tree = SkeletonTree.from_mjcf(model_xml_path)
                self.sk_tree = skeleton_tree
                self.smpl_robot = smpl_local_robot



                # robot_cfg = {
                #     "mesh": True,
                #     "replace_feet": True,
                #     "rel_joint_lm": True,
                #     "upright_start": True,
                #     "remove_toe": False,
                #     "freeze_hand": True, 
                #     "real_weight_porpotion_capsules": False,
                #     "real_weight": False,
                #     "masterfoot": False,
                #     "master_range": 30,
                #     "big_ankle": False,
                #     "box_body": False,
                #     "model": "smpl",
                #     "body_params": {},
                #     "joint_params": {},
                #     "geom_params": {},
                #     "actuator_params": {},
                # }
                # asset_id = uuid4()
                # asset_file_real = f"/tmp/smpl/smpl_humanoid_{asset_id}.xml"
                # model_xml_path = f"embodied_pose/data/assets/mjcf/smpl_mesh_humanoid_v1_test.xml"
                # smpl_robot = Robot(robot_cfg, model_xml_path=model_xml_path,data_dir="data/smpl",)
                # gender_beta = np.zeros(17)
                # smpl_robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]), gender=gender_beta[0:1], objs_info=None)
                # smpl_robot.write_xml(asset_file_real,)
                # self.sk_tree = SkeletonTree.from_mjcf(asset_file_real)




                root_trans = trans + skeleton_tree.local_translation[0].numpy()
                verts, joints = smpl_parser.get_joints_verts(pose=torch.from_numpy(pose_aa),th_betas=torch.from_numpy(beta[None, ]),th_trans=torch.from_numpy(trans))
                # min_verts_h = verts[..., 2].min().item()
                min_verts_h = verts[..., 2].min(dim=-1)[0].mean().item()

                new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,torch.from_numpy(pose_quat),torch.from_numpy(root_trans),is_local=True)
                new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
                
                new_motion_out = new_motion.to_dict()
                new_motion_out['seq_name'] = 'test'
                new_motion_out['seq_idx'] = 0
                new_motion_out['trans'] = trans
                new_motion_out['root_trans'] = root_trans
                new_motion_out['pose_aa'] = pose_aa
                new_motion_out['beta'] = beta
                new_motion_out['beta_idx'] = 0
                new_motion_out['gender'] = gender
                new_motion_out['min_verts_h'] = min_verts_h
                new_motion_out['body_scale'] = 1.0
                new_motion_out['__name__'] = "SkeletonMotion"
                motion_lib_input_dict['test'] = new_motion_out
                self._motion_lib = MotionLib(motion_file=motion_lib_input_dict,
                    dof_body_ids=info['dof_body_ids'],
                    dof_offsets=info['dof_offsets'],
                    key_body_ids=info['key_body_ids'],
                    device=device,
                    clean_up=True
                )
                smpl_local_robot.clean_up()
                print(f'Loading motion file to {device}: {motion_file}')
                ##########################


        self._motion_lib._device = device
        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        if len(env_ids) > 0:
            self._state_reset_happened = True
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)

        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self._state_reset_happened:
            env_ids = self._reset_ref_env_ids
            self._rigid_body_pos[env_ids] = self._reset_rb_pos
            self._rigid_body_rot[env_ids] = self._reset_rb_rot
            self._rigid_body_vel[env_ids] = 0
            self._rigid_body_ang_vel[env_ids] = 0
            self._state_reset_happened = False

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidSMPLIM.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidSMPLIM.StateInit.Start
              or self._state_init == HumanoidSMPLIM.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidSMPLIM.StateInit.Hybrid):
            # self._reset_hybrid_state_init(env_ids)
            # def _reset_hybrid_state_init(self, env_ids):
            num_envs = env_ids.shape[0]
            ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
            ref_init_mask = torch.bernoulli(ref_probs) == 1.0

            ref_reset_ids = env_ids[ref_init_mask]
            if (len(ref_reset_ids) > 0):
                self._reset_ref_state_init(ref_reset_ids)

            default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
            if (len(default_reset_ids) > 0):
                self._reset_default(default_reset_ids)
       
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        use_env_ids = not (len(env_ids) == self.num_envs and torch.all(env_ids == torch.arange(self.num_envs, device=self.device)))
        num_envs = env_ids.shape[0]
        motion_ids = self._reset_ref_motion_ids
        if use_env_ids:
            motion_ids = motion_ids[env_ids]

        if (self._state_init == HumanoidSMPLIM.StateInit.Random
            or self._state_init == HumanoidSMPLIM.StateInit.Hybrid):
            truncate_time = self.context_length * self.dt if self.truncate_time else None
            motion_times = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time).to(self.device)
        elif (self._state_init == HumanoidSMPLIM.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot \
               = self._motion_lib.get_motion_state(motion_ids, motion_times.to(self._motion_lib._device), return_rigid_body=True, device=self.device, adjust_height=True, ground_tolerance=self.ground_tolerance)

        # <------ set env state
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        self._rigid_body_pos[env_ids] = rb_pos
        self._rigid_body_rot[env_ids] = rb_rot
        self._rigid_body_vel[env_ids] = 0
        self._rigid_body_ang_vel[env_ids] = 0
        self._reset_rb_pos = rb_pos.clone()
        self._reset_rb_rot = rb_rot.clone()

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        # ----------->

        self._reset_ref_env_ids = env_ids
        if not use_env_ids:
            self._reset_ref_motion_times = motion_times
            self._cur_ref_motion_times = self._reset_ref_motion_times.clone()
        else:
            self._reset_ref_motion_times[env_ids] = motion_times
            self._cur_ref_motion_times[env_ids] = motion_times
        self._set_target_motion_state(env_ids=env_ids if use_env_ids else None)

        # <----------- init context
        self._init_context(motion_ids, motion_times)
        # -------->
        return

    def _init_context(self, motion_ids, motion_times):
        motion_times = motion_times + self.dt
        context_padded_length = self.context_length + self.context_padding * 2
        all_motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, context_padded_length])
        time_steps = self.dt * torch.arange(-self.context_padding, self.context_length + self.context_padding, device=motion_times.device)
        all_motion_times = motion_times.unsqueeze(-1) + time_steps

        all_motion_ids_flat = all_motion_ids.view(-1)
        all_motion_times_flat = all_motion_times.view(-1)

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot \
               = self._motion_lib.get_motion_state(all_motion_ids_flat, all_motion_times_flat.to(self._motion_lib._device), return_rigid_body=True, device=self.device, adjust_height=True, ground_tolerance=self.ground_tolerance)

        context_dict = {
            'body_pos': rb_pos,
            'body_rot': rb_rot,
            'dof_pos': dof_pos,
            'body_pos_gt': rb_pos.clone(),
            'dof_pos_gt': dof_pos.clone()
        }

        # <<----------############## transform target
        # self._transform_target(context_dict)
        # def _transform_target(self, context_dict):
        transform_specs = self.cfg['env'].get('transform_specs', dict())
        context_dict['joint_conf'] = joint_conf = torch.ones_like(context_dict['body_pos'][..., 0])
        for transform, specs in transform_specs.items():
            if transform == 'mask_joints':
                body_pos = context_dict['body_pos']
                joint_index = [self.body_names.index(joint) for joint in specs['joints']]
                joint_conf[..., joint_index] = 0.0
                context_dict['body_pos'] = body_pos * joint_conf.unsqueeze(-1)
            elif transform == 'noisy_joints':
                noise_std = torch.ones_like(context_dict['joint_conf']) * specs['noise_std']
                noise_mask = torch.bernoulli(torch.ones(context_dict['joint_conf'].shape) * specs['prob'])
                noise_std[noise_mask == 0.0] = 0.0
                noise = torch.randn_like(context_dict['body_pos']) * noise_std.unsqueeze(-1)
                noise_norm = noise.norm(dim=-1) / (np.sqrt(3) * specs['conf_std'])
                conf = (1 - torch.tensor(norm.cdf(noise_norm.cpu()), device=noise.device, dtype=noise.dtype)) * 2
                context_dict['body_pos'] += noise
                context_dict['joint_conf'] = conf
                # remove occluded joints
                occluded_joints = context_dict['joint_conf'] < specs['min_conf']
                context_dict['joint_conf'][occluded_joints] = 0.0
                context_dict['body_pos'][occluded_joints] = 0.0
            elif transform == 'mask_random_joints':
                drop_mask = torch.bernoulli(torch.ones(context_dict['joint_conf'].shape) * specs['prob']) == 1.0
                drop_mask[..., 0] = 0.0
                context_dict['joint_conf'][drop_mask] = 0.0
                context_dict['body_pos'][drop_mask] = 0.0
        # ------->>

        self.context_feat = torch.cat([context_dict[x].view(context_dict[x].shape[0], -1) for x in self.context_names], dim=-1)
        self.context_feat = self.context_feat.view(self.num_envs, -1, self.context_feat.shape[-1])

        self.context_mask = all_motion_times <= (self._motion_lib._motion_lengths[self._reset_ref_motion_ids] + 2 * self.dt).unsqueeze(-1)

        if self.model is not None:
            if not self.is_env_dim_setup:
                self.model.a2c_network.setup_env_named_dims(self.obs_names, self.obs_shapes, self.obs_dims, self.context_names, self.context_shapes, self.context_dims)
                self.is_env_dim_setup = True
            with torch.no_grad():
                self.model.a2c_network.forward_context(self.context_feat, self.context_mask)

    def _set_target_motion_state(self, env_ids=None):
        if env_ids is None:
            motion_ids = self._reset_ref_motion_ids
            motion_times = self._cur_ref_motion_times + self.dt     # next frame
        else:
            motion_ids = self._reset_ref_motion_ids[env_ids]
            motion_times = self._cur_ref_motion_times[env_ids] + self.dt     # next frame
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot \
               = self._motion_lib.get_motion_state(motion_ids, motion_times.to(self._motion_lib._device), return_rigid_body=True, device=self.device, adjust_height=True, ground_tolerance=self.ground_tolerance)
        # new target
        if env_ids is None:
            self._target_root_pos = root_pos
            self._target_root_rot = root_rot
            self._target_dof_pos = dof_pos
            self._target_root_vel = root_vel
            self._target_root_ang_vel = root_ang_vel
            self._target_dof_vel = dof_vel
            self._target_key_pos = key_pos
            self._target_rb_pos = rb_pos
            self._target_rb_rot = rb_rot
        else:
            self._target_root_pos[env_ids] = root_pos
            self._target_root_rot[env_ids] = root_rot
            self._target_dof_pos[env_ids] = dof_pos
            self._target_root_vel[env_ids] = root_vel
            self._target_root_ang_vel[env_ids] = root_ang_vel
            self._target_dof_vel[env_ids] = dof_vel
            self._target_key_pos[env_ids] = key_pos
            self._target_rb_pos[env_ids] = rb_pos
            self._target_rb_rot[env_ids] = rb_rot
        return

    def _save_prev_target_motion_state(self):
        # previous target
        self._prev_target_root_pos = self._target_root_pos.clone()
        self._prev_target_root_rot = self._target_root_rot.clone()
        self._prev_target_dof_pos = self._target_dof_pos.clone()
        self._prev_target_root_vel = self._target_root_vel.clone()
        self._prev_target_root_ang_vel = self._target_root_ang_vel.clone()
        self._prev_target_dof_vel = self._target_dof_vel.clone()
        self._prev_target_key_pos = self._target_key_pos.clone()
        self._prev_target_rb_pos = self._target_rb_pos.clone()
        self._prev_target_rb_rot = self._target_rb_rot.clone()

    def _compute_humanoid_obs(self, env_ids=None):
        obs_dict = {
            'body_pos': self._rigid_body_pos,
            'body_rot': self._rigid_body_rot,
            'body_vel': self._rigid_body_vel,
            'body_ang_vel': self._rigid_body_ang_vel,
            'dof_pos': self._dof_pos,
            'dof_vel': self._dof_vel,
            'target_pos': self._target_rb_pos,
            'target_rot': self._target_rb_rot,
            'target_dof_pos': self._target_dof_pos,
            'motion_bodies': self._reset_ref_motion_bodies
        }
        obs = [(obs_dict[x] if env_ids is None else obs_dict[x][env_ids]) for x in self.obs_names]
        obs = torch.cat([x.reshape(x.shape[0], -1) for x in obs], dim=-1)
        return obs

    def get_aux_losses(self, model_res_dict):
        aux_loss_specs = self.cfg['env'].get('aux_loss_specs', dict())
        context = model_res_dict['extra']['context']
        aux_losses = {}
        aux_losses_weighted = {}

        # dof loss
        w_dof = aux_loss_specs.get('w_dof', 0.0)
        if w_dof > 0:
            dof_pos = context['dof_pos']
            target_dof_pos = context['dof_pos_gt']
            dof_obs = angle_axis_to_rot6d(dof_pos.view(*dof_pos.shape[:-1], -1, 3)).view(*dof_pos.shape[:-1], -1)
            target_dof_obs = angle_axis_to_rot6d(target_dof_pos.view(*target_dof_pos.shape[:-1], -1, 3)).view(*target_dof_pos.shape[:-1], -1)
            diff_dof_obs = dof_obs - target_dof_obs
            dof_obs_loss = (diff_dof_obs ** 2).mean()
            aux_losses['aux_dof_rot6d_loss'] = dof_obs_loss
            aux_losses_weighted['aux_dof_rot6d_loss'] = dof_obs_loss * w_dof

        # body pos loss
        w_pos = aux_loss_specs.get('w_pos', 0.0)
        if w_pos > 0:
            body_pos = context['body_pos']
            target_pos = context['body_pos_gt']
            diff_body_pos = target_pos - body_pos
            diff_body_pos = diff_body_pos * self.body_pos_weights[:, None]
            body_pos_loss = (diff_body_pos ** 2).mean()
            aux_losses['aux_body_pos_loss'] = body_pos_loss
            aux_losses_weighted['aux_body_pos_loss'] = body_pos_loss * w_pos
        return aux_losses, aux_losses_weighted

    def _compute_reset(self):
        cur_ref_motion_times = self._cur_ref_motion_times
        ref_motion_lengths = self._motion_lib._motion_lengths.to(self.device)[self._reset_ref_motion_ids]

        old_reset_buf = self.reset_buf.clone()
        old_terminate_buf = self._terminate_buf.clone()
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._contact_forces, self._contact_body_ids,
                                                                           self._rigid_body_pos, self.max_episode_length,
                                                                           self._enable_early_termination, self._termination_heights,
                                                                           cur_ref_motion_times, ref_motion_lengths)
        reset_mask = old_reset_buf == 1
        if torch.any(reset_mask):
            self.reset_buf[reset_mask] = 1
            self._terminate_buf[reset_mask] = old_terminate_buf[reset_mask]
        return
    
    def start_recording(self):
        filename = self.cfg['env'].get('rec_fname', self._video_path % datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        self.writer = imageio.get_writer(filename, fps=30, quality=8, macro_block_size=None)
        self.frame_index = 0
        print(f"============ Writing video to {filename} ============")

    def start_recording_npz(self):
        self.frame_index = 0
        self.rec_pose_aa_list = []
        self.rec_trans_list = []
        print(f"============ start recording NPZ ============")

    def end_recording(self):
        self.writer.close()
        print(f"============ Video finished writing ============")

    def end_recording_npz(self):
        trans = torch.cat(self.rec_trans_list).cpu().numpy()
        poses = np.concatenate(self.rec_pose_aa_list)
        to_save = {
            'betas': np.zeros(10), # 10
            'trans': trans, # B,3
            'poses': np.concatenate([poses[:,:22,:],np.zeros((trans.shape[0],33,3))], axis=1).reshape(-1, 55, 3)[::1],
            'gender': 'male',
            'mocap_framerate': 30,
        }
        out_path = self.cfg['env']['record_npz_path']
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, **to_save)
        print(f"============ saved to {out_path} ============")

    def key_callback(self, key, action, mods):
        if action != glfw.RELEASE:
            return False
        if key == glfw.KEY_R:
            self.recording = not self.recording
            if self.recording:
                self.start_recording()
            else:
                self.end_recording()

        if key == glfw.KEY_C:
            print(f'cam_distance: {self.mj_viewer.cam.distance:.3f}')
            print(f'cam_elevation: {self.mj_viewer.cam.elevation:.3f}')
            print(f'cam_azimuth: {self.mj_viewer.cam.azimuth:.3f}')
        else:
            return False
            
        return True

    def render(self, sync_frame_time=False):
        if self.headless: return
        if self.viewer:

            # <--- init camera
            # self._update_camera()
            # def _update_camera(self):
            self.gym.refresh_actor_root_state_tensor(self.sim)
            char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

            cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
            cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
            cam_delta = cam_pos - self._cam_prev_char_pos

            new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
            new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0],
                                    char_root_pos[1] + cam_delta[1],
                                    cam_pos[2])

            self.gym.set_camera_location(self.viewer_camera_handle, self.envs[0],
                                        new_cam_pos, new_cam_target)
            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

            self._cam_prev_char_pos[:] = char_root_pos
            # return
            # --------->

            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "toggle_video_record" and evt.value > 0:
                    if not self.recording:
                        filename = self._video_path % datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                        self.writer = imageio.get_writer(filename, fps=30, quality=7, macro_block_size=None)
                        print(f"============ Writing video to {filename} ============")

                    self.recording = not self.recording
                    
                    if not self.recording:
                        self.writer.close()
                        print(f"============ Video finished writing ============")

            if self.recording:

                self.gym.render_all_camera_sensors(self.sim)
                color_image  = self.gym.get_camera_image(self.sim, self.envs[0], self.viewer_camera_handle, gymapi.IMAGE_COLOR)
                color_image = color_image.reshape(color_image.shape[0], -1 , 4)

                self.writer.append_data(color_image[..., :3])

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)

        return

    def render_vis(self, init=False):
        # if self.headless: return
        # self._sync_ref_motion(init)

        if self.mj_viewer is not None:
            dof_euler = self.convert_dof_pos_to_dof_euler(self._dof_pos[0])
            self.set_mj_actor_qpos(actor_qpos=self.mj_data.qpos[:self.nq], 
                                root_pos=self._rigid_body_pos[0, 0].cpu().numpy(), 
                                root_rot=self._rigid_body_rot[0, 0].cpu().numpy(), 
                                dof_pos=dof_euler.cpu().numpy())

            if not init:
                target_dof_euler = self.convert_dof_pos_to_dof_euler(self._prev_target_dof_pos[0])
                self.set_mj_actor_qpos(actor_qpos=self.mj_data.qpos[self.nq: 2 * self.nq], 
                                    root_pos=self._prev_target_root_pos[0].cpu().numpy(), 
                                    root_rot=self._prev_target_root_rot[0].cpu().numpy(), 
                                    dof_pos=target_dof_euler.cpu().numpy())

                self.mj_data.qpos[self.nq * 2:] = self.mj_data.qpos[:self.nq]
                self.mj_data.qpos[self.nq * 2 + 2] += 1000.0
            else:
                self.mj_data.qpos[self.nq: 2 * self.nq] = self.mj_data.qpos[:self.nq]
                self.mj_data.qpos[self.nq * 2:] = self.mj_data.qpos[:self.nq]
                self.mj_data.qpos[self.nq * 2 + 2] += 1000.0

            self.mj_data.qpos[self.nq] += 1.0
            self.mj_sim.forward()
            # return        

            # self.mj_viewer_setup(init)
            self.mj_viewer.cam.lookat[:2] = self.mj_data.qpos[:2]
            self.mj_viewer.cam.lookat[0] += 0.5
            self.mj_viewer.cam.lookat[2] = 0.8

            if not self.cam_inited:
                self.mj_viewer.cam.distance = self.mj_model.stat.extent * 1.1
                self.mj_viewer.cam.azimuth = 45
                self.mj_viewer.cam.elevation = -10
                self.cam_inited = True

            for _ in range(30 if self.recording else 10):
                self.mj_viewer.render()

            if self.recording:
                self.frame_index += 1
                self.writer.append_data(self.mj_viewer._read_pixels_as_in_window())
                if self.frame_index >= self.cfg['env'].get('num_rec_frames', 300):
                    self.recording = False
                    self.end_recording()
                    quit()
        
        # <------------
        if self.recording_npz:
            # import ipdb;ipdb.set_trace()
            assert(self._rigid_body_rot.shape[0] == 1)
            body_quat = self._rigid_body_rot
            root_trans = self._rigid_body_pos[:, 0, :]
                
            N = body_quat.shape[0]
            offset = self.sk_tree.local_translation[0].cuda()
            root_trans_offset = root_trans - offset
            
            # self.pre_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
            # pose_quat = (sRot.from_quat(body_quat.cpu().reshape(-1, 4).numpy()) * self.pre_rot).as_quat().reshape(N, -1, 4)
            pose_quat = (sRot.from_quat(body_quat.cpu().reshape(-1, 4).numpy())).as_quat().reshape(N, -1, 4)
            new_sk_state = SkeletonState.from_rotation_and_root_translation(self.sk_tree, torch.from_numpy(pose_quat), root_trans.cpu(), is_local=False)
            local_rot = new_sk_state.local_rotation
            pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(N, -1, 3)
            pose_aa = pose_aa[:, self.mujoco_2_smpl, :]

            self.rec_pose_aa_list.append(pose_aa)
            self.rec_trans_list.append(root_trans_offset)

            self.frame_index += 1
            self.record_npz_tqdm_bar.update(1)
            if self.frame_index >= self.cfg['env'].get('num_rec_frames', 300):
                self.recording_npz = False
                self.end_recording_npz()
                quit()
        # -------->

    def set_mj_actor_qpos(self, actor_qpos, root_pos, root_rot, dof_pos):
        actor_qpos[:3] = root_pos
        actor_qpos[3:7] = root_rot[[3, 0, 1, 2]]
        actor_qpos[7:] = dof_pos
        return

    def convert_dof_pos_to_dof_euler(self, dof_pos):
        dof_quat = angle_axis_to_quaternion(dof_pos.view(-1, 3))
        dof_euler = ypr_euler_from_quat(dof_quat)[..., [2, 1, 0]].reshape(-1)
        return dof_euler



#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets):
    # type: (Tensor, int, List[int]) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # jp hack
        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif (dof_size == 1):
            axis = torch.tensor([0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device)
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            joint_pose_q = None
            assert(False), "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size):((j + 1) * joint_obs_size)] = joint_dof_obs

    assert((num_joints * joint_obs_size) == dof_obs_size)

    return dof_obs

@torch.jit.script
def compute_humanoid_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos,
                                  local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = torch_utils.my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    return obs

@torch.jit.script
def remove_base_rot(quat):
    # ZL: removing the base rotation for SMPL model
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat))
    return quat_mul(quat, base_rot.repeat(quat.shape[0], 1))

@torch.jit.script
def compute_humanoid_observations_imitation(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, target_dof_pos, body_vel, body_ang_vel, motion_bodies, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    root_rot = remove_base_rot(root_rot)
    heading_rot, heading = torch_utils.calc_heading_quat_inv_with_heading(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    """target"""
    # target root height    [N, 1]
    target_root_pos = target_pos[:, 0, :]
    target_root_rot = target_rot[:, 0, :]
    target_rel_root_h = root_h - target_root_pos[:, 2:3]
    # target root rotation  [N, 6]
    target_root_rot = remove_base_rot(target_root_rot)
    target_heading_rot, target_heading = torch_utils.calc_heading_quat_inv_with_heading(target_root_rot)
    target_rel_root_rot = quat_mul(target_root_rot, quat_conjugate(root_rot))
    target_rel_root_rot_obs = torch_utils.quat_to_tan_norm(target_rel_root_rot)
    # target 2d pos [N, 2]
    target_rel_pos = target_root_pos[:, :3] - root_pos[:, :3]
    target_rel_pos = torch_utils.my_quat_rotate(heading_rot, target_rel_pos)
    target_rel_2d_pos = target_rel_pos[:, :2]
    # target heading    [N, 2]
    target_rel_heading = target_heading - heading
    target_rel_heading_vec = heading_to_vec(target_rel_heading)
    # target target dof   [N, dof]
    target_rel_dof_pos = target_dof_pos - dof_pos
    # target body pos   [N, 3xB]
    target_rel_body_pos = target_pos - body_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1], target_rel_body_pos.shape[2])
    flat_target_rel_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_target_rel_body_pos)
    target_rel_body_pos = flat_target_rel_body_pos.reshape(target_rel_body_pos.shape[0], target_rel_body_pos.shape[1] * target_rel_body_pos.shape[2])
    # target body rot   [N, 6xB]
    target_rel_body_rot = quat_mul(quat_conjugate(body_rot), target_rot)
    target_rel_body_rot_obs = torch_utils.quat_to_tan_norm(target_rel_body_rot.view(-1, 4)).view(target_rel_body_rot.shape[0], -1)


    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, dof_vel,
                     target_rel_root_h, target_rel_root_rot_obs, target_rel_2d_pos, target_rel_heading_vec, target_rel_dof_pos, target_rel_body_pos, target_rel_body_rot_obs, motion_bodies), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_observations_imitation_jpos(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, target_dof_pos, body_vel, body_ang_vel, motion_bodies, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    root_rot = remove_base_rot(root_rot)
    heading_rot, heading = torch_utils.calc_heading_quat_inv_with_heading(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    """target"""
    # target root height    [N, 1]
    target_root_pos = target_pos[:, 0, :]
    target_rel_root_h = root_h - target_root_pos[:, 2:3]
    # target 2d pos [N, 2]
    target_rel_pos = target_root_pos[:, :3] - root_pos[:, :3]
    target_rel_pos = torch_utils.my_quat_rotate(heading_rot, target_rel_pos)
    target_rel_2d_pos = target_rel_pos[:, :2]
    # target body pos   [N, 3xB]
    target_rel_body_pos = target_pos - body_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1], target_rel_body_pos.shape[2])
    flat_target_rel_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_target_rel_body_pos)
    target_rel_body_pos = flat_target_rel_body_pos.reshape(target_rel_body_pos.shape[0], target_rel_body_pos.shape[1] * target_rel_body_pos.shape[2])

    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, dof_vel,
                     target_rel_root_h, target_rel_2d_pos, target_rel_body_pos, motion_bodies), dim=-1)
    return obs
    
@torch.jit.script
def compute_humanoid_reward(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, target_dof_pos, target_dof_vel, body_vel, body_ang_vel, dof_obs_size, dof_offsets, body_pos_weights, reward_specs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, List[int], Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor, str]
    
    k_dof, k_vel, k_pos, k_rot = reward_specs['k_dof'], reward_specs['k_vel'], reward_specs['k_pos'], reward_specs['k_rot']
    w_dof, w_vel, w_pos, w_rot = reward_specs['w_dof'], reward_specs['w_vel'], reward_specs['w_pos'], reward_specs['w_rot']
    
    # dof rot reward
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    target_dof_obs = dof_to_obs(target_dof_pos, dof_obs_size, dof_offsets)
    diff_dof_obs = dof_obs - target_dof_obs
    diff_dof_obs_dist = (diff_dof_obs ** 2).mean(dim=-1)
    dof_reward = torch.exp(-k_dof * diff_dof_obs_dist)

    # velocity reward
    diff_dof_vel = target_dof_vel - dof_vel
    diff_dof_vel_dist = (diff_dof_vel ** 2).mean(dim=-1)
    vel_reward = torch.exp(-k_vel * diff_dof_vel_dist)

    # body pos reward
    diff_body_pos = target_pos - body_pos
    diff_body_pos = diff_body_pos * body_pos_weights[:, None]
    diff_body_pos_dist = (diff_body_pos ** 2).mean(dim=-1).mean(dim=-1)
    body_pos_reward = torch.exp(-k_pos * diff_body_pos_dist)

    # body rot reward
    diff_body_rot = quat_mul(target_rot, quat_conjugate(body_rot))
    diff_body_rot_angle = torch_utils.quat_to_angle_axis(diff_body_rot)[0]
    diff_body_rot_angle_dist = (diff_body_rot_angle ** 2).mean(dim=-1)
    body_rot_reward = torch.exp(-k_rot * diff_body_rot_angle_dist)

    # reward = dof_reward * vel_reward * body_pos_reward * body_rot_reward
    reward = w_dof * dof_reward + w_vel * vel_reward + w_pos * body_pos_reward + w_rot * body_rot_reward
    sub_rewards = torch.stack([dof_reward, vel_reward, body_pos_reward, body_rot_reward], dim=-1)
    sub_rewards_names = 'dof_reward,vel_reward,body_pos_reward,body_rot_reward'
    return reward, sub_rewards, sub_rewards_names

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, max_episode_length, enable_early_termination, termination_heights, cur_ref_motion_times, ref_motion_lengths):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    
    # enable_early_termination = False
    if (enable_early_termination):
        # masked_contact_buf = contact_buf.clone()
        # masked_contact_buf[:, contact_body_ids, :] = 0
        # fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        # fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        # has_fallen = torch.logical_and(fall_contact, fall_height)
        has_fallen = fall_height

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reach_max_length = progress_buf >= max_episode_length - 1
    reach_max_dur = cur_ref_motion_times >= ref_motion_lengths
    reset_cond = torch.logical_or(reach_max_length, reach_max_dur)
    reset = torch.where(reset_cond, torch.ones_like(reset_buf), terminated)
#     reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
