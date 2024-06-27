import cv2
import torch
import numpy as np

from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer.cameras import look_at_rotation
from pytorch3d.ops import interpolate_face_attributes


from .tools import get_colors, checkerboard_geometry
from lib.loco.trajdiff import *


import pickle


from matplotlib import cm

color_map = cm.get_cmap("jet", 55)

from easydict import EasyDict


def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]

    roi_image[mask] = image[mask]
    out_image[bbox[1] : bbox[3], bbox[0] : bbox[2]] = roi_image

    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype

    K = torch.zeros((K_org.shape[0], 4, 4)).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1

    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)

        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        # new_cx = new_width/2
        # new_cy = new_height

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = right - left
    height = bottom - top

    new_left = torch.clamp(cx - width / 2 * scaleFactor, min=0, max=img_w - 1)
    new_right = torch.clamp(cx + width / 2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h - 1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = (
        torch.stack(
            (
                new_left.detach(),
                new_top.detach(),
                new_right.detach(),
                new_bottom.detach(),
            )
        )
        .int()
        .float()
        .T
    )

    return bbox


class DepthShader(torch.nn.Module):
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        # import ipdb;ipdb.set_trace()
        return fragments.zbuf


class NormalShader(torch.nn.Module):
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        # import ipdb;ipdb.set_trace()
        vertex_normals = meshes.verts_normals_packed()  # V,3; V=10475
        faces = meshes.faces_packed()  # F,3; F=20908
        faces_normals = vertex_normals[faces]  # F,3,3
        ones = torch.ones_like(fragments.bary_coords)
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, ones, faces_normals
        )  # 1,H,W,1,3
        return pixel_normals

        # # 对顶点法线进行插值
        # # fragments.pix_to_face.shape: 1, H, W, 1
        # pixel_normals = fragments.pix_to_face.unsqueeze(-1) * vertex_normals[fragments.faces_verts_idx]

        # # 计算每个像素的法线
        # normal_map = pixel_normals.sum(dim=-2)

        # # 将法线转换为颜色表示（0-1范围）
        # # normal_map = (normal_map + 1.0) / 2.0

        # return normal_map


class Renderer:
    def __init__(self, width, height, focal_length, device, faces=None):

        self.width = width
        self.height = height
        self.focal_length = focal_length

        self.device = device
        if faces is not None:
            self.faces = (
                torch.from_numpy((faces).astype("int")).unsqueeze(0).to(self.device)
            )

        self.initialize_camera_params()
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer()

    def create_renderer(self):

        # import ipdb;ipdb.set_trace()
        rasterizer = MeshRasterizer(
            raster_settings=RasterizationSettings(
                image_size=self.image_sizes[0], blur_radius=1e-5
            ),
        )
        self.rasterizer = rasterizer

        # 使用自定义的shader创建一个渲染器
        self.depth_shader = DepthShader()
        self.depth_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=self.depth_shader,
        )

        # 创建法线图渲染器
        self.normal_shader = NormalShader()
        self.normal_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=self.normal_shader,
        )

        self.soft_phong_shader = SoftPhongShader(
            device=self.device,
            lights=self.lights,
        )
        self.renderer = MeshRenderer(
            rasterizer=rasterizer, shader=self.soft_phong_shader
        )

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        # import ipdb;ipdb.set_trace()
        return PerspectiveCameras(
            device=self.device,
            R=self.R.mT,
            T=self.T,
            K=self.K_full,
            image_size=self.image_sizes,
            in_ndc=False,
        )

    def initialize_camera_params(self):
        """Hard coding for camera parameters
        TODO: Do some soft coding"""

        # Extrinsics
        self.R = (
            torch.diag(torch.tensor([1, 1, 1])).float().to(self.device).unsqueeze(0)
        )

        self.T = torch.tensor([0, 0, 0]).unsqueeze(0).float().to(self.device)

        # Intrinsics
        self.K = (
            torch.tensor(
                [
                    [self.focal_length, 0, self.width / 2],
                    [0, self.focal_length, self.height / 2],
                    [0, 0, 1],
                ]
            )
            .unsqueeze(0)
            .float()
            .to(self.device)
        )
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()

    def update_K(self, focal_length, vertices):
        self.K = (
            torch.tensor(
                [
                    [focal_length, 0, self.width / 2],
                    [0, focal_length, self.height / 2],
                    [0, 0, 1],
                ]
            )
            .unsqueeze(0)
            .float()
            .to(self.device)
        )
        self.update_bbox(vertices[::50], scale=1.2)

    def set_ground(self, length, center_x, center_z):
        device = self.device
        v, f, vc, fc = map(
            torch.from_numpy,
            checkerboard_geometry(length=length, c1=center_x, c2=center_z, up="y"),
        )
        v, f, vc = v.to(device), f.to(device), vc.to(device)
        self.ground_geometry = [v, f, vc]

    def update_bbox(self, x3d, scale=2.0, mask=None):
        """Update bbox of cameras from the given 3d points

        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """
        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(
                x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1)
            )

        if mask is not None:
            x2d = x2d[:, ~mask]

        # import ipdb;ipdb.set_trace()
        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(
        self,
    ):
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def render_mesh(self, vertices, background, colors=[0.8, 0.8, 0.8]):
        self.update_bbox(vertices[::50], scale=1.2)
        vertices = vertices.unsqueeze(0)
        # import ipdb;ipdb.set_trace()

        if False:
            if not hasattr(self, "verts_features"):
                segm = "model_files/data_ipman/essentials/models_utils/smplx/smplx_parts_segm.pkl"
                with open(segm, "rb") as file:
                    segm_data = pickle.load(file, encoding="latin1")

                verts_features = torch.zeros((1, vertices.shape[1], 3)).type_as(
                    vertices
                )
                tmp = toth(segm_data["segm"])
                # part_color_map=[color_map(ii)[:3] for ii in range(tmp.min().int(), tmp.max().int())]

                part_color_map = []
                for ii in range(tmp.min().int(), tmp.max().int() + 1):
                    part_color_map += [color_map(ii)[:3]]
                part_color_map = torch.tensor(part_color_map).type_as(vertices)
                for idx in range(3):
                    verts_features[0, self.faces[0, :, idx], :] = part_color_map[
                        tmp.long()
                    ]
                self.verts_features = verts_features
        else:
            if colors[0] > 1:
                colors = [c / 255.0 for c in colors]
            verts_features = (
                torch.tensor(colors)
                .reshape(1, 1, 3)
                .to(device=vertices.device, dtype=vertices.dtype)
            )
            verts_features = verts_features.repeat(1, vertices.shape[1], 1)
            self.verts_features = verts_features

        textures = TexturesVertex(verts_features=self.verts_features)

        mesh = Meshes(
            verts=vertices,
            faces=self.faces,
            textures=textures,
        )

        materials = Materials(device=self.device, specular_color=(colors,), shininess=0)

        # if 1:

        # normal_img = self.normal_renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights)
        # depth_img = self.depth_renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights)

        fragment = self.rasterizer(mesh, cameras=self.cameras)

        if False:
            # get depth image
            depth_img = self.depth_shader(fragment, mesh)
            depth_img = torch.flip(depth_img, [1, 2])
            depth_img = depth_img.squeeze()

            # get visiable mask
            vis_mask = depth_img > 0.0

            # get normal map
            normal_img = self.normal_shader(fragment, mesh)
            normal_img = torch.flip(normal_img, [1, 2])
            normal_img = normal_img.squeeze()

            depth_img = (depth_img - depth_img[vis_mask].min()) / (
                depth_img[vis_mask].max() - depth_img[vis_mask].min()
            )
            normal_img = (normal_img - normal_img[vis_mask].min()) / (
                normal_img[vis_mask].max() - normal_img[vis_mask].min()
            )

            depth_img = 1 - depth_img

            depth_img[~vis_mask] = 0
            normal_img[~vis_mask] = 0

            vis_mask = vis_mask * 255.0
            depth_img = depth_img * 255.0
            normal_img = normal_img * 255.0

            # normal_img = (normal_img+1.0)/2.0
            # depth_img = (depth_img+1.0)/2.0
            # import cv2; cv2.imwrite('t.png', tonp(depth_img)*255)
            # import cv2; cv2.imwrite('t.png', tonp(normal_img[:,:,:])*255)

        # get RGB image
        # rgb_img = self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights)
        rgb_img = self.soft_phong_shader(
            fragment,
            mesh,
            materials=materials,
            cameras=self.cameras,
            lights=self.lights,
        )
        results = torch.flip(rgb_img, [1, 2])
        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3

        image = overlay_image_onto_background(
            image, mask, self.bboxes, background.copy()
        )

        if False:
            # import ipdb;ipdb.set_trace()
            depth_img = overlay_image_onto_background(
                depth_img.unsqueeze(-1).repeat(1, 1, 3),
                mask,
                self.bboxes,
                np.zeros_like(background),
            )
            normal_img = overlay_image_onto_background(
                normal_img, mask, self.bboxes, np.zeros_like(background)
            )

        self.reset_bbox()
        return EasyDict(
            image=image,
            # depth_img=depth_img,
            # normal_img=normal_img
        )

    def render_with_ground(self, verts, faces, colors, cameras, lights):
        """
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (B, 3)
        """

        # (B, V, 3), (B, F, 3), (B, V, 3)
        verts, faces, colors = prep_shared_geometry(verts, faces, colors)
        # (V, 3), (F, 3), (V, 3)
        gv, gf, gc = self.ground_geometry
        verts = list(torch.unbind(verts, dim=0)) + [gv]
        faces = list(torch.unbind(faces, dim=0)) + [gf]
        colors = list(torch.unbind(colors, dim=0)) + [gc[..., :3]]
        mesh = create_meshes(verts, faces, colors)

        materials = Materials(device=self.device, shininess=0)

        results = self.renderer(
            mesh, cameras=cameras, lights=lights, materials=materials
        )
        image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

        return image


def prep_shared_geometry(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (F, 3)
    :param colors (B, 4)
    """
    B, V, _ = verts.shape
    F, _ = faces.shape
    colors = colors.unsqueeze(1).expand(B, V, -1)[..., :3]
    faces = faces.unsqueeze(0).expand(B, F, -1)
    return verts, faces, colors


def create_meshes(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (B, F, 3)
    :param colors (B, V, 3)
    """
    textures = TexturesVertex(verts_features=colors)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return join_meshes_as_scene(meshes)


def get_global_cameras(verts, device, distance=5, position=(-5.0, 5.0, 0.0)):
    positions = torch.tensor([position]).repeat(len(verts), 1)
    targets = verts.mean(1)

    directions = targets - positions
    directions = directions / torch.norm(directions, dim=-1).unsqueeze(-1) * distance
    positions = targets - directions

    rotation = look_at_rotation(
        positions,
        targets,
    ).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)

    lights = PointLights(device=device, location=[position])
    return rotation, translation, lights


def _get_global_cameras(verts, device, min_distance=3, chunk_size=100):

    # split into smaller chunks to visualize
    start_idxs = list(range(0, len(verts), chunk_size))
    end_idxs = [min(start_idx + chunk_size, len(verts)) for start_idx in start_idxs]

    Rs, Ts = [], []
    for start_idx, end_idx in zip(start_idxs, end_idxs):
        vert = verts[start_idx:end_idx].clone()
        import pdb

        pdb.set_trace()
