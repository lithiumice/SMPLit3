from functools import partial, reduce
from loguru import logger
import platform
import os
import torch
import numpy as np
import re
import sys


class path_enter(object):
    def __init__(self, target_path=None):
        self.origin_path = None
        self.target_path = target_path

    def __enter__(self):
        if sys.path[0] != self.target_path:
            sys.path.insert(0, self.target_path)

        if self.target_path:
            self.origin_path = os.getcwd()
            os.chdir(self.target_path)
            logger.info(f"entered: {self.target_path}; origin_path: {self.origin_path}")

    def __exit__(self, exc_type, exc_value, trace):
        if self.origin_path:
            os.chdir(self.origin_path)
            logger.info(f"exit to origin_path: {self.origin_path}")


def match_faces(img, face_ider, person_face_emb):
    # img: bgr,hw3,uint8
    faces = face_ider.get(img)
    if faces is None:
        return None, None
    # face_ider: 1.func:get(np_img) --> {2.normed_embedding,3.bbox}
    for face in faces:
        cur_emb = face.normed_embedding
        sim = face_ider.cal_emb_sim(cur_emb, person_face_emb)
        if sim >= face_ider.threshold:
            logger.info(f"found sim:{sim}")
            correspond_bbox = face.bbox
            xmin, ymin, xmax, ymax = correspond_bbox
            correspond_center = [int((xmin + xmax) / 2), int((ymin + ymax) / 2)]
            return correspond_center, correspond_bbox
        logger.info(f"not found: {sim}")
    return None, None


"""
在python中a为一个变量, a中还有子类, 写一个函数历遍a及其子类中的所有成员, 
如果成员为字符串则将其字符串中的"{{ fileDirname }}"替换为当前文件的目录路径, 并保存到原变量, 不要打印出来
"""
import inspect
import os


def replace_file_dirname(obj):
    if isinstance(obj, str):
        print("a")
        obj = obj.replace("{{ fileDirname }}", os.path.dirname(__file__))
    elif inspect.isclass(obj):
        for name, member in inspect.getmembers(obj):
            setattr(obj, name, replace_file_dirname(member))
    elif inspect.ismodule(obj):
        for name, member in inspect.getmembers(obj):
            setattr(obj, name, replace_file_dirname(member))
    elif inspect.isroutine(obj):
        for name, member in inspect.getmembers(obj):
            setattr(obj, name, replace_file_dirname(member))
    return obj


def replace_spec_code(in_name):

    disp_name = re.sub(
        re.compile(
            r"[^a-zA-Z0-9]"
            # r'[-,$()#+&*]'
        ),
        "_",
        in_name,
    )

    return disp_name


def cvt_dict_to_tensor(data, device, dtype):
    if isinstance(data, list):
        return [cvt_dict_to_tensor(v, device, dtype) for v in data]
    elif isinstance(data, dict):
        return {
            k: (cvt_dict_to_tensor(v, device, dtype) if k != "seg_stack" else v)
            for k, v in data.items()
        }
    else:
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device=device, dtype=dtype)
        elif isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=dtype)
        else:
            raise ValueError(f"not support type {type(data)}")


def expand_var_shape(var, target_len=300, expand_axis=-1):
    if isinstance(var, np.ndarray):
        org_len = var.shape[expand_axis]
        if target_len == org_len:
            return
        oth_len = var.shape[:expand_axis]
        new_var = np.concatenate(
            [var, np.zeros(*oth_len, target_len - org_len)], axis=expand_axis
        )
        return new_var
    if isinstance(var, torch.Tensor):
        org_len = var.shape[expand_axis]
        if target_len == org_len:
            return
        oth_len = var.shape[:expand_axis]
        new_var = torch.cat(
            [var, torch.zeros(*oth_len, target_len - org_len)], axis=expand_axis
        )
        return new_var


def str_to_torch_dtype(s):
    dtype = torch.float32
    if s == "float64":
        dtype = torch.float64
    elif s == "float32":
        dtype = torch.float32
    return dtype


def reload_module(s: str = "SHOW.smplx_dataset"):
    import imp

    eval(f"import {s} as target")
    imp.reload(locals()["target"])


def print_args(args: dict):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


def print_dict_losses(losses):
    return reduce(
        lambda a, b: a + f" {b}={round(losses[b].item(), 4)}",
        [""] + list(losses.keys()),
    )


def platform_init():
    import platform

    # if platform.system() == "Linux":
    #     os.environ['PYOPENGL_PLATFORM'] = 'egl'
    # else:
    #     if 'PYOPENGL_PLATFORM' in os.environ:
    #         os.environ.__delitem__('PYOPENGL_PLATFORM')


def work_seek_init(rank=42):
    import torch
    import torch.backends.cudnn
    import numpy as np

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(rank)

    torch.backends.cudnn.enabled = False
