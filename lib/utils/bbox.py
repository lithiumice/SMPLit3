import torch
import numpy as np
import cv2


def msk_to_xywh(msk):
    """
    calculate box [left upper width height] from mask.
    :param msk: nd.array, single-channel or 3-channels mask
    :return: float, iou score    
    """
    if len(msk.shape) > 2:
        msk = msk[..., 0]
    nonzeros = np.nonzero(msk.astype(np.uint8))
    u, l = np.min(nonzeros, axis=1)
    b, r = np.max(nonzeros, axis=1)
    return np.array((l, u, r-l+1, b-u+1))

def msk_to_xyxy(msk):
    """
    calculate box [left upper right bottom] from mask.
    :param msk: nd.array, single-channel or 3-channels mask
    :return: float, iou score    
    """
    if len(msk.shape) > 2:
        msk = msk[..., 0]
    nonzeros = np.nonzero(msk.astype(np.uint8))
    u, l = np.min(nonzeros, axis=1)
    b, r = np.max(nonzeros, axis=1)
    return np.array((l, u, r+1, b+1))

def get_edges(msk):
    """
    get edge from mask
    :param msk: nd.array, single-channel or 3-channel mask
    :return: edges: nd.array, edges with same shape with mask
    """
    msk_sp = msk.shape
    if len(msk_sp) == 2:
        c = 1 # single channel
    elif (len(msk_sp) == 3) and (msk_sp[2] == 3):
        c = 3 # 3 channels
        msk = msk[:, :, 0] != 0        
    edges = np.zeros(msk_sp[:2])
    edges[:-1, :] = np.logical_and(msk[:-1, :] != 0, msk[1:, :] == 0) + edges[:-1, :]
    edges[1:, :] = np.logical_and(msk[1:, :] != 0, msk[:-1, :] == 0) + edges[1:, :]
    edges[:, :-1] = np.logical_and(msk[:, :-1] != 0, msk[:, 1:] == 0) + edges[:, :-1]
    edges[:, 1:] = np.logical_and(msk[:, 1:] != 0, msk[:, :-1] == 0) + edges[:, 1:]
    if c == 3:
        return np.dstack((edges, edges, edges))
    else:
        return edges
    
    

def scale_bbox(bbox, h, w, scale = 1.8):
    # import numpy as np
    # bbox_x0,bbox_y0,bbox_x1,bbox_y1 = [int(ii) for ii in bbox]
    sw=(bbox[2] - bbox[0])/2
    sh=(bbox[3] - bbox[1])/2
    cy = (bbox[1] + bbox[3]) / 2
    cx = (bbox[0] + bbox[2]) / 2
    # scale_bbox = xys_to_xyxy(cx,cy,s, scale=scale)
    sw*=scale
    sh*=scale
    scale_bbox = [cx-sw,cy-sh,cx+sw,cy+sh]
    scale_bbox[0] = np.clip(scale_bbox[0], 0, w)
    scale_bbox[2] = np.clip(scale_bbox[2], 0, w)
    scale_bbox[1] = np.clip(scale_bbox[1], 0, h)
    scale_bbox[3] = np.clip(scale_bbox[3], 0, h)
    return scale_bbox

def bs_kpts_to_one_bbox(landmarks):
    left = np.min(landmarks[:, :, 0])
    right = np.max(landmarks[:, :, 0])
    top = np.min(landmarks[:, :, 1])
    bottom = np.max(landmarks[:, :, 1])
    return [left, top, right, bottom]
    
def kpts_to_bbox(landmarks):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])
    return [left, top, right, bottom]

def get_move_area(bbox, fw, fh):
    move_area_bbox = [
        bbox[:,0].min(),
        bbox[:,1].min(),
        bbox[:,2].max(),
        bbox[:,3].max(),
    ]
    
    if move_area_bbox[0]<0: move_area_bbox[0]=0
    if move_area_bbox[1]<0: move_area_bbox[1]=0
    if move_area_bbox[2]>fw: move_area_bbox[2]=fw
    if move_area_bbox[3]>fh: move_area_bbox[3]=fh
    return move_area_bbox
    