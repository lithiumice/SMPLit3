import torch
import glob
import os
import numpy as np
import torch
import json
import mmcv

# yapf:disable
import os
import sys
import os.path as osp
import matplotlib.pyplot as plt
from .op_base import op_base
from ..utils import smirnov_grubbs as grubbs
from ..utils.smirnov_grubbs import OutputType
# yapf:enable

# from SHOW.datasets.op_base import op_base
# from SHOW.utils import smirnov_grubbs as grubbs
# from SHOW.utils.smirnov_grubbs import OutputType
from loguru import logger

op=op_base()


def filter_abnormal_hands(kpts, alpha=0.10):
    
    err_list_42=process_raw(kpts)
    
    
    for idx_joint in range(67-25):
        
        err_list=[i['diff_frame'][idx_joint].item() for i in err_list_42]
        err_list=torch.tensor(err_list)
        
        valid_list=[i['valid_flag'][idx_joint].item() for i in err_list_42]
        valid_list=torch.tensor(valid_list)
        
        err_list=err_list[valid_list.bool()]
        
        valid_frame_map={}
        cur_pos=0
        for idx, after_bool in enumerate(valid_list):
            if after_bool:
                valid_frame_map[cur_pos]=idx
                cur_pos+=1
        
        print(f'valid_frame_map: {valid_frame_map}')
        
        err_list=err_list.numpy()
        # ret=filter_abnormal_hands_err_list(err_list, alpha=0.02)
        # ret=filter_abnormal_hands_err_list(err_list, alpha=0.10)
        ret=filter_abnormal_hands_err_list(err_list, alpha=alpha)
        
        # tmp=[valid_frame_map[idx_frame] for idx_frame in ret['outlier_idx']]
        
        for idx_frame in ret['outlier_idx']:
            
            idx_frame=valid_frame_map[idx_frame]
            
            kpts[idx_frame,25+idx_joint,2]=0
            
            if idx_frame+1<kpts.shape[0]:
                kpts[idx_frame+1,25+idx_joint,2]=0
                
            
    return dict(
        kpts=kpts,
        err_list=err_list,
        **ret
    )

    
def filter_abnormal_hands_err_list(err_list, alpha=0.05):
    
    outlier_idx=grubbs._max_test(
        err_list,alpha=alpha,
        output_type=OutputType.INDICES)
    
    outlier_val=[err_list[idx] for idx in outlier_idx]
    
    logger.info(f'err_list: {err_list}')
    logger.info(f'filtered idx: {outlier_idx} with value: {outlier_val}')

    return dict(
        outlier_idx=outlier_idx,
        outlier_val=outlier_val,
    )


def read_kpts(op_path):
    op_list=glob.glob(os.path.join(op_path,'*.json'))
    kpts_list=[]
    for op_file_path in op_list:# op_file_path=op_list[0]
        d=op_base.read_keypoints(op_file_path)
        kpts=d.keypoints_2d[0]
        kpts_list.append(kpts)
    kpts=np.stack(kpts_list,axis=0)
    kpts=torch.from_numpy(kpts).float()
    return kpts


def process_raw(kpts):
    # kpts: (bs,135,3)
    
    # process hands only
    kpts=kpts[:,25:67,:]
    conf=kpts[:,:,2]
    kpts=kpts[:,:,:2]

    # kpts_x=kpts[:,:,0][conf>0]
    # kpts_y=kpts[:,:,1][conf>0]
    # body_w=(kpts_x.max())-(kpts_x.min())
    # body_h=(kpts_y.max())-(kpts_y.min())
    # diff_threshold=40/body_w

    # temporary diff
    diff_kpts=kpts[1:,:,:]-kpts[:-1,:,:]
    diff_conf1=conf[1:,:]
    diff_conf2=conf[:-1,:]
    
    # MSE criteria
    diff_kpts=torch.sqrt(
        torch.square(diff_kpts[:,:,0])+
        torch.square(diff_kpts[:,:,1])
    )

    mean_diff_list=[]
    for idx in range(diff_kpts.shape[0]):#idx=13
        
        diff_frame=diff_kpts[idx,:]
        frame_valid1=diff_conf1[idx,:]>0
        frame_valid2=diff_conf2[idx,:]>0
        valid_flag=torch.logical_and(
            frame_valid1,frame_valid2)
        
        no_valid_flag=torch.logical_not(valid_flag)
        diff_frame[no_valid_flag]=0
        
        mean_diff_list.append(
            dict(
                diff_frame=diff_frame,
                valid_flag=valid_flag,
            )
        )
        
    # mean_diff_list：list of length bs, err val
    return mean_diff_list
        

def three_xigma_criteria(err_list):
    assert isinstance(err_list,np.ndarray), 'must be numpy array'
    
    def cal_std_mean(err_list):
        mean=np.mean(err_list)
        std=np.std(err_list)
        return mean,std
    
    err_list_bak=err_list.copy()
    delete_item_idx=[]
    while True:
        has_abnormal_flag=False
        mean,std=cal_std_mean(err_list)
        
        # print(mean,std)
        relative_values = abs(err_list - mean)
        index = relative_values.argmax()
        value = relative_values[index]

        if (
            value<mean-3*std or
            value>mean+3*std
        ):
            delete_item_idx.append(index)
            
            # err_list[idx]=0
            np.delete(err_list, index)
            
            print(index,value)
            
            has_abnormal_flag=True
            break
        
        # for idx,item in enumerate(err_list.tolist()):
        #     if (
        #         item<mean-3*std or
        #         item>mean+3*std
        #     ):
        #         delete_item_idx.append(idx)
                
        #         # err_list[idx]=0
        #         np.delete(err_list, idx)
                
        #         print(idx,item)
        #         has_abnormal_flag=True
        #         break
                
        if not has_abnormal_flag:
            break
        
    return dict(
        delete_item_idx=delete_item_idx,
    )
            
if __name__ == '__main__':
    op_path=r'C:\Users\lithiumice\Desktop\op'
    op_path=r'C:\Users\lithiumice\code\speech2gesture_dataset\crop\chemistry\ATP_and_Metabolism-ma-MKQ2TAGk.mp4\67518-00_02_15-00_02_18\op'
    img_path=r'C:\Users\lithiumice\code\speech2gesture_dataset\crop\chemistry\ATP_and_Metabolism-ma-MKQ2TAGk.mp4\67518-00_02_15-00_02_18\image'
    kpts=read_kpts(op_path)
    filter_abnormal_hands(kpts)
    # plt.plot(mean_diff_list)
