# exp_dir_dict: key: exp dir path, value: exp alias name
exp_dir_dict = {
    '/apdcephfs/share_1330077/wallyliang/ar_diff_loco_gen_save/pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_100Styles_30fps_finetuneMDM_fitModel50000': 
        'finetune MDM, lr 1e-4, base 50000 steps',
    '/apdcephfs/share_1330077/wallyliang/ar_diff_loco_gen_save/pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_100Styles_30fps_finetuneMDM':
        'finetune MDM, lr 1e-4',
    '/apdcephfs/share_1330077/wallyliang/ar_diff_loco_gen_save/pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_cmdm_100Styles_30fps_finetune2_lr5':
        'finetune MDM&CMDM, lr 1e-5',
    '/apdcephfs/share_1330077/wallyliang/ar_diff_loco_gen_save/pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_cmdm_100Styles_30fps_finetune2_lr5_basemodel000601825':
        'finetune MDM&CMDM, lr 1e-5, retrain',
    '/apdcephfs/share_1330077/wallyliang/ar_diff_loco_gen_save/pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_cmdm_100Styles_30fps_finetune2':
        'finetune MDM&CMDM, lr 1e-4',
    '/apdcephfs/share_1330077/wallyliang/ar_diff_loco_gen_save/pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_100Styles_30fps':
        'baseline: direct train on 100 style with MDM',
    '/apdcephfs/share_1330077/wallyliang/ar_diff_loco_gen_save/pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_cmdm_100Styles_30fps_mdm_cmdm_ablation':
        'ablation: direct train on 100 style with MDM+CMDM',
        
}

import os
import re
import matplotlib.pyplot as plt
import json


# 用于保存结果的字典
results = {}

# 遍历所有实验目录
for exp_dir, exp_alias in exp_dir_dict.items():
    # 在每个实验目录下查找符合条件的日志文件
    for filename in os.listdir(exp_dir):
        if filename.startswith('eval') and filename.endswith('_debug.json'):
        # if filename.startswith('eval') and filename.endswith('_debug.log'):
        # if filename.startswith('eval') and filename.endswith('.log'):
            print(f'[INFO] find {filename} for {exp_alias}')
            # 提取训练轮次
            epoch = int(re.search(r'\d{9}', filename).group(0))
            
            if epoch == 0:
                continue
            
            # 读取日志文件
            with open(os.path.join(exp_dir, filename), 'r') as f:
                content = f.read()
                if len(content)==0: continue
                # import ipdb;ipdb.set_trace()
                content = json.loads(content)
                print(content)
                fid_mean = content['mean_dict']['fid_vald']
                diversity_mean = content['mean_dict']['diversity_vald']
                fid_cinterval = content['conf_dict']['fid_vald']
                diversity_cinterval = content['conf_dict']['diversity_vald']
                
                if exp_alias not in results:
                    results[exp_alias] = {}
                results[exp_alias][epoch] = {'fid_mean': fid_mean, 'fid_cinterval': fid_cinterval, 'diversity_mean': diversity_mean, 'diversity_cinterval': diversity_cinterval}            

# det_min_stop_epoch = 1e10
det_min_stop_epoch = 5e5

# 绘制结果
plt.figure(figsize=(15, 4))
for exp_alias, data in results.items():
    epochs = sorted(data.keys())
    # fids = [data[epoch]['fid'] for epoch in epochs if epoch<=det_min_stop_epoch]
    # diversities = [data[epoch]['diversity'] for epoch in epochs if epoch<=det_min_stop_epoch]
    fid_means = [data[epoch]['fid_mean'] for epoch in epochs if epoch<=det_min_stop_epoch]
    fid_cintervals = [data[epoch]['fid_cinterval'] for epoch in epochs if epoch<=det_min_stop_epoch]
    diversity_means = [data[epoch]['diversity_mean'] for epoch in epochs if epoch<=det_min_stop_epoch]
    diversity_cintervals = [data[epoch]['diversity_cinterval'] for epoch in epochs if epoch<=det_min_stop_epoch]
    epochs = [epoch for epoch in epochs if epoch<=det_min_stop_epoch]
    
    fids = fid_means
    diversities = diversity_means
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, fid_means, label=exp_alias)
    plt.fill_between(epochs, [mean - cinterval for mean, cinterval in zip(fid_means, fid_cintervals)], [mean + cinterval for mean, cinterval in zip(fid_means, fid_cintervals)], alpha=0.1)
    plt.xlabel('Train Steps')
    plt.ylabel('FID')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, diversity_means, label=exp_alias)
    plt.fill_between(epochs, [mean - cinterval for mean, cinterval in zip(diversity_means, diversity_cintervals)], [mean + cinterval for mean, cinterval in zip(diversity_means, diversity_cintervals)], alpha=0.1)
    plt.xlabel('Train Steps') 
    plt.ylabel('Diversity')
    min_fid = min(fids)
    min_dev_resp_fid = diversities[fids.index(min_fid)]
    print(f'{exp_alias}: min FID {min_fid} DIV {min_dev_resp_fid}')
             
                   
plt.legend()
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title='Experiments', title_fontsize='medium')
save_fig_path = f'pretrain_eval_compare_epoch{det_min_stop_epoch}.png'
plt.savefig(save_fig_path)
print(f'save to {save_fig_path}')
