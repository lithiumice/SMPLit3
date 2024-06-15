
import os
from os.path import join as pjoin
from easydict import EasyDict
from tqdm import tqdm
import pandas
import joblib

os.chdir('/root/apdcephfs/private_wallyliang/PLANT/difftraj/text-to-motion')
import sys; sys.path.append('/root/apdcephfs/private_wallyliang/PLANT/difftraj/text-to-motion')
from networks.modules import MovementConvEncoder3,MovementConvDecoder3
from networks.modules import StylePredictor
import utils.paramUtil as paramUtil
from options.train_options import TrainDecompOptions
from utils.plot_script import *
from networks.modules import *
from networks.trainers import DecompTrainerDisc
from data.dataset import StyleMotionDataset
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator

title = 'FIttingDataAnd100StyleData'
ae_type = 'vae'
device = 'cuda'
dim_pose = 203
opt = EasyDict(window_size = 24, 
               batch_size = 4096, 
            #    batch_size = 40960, 
               log_interval = 3, 
               use_vae = True,
               in_type = 'debug',
               )
# opt.save_root = '/root/apdcephfs/private_wallyliang/PLANT/difftraj/text-to-motion/checkpoints/style100/Style_Decomp_SP001_SM001_H512_tripletLoss_stylePred_vae'
# ckpt_file_path = '/root/apdcephfs/private_wallyliang/PLANT/difftraj/text-to-motion/checkpoints/style100/Style_Decomp_SP001_SM001_H512_tripletLoss_stylePred_vae/model/latest.tar'
opt.save_root = '/root/apdcephfs/private_wallyliang/PLANT/difftraj/text-to-motion/checkpoints/style100/FitData_Style100_VAE_windowSize24'
ckpt_file_path = '/root/apdcephfs/private_wallyliang/PLANT/difftraj/text-to-motion/checkpoints/style100/FitData_Style100_VAE_windowSize24/model/latest.tar'
opt.in_type = 'style100_and_fitData'
opt.pred_style = False
opt.use_vae = True
train_dataset = StyleMotionDataset(opt, 'train')
val_dataset = StyleMotionDataset(opt, 'test')    
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,shuffle=True, pin_memory=True)

movement_enc = MovementConvEncoder3(opt, dim_pose, 256, 256)
movement_dec = MovementConvDecoder3(opt, 256, 256, dim_pose)
checkpoint = torch.load(ckpt_file_path,map_location=device)
print(f'load from {ckpt_file_path}')
movement_enc.load_state_dict(checkpoint['movement_enc'])
movement_dec.load_state_dict(checkpoint['movement_dec'])
movement_enc = movement_enc.to(device)
movement_dec = movement_dec.to(device)
    

# ae_type = 'ae'
# device = 'cuda'
# dim_pose = 203
# opt = EasyDict(window_size = 24, 
#                batch_size = 40960, 
#                use_vae = False,
#                in_type = 'debug',
#                log_interval = 3, 
#                )
# opt.save_root = '/root/apdcephfs/private_wallyliang/PLANT/difftraj/text-to-motion/checkpoints/style100/Style_Decomp_SP001_SM001_H512_disc'
# ckpt_file_path = '/root/apdcephfs/private_wallyliang/PLANT/difftraj/text-to-motion/checkpoints/style100/Style_Decomp_SP001_SM001_H512_disc/model_no_style_pred/latest.tar'
# train_dataset = StyleMotionDataset(opt, 'train')
# val_dataset = StyleMotionDataset(opt, 'test')    
# train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,shuffle=True, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,shuffle=True, pin_memory=True)

# from networks.modules import MovementConvEncoder3NoStylePred
# movement_enc = MovementConvEncoder3NoStylePred(opt, dim_pose, 256, 256)
# movement_dec = MovementConvDecoder3(opt, 256, 256, dim_pose)
# checkpoint = torch.load(ckpt_file_path,map_location=device)
# print(f'load from {ckpt_file_path}')
# movement_enc.load_state_dict(checkpoint['movement_enc'])
# movement_dec.load_state_dict(checkpoint['movement_dec'])
# movement_enc = movement_enc.to(device)
# movement_dec = movement_dec.to(device)



import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
# import ipdb;ipdb.set_trace()
latent_features = []
gt_styles = []
with torch.no_grad():
    for batch_idx, (data, gt_style) in enumerate(val_loader):
        if batch_idx > 3: break
        data = data.to(device)
        encoded, _, _, _ = movement_enc(data)
        latent_features.append(encoded.cpu().numpy())
        gt_styles.append(gt_style.numpy())

latent_features = np.vstack(latent_features)
gt_styles = np.vstack(gt_styles)
gt_style_labels = np.argmax(gt_styles, axis=1)

tsne = TSNE(n_components=2, random_state=42)
latent_features_2D = tsne.fit_transform(latent_features)

df = pd.DataFrame(latent_features_2D, columns=['Component 1', 'Component 2'])
df['GT_Style'] = gt_style_labels

plt.figure(figsize=(10, 8))
sns.scatterplot(x='Component 1', y='Component 2', hue='GT_Style', palette='viridis', data=df, legend="full", alpha=0.7)
plt.title(f'{title} t-SNE Visualization of {ae_type} Latent Space')
plt.savefig(f'{title}_{ae_type}_tsne_vis.png')



# <======use cluster
# def get_cluster_labels(data, n_clusters):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     cluster_labels = kmeans.fit_predict(data)
#     return cluster_labels
# cluster_labels = get_cluster_labels(latent_features, 30)
# df = pd.DataFrame(latent_features_2D, columns=['Component 1', 'Component 2'])
# df['Cluster'] = cluster_labels


# <======plot with matplotlib
# plt.figure()
# cmap = plt.cm.get_cmap('viridis', np.unique(cluster_labels).size)
# fig, ax = plt.subplots()
# sc = ax.scatter(latent_features_2D[:, 0], latent_features_2D[:, 1], c=cluster_labels, cmap=cmap, marker='o', alpha=0.5)
# norm = plt.Normalize(cluster_labels.min(), cluster_labels.max())
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# plt.colorbar(sm, ticks=np.unique(cluster_labels), label='Cluster ID', ax=ax)
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.title('t-SNE Visualization of AE Latent Space with Cluster Colors')
# plt.savefig('vae_tsne_vis_cluster_colors.png')



