
import os
from os.path import join as pjoin
from easydict import EasyDict
from tqdm import tqdm
import pandas
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd


os.chdir('/root/apdcephfs/private_wallyliang/PLANT/text-to-motion')
import sys; sys.path.append('/root/apdcephfs/private_wallyliang/PLANT/text-to-motion')
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
from tqdm import tqdm

device = 'cuda'
dim_pose = 203
parser = TrainDecompOptions()
opt = parser.parse()
opt.ex_fps = 30
opt.window_size = 60
opt.in_type = 'whamFitData_100Styles'
ae_type = 'ae'

ckpt_file_path = '/root/apdcephfs/private_wallyliang/PLANT/text-to-motion/checkpoints/style100/style100_fitData_AE_FMD_win60_fps30/model/latest.tar'
movement_enc = MovementConvEncoder3(opt, dim_pose, 256, 256)
movement_dec = MovementConvDecoder3(opt, 256, 256, dim_pose)
checkpoint = torch.load(ckpt_file_path,map_location=device)
print(f'load from {ckpt_file_path}')
movement_enc.load_state_dict(checkpoint['movement_enc'])
movement_dec.load_state_dict(checkpoint['movement_dec'])
movement_enc = movement_enc.to(device)
movement_dec = movement_dec.to(device)


# train_dataset = StyleMotionDataset(opt, 'train')
val_dataset = StyleMotionDataset(opt, 'test')    
# train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,shuffle=True, pin_memory=True)

# import ipdb;ipdb.set_trace()
latent_features = []
gt_styles = []
with torch.no_grad():
    for batch_idx, (data, gt_style) in tqdm(enumerate(val_loader)):
        if batch_idx > 100: break
        data = data.to(device)
        encoded, _, _, _ = movement_enc(data)
        latent_features.append(encoded.cpu().numpy())
        gt_styles.append(gt_style.numpy())

latent_features = np.vstack(latent_features)
gt_styles = np.vstack(gt_styles)
gt_style_labels = np.argmax(gt_styles, axis=1)

tsne = TSNE(n_components=2, random_state=42)
latent_features_2D = tsne.fit_transform(latent_features)


style_onehot_save_path = '/root/apdcephfs/private_wallyliang/PLANT/difftraj/style_str_to_onehot.pt'
style_str_to_onehot = torch.load(style_onehot_save_path)  
sytle_onhot_to_str = {v: k for k, v in style_str_to_onehot.items()}
style_name_list = list(style_str_to_onehot.keys())
sytle_int_to_str = {v.argmax().item(): k for k, v in style_str_to_onehot.items()}
print(f'style_name_list: {style_name_list}')


df = pd.DataFrame(latent_features_2D, columns=['Component 1', 'Component 2'])
sytle_int_to_str[96] = 'Neutral&p-GT-Data'
df['GT_Style'] = [sytle_int_to_str[label] for label in gt_style_labels.tolist()]

# plt.figure(figsize=(10, 8))
# title = 'FIttingDataAnd100StyleData'
# sns.scatterplot(x='Component 1', y='Component 2', hue='GT_Style', palette='viridis', data=df, legend="full", alpha=0.7)
# plt.title(f'{title} t-SNE Visualization of {ae_type} Latent Space')
# plt.savefig(f'{title}_{ae_type}_tsne_vis.png')

plt.figure(figsize=(10, 8))
sns.scatterplot(x='Component 1', y='Component 2', hue='GT_Style', palette='viridis', data=df, legend="full", alpha=0.7)
# plt.title(f'{opt.in_type} t-SNE Visualization of {ae_type} Latent Space')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title='GT_Style', title_fontsize='medium')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title='GT_Style', title_fontsize='medium', ncol=3)
plt.savefig(f'100STYLES_and_pGT_tsne_vis.png', bbox_inches='tight')
# plt.savefig(f'{opt.in_type}_{ae_type}_tsne_vis.png', bbox_inches='tight')

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



