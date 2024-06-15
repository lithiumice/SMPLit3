import os
from os.path import join as pjoin
from easydict import EasyDict
from tqdm import tqdm
import pandas
import joblib


import sys; sys.path.append('/root/apdcephfs/private_wallyliang/PLANT/difftraj/text-to-motion')
from networks.modules import *
import utils.paramUtil as paramUtil
from options.train_options import TrainDecompOptions
from utils.plot_script import *
from networks.modules import *
from networks.trainers import DecompTrainerDisc
from data.dataset import StyleMotionDataset
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator

device = 'cuda'
dim_pose = 203


# ckpt_file_path = '/root/apdcephfs/private_wallyliang/PLANT/difftraj/text-to-motion/checkpoints/style100/Style_Decomp_SP001_SM001_H512_disc/model/latest.tar'
# movement_enc = MovementConvEncoder3(dim_pose, 256, 256)
# movement_dec = MovementConvDecoder3(256, 256, dim_pose)
# checkpoint = torch.load(ckpt_file_path,map_location=device)
# print(f'load from {ckpt_file_path}')
# movement_enc.load_state_dict(checkpoint['movement_enc'])
# movement_dec.load_state_dict(checkpoint['movement_dec'])
# movement_enc = movement_enc.to(device)
# movement_dec = movement_dec.to(device)


parser = TrainDecompOptions()
opt = parser.parse()

train_dataset = StyleMotionDataset(opt, 'train')
val_dataset = StyleMotionDataset(opt, 'test')    
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,shuffle=True, pin_memory=True)

num_epochs = 100
# style_predictor = StylePredictor().to(device)
style_predictor = StylePredictorFromRawPose(opt, dim_pose, 256, 256, latent_dim = 512, classes_num = 100).to(device)
optimizer = torch.optim.Adam(style_predictor.parameters(), lr=opt.lr)
criterion = nn.CrossEntropyLoss()    
# criterion = nn.MSELoss()    
from tqdm import tqdm
pbar = tqdm(range(num_epochs))
for epoch in pbar:
    for batch_idx, (motions, labels) in enumerate(train_loader):
        motions = motions.to(device).float()
        labels = labels.to(device).float()
        
        # with torch.no_grad():
        #     motion_embedding, _, _, _ = movement_enc(motions)
        # style_pred = style_predictor(motion_embedding)
        # loss = criterion(style_pred, labels)
        
        latent, disc_pred = style_predictor(motions)
        loss = criterion(disc_pred, labels)
        
        pred_style = torch.argmax(disc_pred, dim = 1).cpu()
        gt_style = torch.argmax(labels, dim = 1).cpu()
        tmp = (gt_style == pred_style)
        acc_radio = torch.nonzero(tmp).size(0) / torch.numel(tmp)
        print(f'acc_radio: {acc_radio}')
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % opt.log_interval == 0:
            log_str=(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
            pbar.set_description(log_str)
    pbar.update(1)
     
import ipdb;ipdb.set_trace()   
opt.save_root = '/root/apdcephfs/private_wallyliang/PLANT/text-to-motion/checkpoints/style100'    
save_pt_path = os.path.join(opt.save_root, 'style_predictor_from_motion.pt')
torch.save({'style_predictor': style_predictor.state_dict()}, save_pt_path)
print(f'save to {save_pt_path}')            

# # 使用torch.argmax()将one-hot编码转换为标签编码
# # one-hot is size (B,class_num)
# labels = torch.argmax(one_hot, dim=1)
# print(labels)