import os
import os.path as osp

import cv2
import torch
import imageio
import numpy as np
from progress.bar import Bar
import sys
from lib.vis.renderer import Renderer, get_global_cameras
from lib.loco.trajdiff import *
     
from lib.vis.tools import vis_keypoints
from tqdm import tqdm
import subprocess
import ffmpeg
from pathlib import Path

def run_ffmpeg_subprocess(ffmpeg_cmd):
    if isinstance(ffmpeg_cmd, list):
        ffmpeg_cmd=' '.join(ffmpeg_cmd)
    print(ffmpeg_cmd)
    ret = subprocess.run(ffmpeg_cmd, shell=True)
    return ret
            
def get_ffmpeg_args(use_nvdec=True):
    if use_nvdec:
        ffmpeg_bin_path = '/is/cluster/fast/hyi/workspace/VidGen/ffmpeg/local/bin/ffmpeg'
        mp4_codec_name = 'h264_nvenc'
        ffmpeg_cmd_prefix = '-hwaccel cuda -hwaccel_output_format cuda'
    else:
        ffmpeg_bin_path = '/usr/bin/ffmpeg'
        mp4_codec_name = 'libx264'
        ffmpeg_cmd_prefix = ''
    return ffmpeg_bin_path, mp4_codec_name, ffmpeg_cmd_prefix


def save_crop_vid(
    move_area_bbox, 
    audiostart, 
    audioend, 
    videofile, 
    save_vid_path, 
    use_nvdec=True,
    resample_fps=-1,
):
    Path(save_vid_path).parent.mkdir(exist_ok=True)
    ffmpeg_bin_path, mp4_codec_name, ffmpeg_cmd_prefix = get_ffmpeg_args(use_nvdec=use_nvdec)
    
    ffmpeg_cmd = [
        ffmpeg_bin_path,
        # "-hwaccel nvdec" if opt.nvdec,
        ffmpeg_cmd_prefix,
        "-y",
        "-loglevel", "error",
        "-ss", str(audiostart),
        "-t", str(audioend-audiostart),
        "-c:v h264_cuvid" if use_nvdec else "",
        f"-r {resample_fps}" if resample_fps!=-1 else "",
        "-i", str(videofile),
    ]   
    
    if move_area_bbox is not None:
        # move_area_bbox: [x0 y0 x1 y1]
        x0 = int(move_area_bbox[0])
        y0 = int(move_area_bbox[1])
        hh = int(move_area_bbox[3]) - y0
        ww = int(move_area_bbox[2]) - x0    
        ffmpeg_cmd += [ 
            f'-vf "hwdownload,format=nv12,crop={ww}:{hh}:{x0}:{y0},hwupload"' 
            if use_nvdec else f'-vf "crop={ww}:{hh}:{x0}:{y0}"', # "-filter:v", f"crop={ww}:{hh}:{x0}:{y0}",
        ]
    # -an -vf "select=between(n\,{audiostart}\,{audioend}),setpts=PTS-STARTPTS" 
    ffmpeg_cmd += [
        "-c:a", "copy",
        "-c:v", mp4_codec_name,
        # "-qscale:v", "1",
        # "-qmin", "1",
        # "-qmax", "1",
        "-vsync", "0",
        str(save_vid_path)
    ]
    run_ffmpeg_subprocess(ffmpeg_cmd)
    print(f'save to {save_vid_path}')
    
    
    
    
def save_kpts2d(image_points, video_file, fps, virtual_im_w, virtual_im_h, dataset="TopDownCocoDataset",bg_gray=255):
    image_list = []
    keypts_list = image_points.clone()
    face_kpts = keypts_list[:,23:23+68]
    # keypts_list = torch.cat([image_points[:,:,:2],torch.ones_like(image_points[:,:,:1])], dim=-1)
    for idx in tqdm(range(keypts_list.shape[0]), desc = 'render 2d kpts visualization'):  
        one_frame_kpt = tonp(keypts_list)[idx:idx+1]
        render_im = vis_keypoints(
            one_frame_kpt,
            (virtual_im_w, virtual_im_h),
            radius=6,
            thickness=3,
            kpt_score_thr=0.3,
            dataset=dataset,
            bg_gray=bg_gray
        )
        # import ipdb;ipdb.set_trace()
        for pt in face_kpts[idx]:
            if pt[2]>0.5:
                cv2.circle(render_im, (int(pt[0]), int(pt[1])), 4, (255,0,0), -1)
        # cv2.imwrite('render_im.png',render_im)
        image_list.append(render_im)
    imageio.mimwrite(video_file, image_list, fps=fps)        

def get_wav_full(audiostart, audioend, total_wav_path, part_wav_path):
    command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 -ss %.3f -to %.3f %s -loglevel error" %(str(total_wav_path), audiostart, audioend, part_wav_path))
    print(f'command: {command}')
    output = subprocess.call(command, shell=True, stdout=None)
    return output

def run_vis_on_demo(
    cfg, video, results, output_pth, smpl, faces, 
    vis_global=True, 
    stride = 1, 
    save_prefix='',
    save_ex=False,
    must_redump=False,
    render_result=True,
    args=None,
):
    toth = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
    # to torch tensor
    print(f'results.keys(): {results.keys()}')
    for sid in results.keys():
        print(f'rendering sid: {sid}')
        val = results[sid]
        
        fname = f'ID{sid}'
        if len(save_prefix)>0:
            fname = f'{save_prefix}_{fname}'
                    
        merge_bbox_xyxy=results[sid]['merge_bbox_xyxy']
                    
        audiostart = results[sid]['frame_ids'][0]*stride/fps
        audioend = results[sid]['frame_ids'][-1]*stride/fps
        
        if not Path(part_wav_path := osp.join(output_pth, f'{fname}_raw.wav')):
            get_wav_full(audiostart, audioend, video, part_wav_path)
        
        
        # raw_split_vid_save_path = osp.join(output_pth, f'{fname}_raw.mp4')
        # if not Path(raw_split_vid_save_path).exists() or must_redump:
        #     save_crop_vid(
        #         merge_bbox_xyxy, 
        #         audiostart, audioend, 
        #         video, raw_split_vid_save_path, 
        #         use_nvdec=False,
        #         resample_fps=30,
        #     )
            
        if not Path(origin_fps_raw_split_vid_save_path:=osp.join(
                output_pth, f'{fname}_raw_origin_fps.mp4')).exists() or must_redump:
            save_crop_vid(
                merge_bbox_xyxy, 
                audiostart, 
                audioend, 
                video, 
                origin_fps_raw_split_vid_save_path, 
                use_nvdec=False,
            )
            
        # test syncnet
        # import ipdb;ipdb.set_trace()
        try:
            save_path = str(origin_fps_raw_split_vid_save_path)[:-4]+'.sync_track'
            if not Path(save_path).exists():
                # if 'sync_agent' not in globals():
                DATA_PROCESS_CODE='/is/cluster/fast/hyi/workspace/VidGen/talking_avatar_data'
                if DATA_PROCESS_CODE not in sys.path:
                    sys.path.append(DATA_PROCESS_CODE)
                    
                from main_syncdet import sync_agent
                
                Ag = sync_agent(
                    origin_fps_raw_split_vid_save_path,
                    Path(output_pth, 'sync_work'),
                    device=f'cuda',
                )
                _ = Ag.get_sync_track()
                Ag.save_sync_track()
                Ag.clean()
        except:
            # import traceback
            # traceback.print_exc()
            print(f'run sync detect failed...')
                    
        left, top, right, bottom = list(map(int,merge_bbox_xyxy[:4]))   
        def crop_im(im): return im[top:bottom,left:right,:]        
        # <=====
        
        
        if not render_result:
            print(f'{render_result=}, continue')
            continue
        
        if Path(save_path:=osp.join(output_pth, f'{fname}_vis.mp4')).exists():
            if args.skip_save_vis_if_exists:
                print(f'{save_path=} exists..')
                continue
                        
        # create renderer with cliff focal length estimation
        cliff_focal_length = (width ** 2 + height ** 2) ** 0.5
        renderer = Renderer(width, height, cliff_focal_length, cfg.DEVICE, tonp(faces))
        
        if vis_global:
            # setup global coordinate subject
            # current implementation only visualize the subject appeared longest
            # n_frames = {k: len(results[k]['frame_ids']) for k in results.keys()}
            # sid = max(n_frames, key=n_frames.get)
            
            params = results[sid]['init_param_th']
            global_orient=toth(results[sid]['difftraj_root'])
            transl=toth(results[sid]['difftraj_trans'])
            cvt = a2m(torch.tensor([[0, np.pi, 0]])).float().to(global_orient.device)
            global_orient = m2a(cvt.mT @ a2m(global_orient))            
            # import ipdb;ipdb.set_trace()
            transl = (cvt.mT @ (transl.unsqueeze(-1))).squeeze(-1)           
            global_output = smpl.get_output(
                body_pose=toth(results[sid]['pose_world'][:, 3:66]), 
                global_orient=global_orient,                
                transl=transl,
                betas=toth(results[sid]['betas']),
                left_hand_pose=s2a(toth(params['hand_pose'])[:,:,0,:,:])[0],
                right_hand_pose=s2a(toth(params['hand_pose'])[:,:,1,:,:])[0],
                expression=toth(params['exp'])[0],
                jaw_pose=s2a(toth(params['jaw']))[0],
                leye_pose=s2a(toth(params['leye']))[0],
                reye_pose=s2a(toth(params['reye']))[0],
            )
                        
            verts_glob = global_output.vertices.cpu()
            # verts_glob[:,:,2]*=-1
            # verts_glob[...,0]*=-1
            verts_glob[..., 1] = verts_glob[..., 1] - verts_glob[..., 1].min()
            cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
            sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
            scale = max(sx.item(), sz.item()) * 20.0
            # scale = max(sx.item(), sz.item()) * 1.5
            
            # set default ground
            renderer.set_ground(scale, cx.item(), cz.item())
            
            # build global camera
            global_R, global_T, global_lights = get_global_cameras(verts_glob, cfg.DEVICE)
        
        # build default camera
        default_R, default_T = torch.eye(3), torch.zeros(3)
        
        # imageio_mp4_cfg = dict(
        #     fps=fps, mode='I', format='FFMPEG', macro_block_size=1
        # )

        # writer = imageio.get_writer(save_path, **imageio_mp4_cfg)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # import ipdb;ipdb.set_trace()
        writer = cv2.VideoWriter(save_path, fourcc, fps, (int(width), int(height)))
        
        bar = Bar('Rendering results ...', fill='#', max=len(val['frame_ids']))
        
        # # save normal map video
        # if save_ex:
        #     normal_map_writer = imageio.get_writer(normal_map_vid_save_path:=osp.join(output_pth, f'{fname}_normal.mp4'), **imageio_mp4_cfg)
        #     depth_map_writer = imageio.get_writer(depth_map_vid_save_path:=osp.join(output_pth, f'{fname}_depth.mp4'), **imageio_mp4_cfg)
            
        
        _global_R, _global_T = None, None
        
        im_to_write = []
        # run rendering
        for frame_i in results[sid]['frame_ids']:
            frame_i2 = np.where(val['frame_ids'] == frame_i)[0]
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i*stride)
            
            flag, org_img = cap.read()
            if not flag: break
            img = org_img[..., ::-1].copy()
            
            # render onto the input video
            renderer.create_camera(default_R, default_T)
            
            frame_i2 = frame_i2[0]
            vert = torch.from_numpy(val['verts'][frame_i2]).to(cfg.DEVICE)
            # renderer.update_K(5000.0, vert)
            ret = renderer.render_mesh(vert, img)
            # img, depth_img, normal_img
            img = ret.image
        
            if vis_global:
                # render the global coordinate
                if frame_i in results[sid]['frame_ids']:
                    # renderer.update_K(cliff_focal_length, vert)
                    
                    frame_i3 = np.where(results[sid]['frame_ids'] == frame_i)[0]
                    verts = verts_glob[[frame_i3]].to(cfg.DEVICE)
                    faces = renderer.faces.clone().squeeze(0)
                    colors = torch.ones((1, 4)).float().to(cfg.DEVICE); colors[..., :3] *= 0.9
                    
                    if _global_R is None:
                        _global_R = global_R[frame_i3].clone(); _global_T = global_T[frame_i3].clone()
                    cameras = renderer.create_camera(global_R[frame_i3], global_T[frame_i3])
                    img_glob = renderer.render_with_ground(verts, faces, colors, cameras, global_lights)
                
                try: img = np.concatenate((img, img_glob), axis=1)
                except: img = np.concatenate((img, np.ones_like(img) * 255), axis=1)
            
            # writer.append_data(img)
            # im_to_write.append(img)
            writer.write(img)
            
            # if save_ex:
            #     norm_im = tonp(normal_img)[...,::-1]
            #     img = crop_im(img)
            #     normal_map_writer.append_data(norm_im)
                
            #     depth_im = tonp(depth_img)
            #     depth_im = crop_im(depth_im)
            #     depth_map_writer.append_data(depth_im)
                
            bar.next()
            frame_i += 1
        # writer.close()
        writer.release()
        # import ipdb;ipdb.set_trace()
        # imageio.mimwrite(save_path, im_to_write, fps=fps)
        print(f'\nsave_path: {save_path}')
        
        # if save_ex:
        #     print(f'\normal_map_vid_save_path: {normal_map_vid_save_path}')
        #     print(f'\ndepth_map_vid_save_path: {depth_map_vid_save_path}')
    
    cap.release()
    