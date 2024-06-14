from glob import glob
import os
import cv2
from tqdm import tqdm
import subprocess
import os.path as osp
from loguru import logger
from pathlib import Path
import sys


from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector


def load_or_run(load_file_path, run_func):
    if os.path.exists(load_file_path):
        return joblib.load(load_file_path)
    else:
        run_ret =  run_func()
        joblib.dump(run_ret, load_file_path)
  
  
  
def pySceneDetect(input_video_path):
    print(f'pySceneDetect processing: {input_video_path}')
    video_manager = VideoManager([input_video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=3.0))
    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)
    scene_list = scene_manager.get_scene_list(base_timecode)
    if scene_list == []: scene_list = [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]
    return scene_list


def video():
    for actor in tqdm(filter(os.path.isdir, glob(f'./test_results/*_rgb_*'))):
        os.system(f'ffmpeg -y -framerate 30 -pattern_type glob -i \'{actor}/video/*.jpg\' -c:v libx264 {actor}.mp4')

def smplpix():
    FRAMES_DIR=r'C:\Users\lithiumice\code\speech2gesture_dataset\crop\stand\test_video\1-00_00_00-00_00_01\image'
    VIDEO_PATH=r'C:\Users\lithiumice\code\smplpix\colab_notebooks\video.mp4'
    os.makedirs(FRAMES_DIR, exist_ok=True)
    cmds=f"ffmpeg -i {VIDEO_PATH} -vf fps=2  -qscale:v 2 {FRAMES_DIR}/%06d.png"
    os.system(cmds)

    
@logger.catch
def video_to_images(
    vid_file,
    prefix='',
    start=0,
    duration=10,
    img_folder=None,
    return_info=False,
    fps=15
):
    '''
    From https://github.com/mkocabas/VIBE/blob/master/lib/utils/demo_utils.py

    fps will sample the video to this rate.
    '''
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-r', str(fps),
               '-ss', str(start),
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/{prefix}%06d.jpg']
    
    if duration is not None:
        command += ['-t', str(duration),]
               
    
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')


@logger.catch
def frame_to_video(
    image_path,
    video_name,
    fps=9,
    cut=-1,
    size = (960,540)
):
    # fps=15,
    # size = (960,540)
    # size = (480,720)
    logger.info(f'video_name: {video_name}')
    f_name=Path(video_name).stem
    ext_name=Path(video_name).suffixes[-1]
    
    
    
    if not Path(image_path).exists():
        logger.warning(f'image_path not exist!')
        return
        
    filelist = os.listdir(image_path)
    if filelist == []:
        logger.warning(f'the filelist is empty!')
        return
    
    # assert(filelist!=[],'the filelist is empty')
    
    filelist=filter(lambda x:(x[-3:]=='jpg') or (x[-3:]=='png'),filelist)
    filelist=list(filelist)
    # filelist.sort(key= lambda x:int(x[:-4].split('_')[0]))
    from natsort import natsorted
    filelist = natsorted(filelist)
    
    if cut!=-1: 
        filelist=filelist[:cut]
    if size==-1:
        size = cv2.imread(os.path.join(image_path, filelist[0])).shape[:2][::-1]
        
    if ext_name=='.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif ext_name=='.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'I420')* 'mp4v'
    else:
        # raise ValueError("Unknown ext_name")
        logger.warning(f'Unknown ext_name:{ext_name}')
        return
    
    
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    for item in tqdm(filelist,desc='frame to video ing...'):
        img_path=os.path.join(image_path, item)
        if not os.path.exists(img_path):
            logger.warning(f'{img_path} not exit...')
            break
        
        if item.endswith('.jpg') or item.endswith('.png'):
            img = cv2.imread(img_path)
            img2=cv2.resize(img,size)
            video.write(img2)
    video.release()
    logger.info('frame_to_video finished')
    
    
if __name__ == '__main__':
    in_dir=sys.argv[1]
    vid_name=sys.argv[2]
    frame_to_video(
        in_dir,
        video_name='/tmp/tmp.mp4',
        fps=30,size=-1
    )
    cmds=f'bash /apdcephfs/private_wallyliang/PLANT/ffmpeg_tools/reencode.sh /tmp/tmp.mp4 {vid_name}'
    print(cmds)
    subprocess.call(cmds,shell=True)
    import shutil;shutil.rmtree(in_dir)
    os.system(f'rm {vid_name}')
