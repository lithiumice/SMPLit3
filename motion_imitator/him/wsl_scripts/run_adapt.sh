conda activate isaac
cd /home/lithiumice/PHC/third-party/CompositeMotion

# composite
python main.py config/juggling+locomotion_walk.py --ckpt ckpt_juggling+locomotion_walk
python main.py config/juggling+locomotion_walk.py --ckpt pretrained/juggling+locomotion_walk --test
python main_composite.py config/juggling+locomotion_walk.py --ckpt /home/lithiumice/PHC/third-party/CompositeMotion/ckpt_juggling+locomotion_walk/ckpt --test


# adaptnet
cd /home/lithiumice/PHC/third-party/CompositeMotion

python main.py config_adaptnet/config_run_lowfriction.py \
--meta pretrained_adaptnet/locomotion_run --ckpt ./adaptnet_tain_save

python main.py config_adaptnet/config_run_lowfriction.py \
--meta pretrained_adaptnet/locomotion_run --ckpt pretrained_adaptnet/run_lowfriction --test

python main.py config_adaptnet/config_walk_stoop.py \
--meta pretrained_adaptnet/locomotion_walk --ckpt pretrained_adaptnet/walk_stoop --test

