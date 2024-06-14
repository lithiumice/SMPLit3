cd /apdcephfs/private_wallyliang/PLANT/NeMF
python src/train_gmp.py re_gmp2.yaml

cd /apdcephfs/private_wallyliang/PLANT/NeMF
python src/train_gmp.py re_gmp3.yaml

cd /apdcephfs/private_wallyliang/PLANT/NeMF
python src/train_gmp_glamr_rep.py re_gmp_glamr_rep.yaml

cd /apdcephfs/private_wallyliang/PLANT/NeMF
python src/train.py generative_w_trans.yaml

cd /apdcephfs/private_wallyliang/PLANT/NeMF
python src/train.py generative2.yaml