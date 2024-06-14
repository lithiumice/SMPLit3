cd /root/apdcephfs/private_wallyliang/PLANT/Thirdparty/NeMF
take_taiji python src/train_hmp.py hmp_L.yaml
rm -r /apdcephfs/share_1330077/wallyliang/hmp_saves/hmp_L

cd /root/apdcephfs/private_wallyliang/PLANT/Thirdparty/NeMF
take_taiji python src/datasets/ReInterHand.py ReInterHand_L.yaml
rm -r /apdcephfs/share_1330077/wallyliang/hmp_saves/ReInterhand_L

