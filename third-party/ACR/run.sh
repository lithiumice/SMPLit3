cd /apdcephfs/private_wallyliang/PLANT/Thirdparty/ACR
export PYOPENGL_PLATFORM=osmesa
python -m acr.main --demo_mode video -t \
--model_path /apdcephfs/private_wallyliang/PLANT/Thirdparty/ACR/wild.pkl \
--inputs /apdcephfs/private_wallyliang/PLANT/Thirdparty/ACR/demo/magic.mp4

pip install imgaug --index-url https://mirrors.tencent.com/pypi/simple/