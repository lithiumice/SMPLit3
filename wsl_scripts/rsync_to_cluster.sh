
# see: C:\Users\lithiumice\.ssh\config
# rsync -av --progress /home/lithiumice/talking_avatar/SMPLit2/ root@devcloud:/apdcephfs/private_wallyliang/SMPLit/

rsync -av --progress \
	--exclude='.*' \
	--exclude='tmp/' \
	--exclude='./env/' \
	--exclude='.*' \
	--exclude='./venv/' \
	--exclude='ENV/' \
	--exclude='Thirdparty/' \
	--exclude='transformations/nerf' \
	--exclude='mlrun*' \
	--exclude='model/*' \
	--exclude='pretrained_weights/' \
	--exclude='tmp/' \
	--exclude='data/' \
	--exclude='exp_output/' \
	--exclude='output*' \
	--exclude='*__pycache__*' \
	--exclude='third-party' \
	--exclude='*__pycache__*' \
	--exclude='third-party/' \
	--exclude='model_files/' \
	--exclude='model_files_part/' \
	--exclude='pretrained_weights/' \
	/home/lithiumice/talking_avatar/SMPLit3/ \
    hyi@hongwei:/is/cluster/fast/hyi/workspace/VidGen/SMPLit3/
