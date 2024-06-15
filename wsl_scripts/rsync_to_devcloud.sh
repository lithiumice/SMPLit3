
# see: C:\Users\lithiumice\.ssh\config
# rsync -av --progress /home/lithiumice/talking_avatar/SMPLit2/ root@devcloud:/apdcephfs/private_wallyliang/SMPLit/

rsync -av --progress \
	--exclude='.*' \
	--exclude='tmp/' \
	--exclude='*__pycache__*' \
	--exclude='third-party/' \
	--exclude='model_files/' \
	--exclude='pretrained_weights/' \
	/home/lithiumice/talking_avatar/SMPLit3/ \
    root@devcloud:/apdcephfs/private_wallyliang/SMPLit/
