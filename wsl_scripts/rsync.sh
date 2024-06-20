# ws to cluster
rsync -av --exclude='./env/' \
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
	--exclude='difftraj/' \
	--exclude='exp_output/' \
	--exclude='*__pycache__*' \
	--exclude='*.md' \
	--exclude='third-party' \
	/home/hyi/data/workspace/talking_avatar/SMPLit \
	/is/cluster/fast/hyi/workspace/VidGen/
