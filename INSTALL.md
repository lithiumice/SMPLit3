# Installation

We provide [anaconda](https://www.anaconda.com/) environment as below.

Install miniconda with:
```bash
# In devcloud
export https_proxy=
export http_proxy=

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

Install system depandence:
```bash
# for centos
sudo yum install git cmake gcc make autoconf automake libtool -y
```

```bash
# Clone the repo
git clone https://github.com/lithiumice/SMPLit --recursive
# git submodule update --init --recursive
# or:
git clone https://github.com/princeton-vl/DPVO.git third-party/DPVO/
git clone https://github.com/ViTAE-Transformer/ViTPose.git third-party/ViTPose/

wget "https://www.dropbox.com/scl/fi/uu87dq9g3wj7v0vejbu8m/models.tar.gz?rlkey=a6me2oodq9v9z7vq3oqoaxir6&st=knu4fgr7&dl=0" -O models.tar.gz 

tar --exclude=expression_templates_famos --exclude=logs -zxvkf models.tar.gz 

mv models model_files

# if in tencent cluster, run:
echo "
[global]
index-url = http://mirrors.tencent.com/pypi/simple/
[install]
trusted-host = mirrors.tencent.com
" > ~/.pip/pip.conf



# Create Conda environment
source ~/miniconda3/bin/activate
conda create -n smplit python=3.10 -y
# Activate environment
conda activate smplit

# Install PyTorch libraries
conda install pytorch=1.13 pytorch-cuda=12.1 torchvision torchaudio -c pytorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# conda install pytorch==1.12.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
# pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# test your torch installation
python -c "import torch;print(torch.cuda.is_available());a=torch.zeros((1,1)).cuda()"

# Install PyTorch3D (optional) for visualization
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
# pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# Install dependencies
pip install -r requirements.txt
# ViTPose, Install in editable mode
pip install -v third-party/ViTPose
# SMPLX
pip install -v third-party/smplx
# ACR
pip install -r third-party/ACR/requirements.txt
# HMR
pip install -r third-party/4D-Humans/requirements.txt
# smrik
pip install -r third-party/smirk/requirements.txt
# difftraj
pip install -r difftraj/requirements.txt
pip install -r third-party/NeMF/requirements.txt
# others
pip install chardet
pip install numpy==1.23.0
pip install 'git+https://github.com/openai/CLIP.git'
pip install 'git+https://github.com/facebookresearch/detectron2.git'


# Install DPVO
cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip
conda install pytorch-scatter=2.0.9 -c rusty1s -y
# conda install cudatoolkit-dev=11.3.1 -c conda-forge -y
# If you can not install with conda, 
pip install torch-scatter
# ONLY IF your GCC version is larger than 10. `gcc --version` < 10
conda install -c conda-forge gxx=9.5 -y
# and make sure cuda version right
module load cuda/11.3
# install dpvo
pip install .

# # install osmesa to use pyrender
# conda install -c conda-forge glew mesalib -y
# conda install -c menpo glfw3 -y
# conda install -c menpo osmesa -y

# test osmesa installation
python -c "import pyrender"

# prepare data for hmr
mkdir -p /root/.cache/4DHumans/
mkdir -p /root/.cache/4DHumans/data/smpl/
cp model_files/uhc_data/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl /root/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl
cp model_files/uhc_data/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl /home/hyi/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl
cp /apdcephfs/private_wallyliang/hmr2_data.tar.gz /root/.cache/4DHumans/hmr2_data.tar.gz

# pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
```

sync the changes from private disk to docker.

```bash
rsync -av --exclude=.* \
--exclude='third-party' \
--exclude='Thirdparty/' \
--exclude='tmp/' \
--exclude='data/' \
--exclude='*__pycache__*' \
PATH_TO_SMPLit \
/root/


# rsync -av --exclude=.* \
# --exclude='tmp/' \
# --exclude='data/' \
# --exclude='*__pycache__*' \
# PATH_TO_SMPLitit \
# /root/
```

