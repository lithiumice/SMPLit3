# wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
tar -xzf /apdcephfs/private_wallyliang/PLANTmodels/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/

# install pip library
pip install -r /apdcephfs/private_wallyliang/PLANT/motion_imitator/requirement.txt --index-url https://mirrors.tencent.com/pypi/simple/
pip install "mujoco-py<2.2,>=2.1" --index-url https://mirrors.tencent.com/pypi/simple/
pip install "cython<3" --index-url https://mirrors.tencent.com/pypi/simple/
pip install "opencv-python==4.5.5.64" -U --index-url https://mirrors.tencent.com/pypi/simple/
pip install "setuptools==59.5.0" -U --index-url https://mirrors.tencent.com/pypi/simple/
pip install "nvitop" -U --index-url https://mirrors.tencent.com/pypi/simple/
# cd /apdcephfs/private_wallyliang/git_codes/vid2player3d/poselib && pip install -e .

# install system dependency
cp /etc/apt/sources.list /etc/apt/sources.list.backup && sh -c 'echo -e "deb http://mirrors.tencent.com/ubuntu/ focal main restricted universe multiverse\ndeb http://mirrors.tencent.com/ubuntu/ focal-security main restricted universe multiverse\ndeb http://mirrors.tencent.com/ubuntu/ focal-updates main restricted universe multiverse\ndeb http://mirrors.tencent.com/ubuntu/ focal-backports main restricted universe multiverse" > /etc/apt/sources.list' && apt update
apt install libosmesa6-dev libgl1-mesa-glx libglfw3 -y


