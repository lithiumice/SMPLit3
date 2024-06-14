# Adversarial Skill Embeddings

Code accompanying the paper:
"ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters" \
(https://xbpeng.github.io/projects/ASE/index.html) \
![Skills](images/ase_teaser.png)


### Installation

Download Isaac Gym from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions.

Once Isaac Gym is installed, install the external dependencies for this repo:

```
pip install -r requirements.txt
```

# train AMP
python run_ase.py --task HumanoidAMP \
--cfg_env data_ase/cfg/humanoid_uhc.yaml \
--cfg_train data_ase/cfg/train/rlg/amp_humanoid.yaml \
--motion_file xxx \
--headless

# train LLC
python run_ase.py --task HumanoidAMPGetup \
--cfg_env data_ase/cfg/humanoid_ase_uhc.yaml \
--cfg_train data_ase/cfg/train/rlg/ase_humanoid.yaml \
--motion_file xxx \
--headless

# train HLC
python run_ase.py --task HumanoidHeading \
--cfg_env data_ase/cfg/humanoid_sword_shield_heading.yaml \
--cfg_train data_ase/cfg/train/rlg/hrl_humanoid.yaml \
--motion_file data_ase/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy \
--llc_checkpoint data_ase/models/ase_llc_reallusion_sword_shield.pth \
--headless

# test location task with high level controller
python run_ase.py --test --task HumanoidLocation --num_envs 16 \
--cfg_env data_ase/cfg/humanoid_sword_shield_location.yaml \
--cfg_train data_ase/cfg/train/rlg/hrl_humanoid.yaml \
--motion_file data_ase/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy \
--llc_checkpoint data_ase/models/ase_llc_reallusion_sword_shield.pth \
--checkpoint data_ase/models/ase_hlc_location_reallusion_sword_shield.pth

### ASE

#### Pre-Training

First, an ASE model can be trained to imitate a dataset of motions clips using the following command:
```
python run_ase.py --task HumanoidAMPGetup --cfg_env data_ase/cfg/humanoid_ase_sword_shield_getup.yaml --cfg_train data_ase/cfg/train/rlg/ase_humanoid.yaml --motion_file data_ase/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --headless
```
`--motion_file` can be used to specify a dataset of motion clips that the model should imitate. 
The task `HumanoidAMPGetup` will train a model to imitate a dataset of motion clips and get up after falling.
Over the course of training, the latest checkpoint `Humanoid.pth` will be regularly saved to `output/`,
along with a Tensorboard log. `--headless` is used to disable visualizations. If you want to view the
simulation, simply remove this flag. To test a trained model, use the following command:
```
python ase/run.py --test --task HumanoidAMPGetup --num_envs 16 --cfg_env data_ase/cfg/humanoid_ase_sword_shield_getup.yaml --cfg_train data_ase/cfg/train/rlg/ase_humanoid.yaml --motion_file data_ase/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --checkpoint [path_to_ase_checkpoint]
```
You can also test the robustness of the model with `--task HumanoidPerturb`, which will throw projectiles at the character.

&nbsp;

#### Task-Training

After the ASE low-level controller has been trained, it can be used to train task-specific high-level controllers.
The following command will use a pre-trained ASE model to perform a target heading task:
```
python ase/run.py --task HumanoidHeading --cfg_env data_ase/cfg/humanoid_sword_shield_heading.yaml --cfg_train data_ase/cfg/train/rlg/hrl_humanoid.yaml --motion_file data_ase/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint [path_to_llc_checkpoint] --headless
```
`--llc_checkpoint` specifies the checkpoint to use for the low-level controller. A pre-trained ASE low-level
controller is available in `data_ase/models/ase_llc_reallusion_sword_shield.pth`.
`--task` specifies the task that the character should perform, and `--cfg_env` specifies the environment
configurations for that task. The built-in tasks and their respective config files are:
```
HumanoidReach: data_ase/cfg/humanoid_sword_shield_reach.yaml
HumanoidHeading: data_ase/cfg/humanoid_sword_shield_heading.yaml
HumanoidLocation: data_ase/cfg/humanoid_sword_shield_location.yaml
HumanoidStrike: data_ase/cfg/humanoid_sword_shield_strike.yaml
```
To test a trained model, use the following command:
```
python ase/run.py --test --task HumanoidHeading --num_envs 16 --cfg_env data_ase/cfg/humanoid_sword_shield_heading.yaml --cfg_train data_ase/cfg/train/rlg/hrl_humanoid.yaml --motion_file data_ase/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint [path_to_llc_checkpoint] --checkpoint [path_to_hlc_checkpoint]
```


&nbsp;

&nbsp;

#### Pre-Trained Models

Pre-trained models are provided in `data_ase/models/`. To run a pre-trained ASE low-level controller,
use the following command:
```
python ase/run.py --test --task HumanoidAMPGetup --num_envs 16 --cfg_env data_ase/cfg/humanoid_ase_sword_shield_getup.yaml --cfg_train data_ase/cfg/train/rlg/ase_humanoid.yaml --motion_file data_ase/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --checkpoint data_ase/models/ase_llc_reallusion_sword_shield.pth
```

Pre-trained models for the different tasks can be run using the following commands:

Heading:
```
python ase/run.py --test --task HumanoidHeading --num_envs 16 --cfg_env data_ase/cfg/humanoid_sword_shield_heading.yaml --cfg_train data_ase/cfg/train/rlg/hrl_humanoid.yaml --motion_file data_ase/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint data_ase/models/ase_llc_reallusion_sword_shield.pth --checkpoint data_ase/models/ase_hlc_heading_reallusion_sword_shield.pth
```

Reach:
```
python ase/run.py --test --task HumanoidReach --num_envs 16 --cfg_env data_ase/cfg/humanoid_sword_shield_reach.yaml --cfg_train data_ase/cfg/train/rlg/hrl_humanoid.yaml --motion_file data_ase/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint data_ase/models/ase_llc_reallusion_sword_shield.pth --checkpoint data_ase/models/ase_hlc_reach_reallusion_sword_shield.pth
```

Location:
```
python ase/run.py --test --task HumanoidLocation --num_envs 16 --cfg_env data_ase/cfg/humanoid_sword_shield_location.yaml --cfg_train data_ase/cfg/train/rlg/hrl_humanoid.yaml --motion_file data_ase/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint data_ase/models/ase_llc_reallusion_sword_shield.pth --checkpoint data_ase/models/ase_hlc_location_reallusion_sword_shield.pth
```

Strike:
```
python ase/run.py --test --task HumanoidStrike --num_envs 16 --cfg_env data_ase/cfg/humanoid_sword_shield_strike.yaml --cfg_train data_ase/cfg/train/rlg/hrl_humanoid.yaml --motion_file data_ase/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint data_ase/models/ase_llc_reallusion_sword_shield.pth --checkpoint data_ase/models/ase_hlc_strike_reallusion_sword_shield.pth
```

&nbsp;

&nbsp;

### AMP

We also provide an implementation of Adversarial Motion Priors (https://xbpeng.github.io/projects/AMP/index.html).
A model can be trained to imitate a given reference motion using the following command:
```
python ase/run.py --task HumanoidAMP --cfg_env data_ase/cfg/humanoid_sword_shield.yaml --cfg_train data_ase/cfg/train/rlg/amp_humanoid.yaml --motion_file data_ase/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy --headless
```
The trained model can then be tested with:
```
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env data_ase/cfg/humanoid_sword_shield.yaml --cfg_train data_ase/cfg/train/rlg/amp_humanoid.yaml --motion_file data_ase/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy --checkpoint /home/lithiumice/ASE/output/Humanoid_01-12-42-22/nn/Humanoid.pth
```

&nbsp;

&nbsp;

### Motion Data

Motion clips are located in `data_ase/motions/`. Individual motion clips are stored as `.npy` files. Motion datasets are specified by `.yaml` files, which contains a list of motion clips to be included in the dataset. Motion clips can be visualized with the following command:
```
python ase/run.py --test --task HumanoidViewMotion --num_envs 2 --cfg_env data_ase/cfg/humanoid_sword_shield.yaml --cfg_train data_ase/cfg/train/rlg/amp_humanoid.yaml --motion_file data_ase/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy
```
`--motion_file` can be used to visualize a single motion clip `.npy` or a motion dataset `.yaml`.


This motion data is provided courtesy of Reallusion, strictly for noncommercial use. The original motion data is available at:

https://actorcore.reallusion.com/motion/pack/studio-mocap-sword-and-shield-stunts

https://actorcore.reallusion.com/motion/pack/studio-mocap-sword-and-shield-moves


If you want to retarget new motion clips to the character, you can take a look at an example retargeting script in `ase/poselib/retarget_motion.py`.
