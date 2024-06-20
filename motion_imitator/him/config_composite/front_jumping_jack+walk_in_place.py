import numpy as np

env_cls = "ICCGANHumanoid"
env_params = dict(
    episode_length=300, motion_file="assets/motions/gym/chest_open+walk_in_place.json"
)

training_params = dict(max_epochs=30000, save_interval=10000, terminate_reward=-1)

discriminators = {
    "front_jumping_jack/upper": dict(
        motion_file="assets/motions/gym/front_jumping_jack.json",
        key_links=[
            "torso",
            "head",
            "right_upper_arm",
            "right_lower_arm",
            "right_hand",
            "left_upper_arm",
            "left_lower_arm",
            "left_hand",
        ],
        parent_link="pelvis",
    ),
    "walk_in_place/lower": dict(
        key_links=[
            "pelvis",
            "right_thigh",
            "right_shin",
            "right_foot",
            "left_thigh",
            "left_shin",
            "left_foot",
        ],
        parent_link=None,
    ),
}
