env_cls = "ICCGANHumanoid"
env_params = dict(episode_length=300, motion_file="assets/motions/iccgan/roll.json")

training_params = dict(max_epochs=5000, save_interval=2000, terminate_reward=-1)

discriminators = {
    "_/full": dict(
        parent_link=None,
    )
}
