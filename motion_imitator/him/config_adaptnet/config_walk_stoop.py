env_cls = "ICCGANHumanoidTarget"

env_params = dict(
    motion_file="assets/motions/iccgan/stoop_walk.json",
    contactable_links=["right_foot", "left_foot"],
    ground_friction=0.15,
)
discriminators = {
    "walk/full": dict(
        key_links=None,
        parent_link=None,
    )
}
