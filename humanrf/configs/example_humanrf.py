import actorshq.evaluation.presets as presets

frame_configs = presets.frame_configs["siggraph_interval_1"]

config = [
    # fmt: off
    "--train", "false",
    "--evaluate", "false",

    "--is_orbited", "false",
    "--is_uniformed", "true",
    "--sample_number", "150",
    #"--radius", "5",
    "--specific_frame", "0",
    "--object_center_addition", "0.2,0.1,0.3",
    "--camera_parameters", "full_body,747,1022,2.7407193318874006,0.09532264501827802,-1.1067867327072083,1.664400091893507,1.4165736816244041,1.697522516802703,1.550998985203991,1.1344146198042888,0.49916487970179935,0.5086164791544809",
    #"--test.trajectory_via_calibration_file", "/content/gdrive/MyDrive/archive/Actor01/Sequence1/4x/deneme.csv",
    "--test.checkpoint", "/content/gdrive/MyDrive/HumanRF/example_workspace/checkpoints/best.pth",

    "--model.log2_hashmap_size", "19",
    "--model.n_features_per_level", "2",
    "--model.n_levels", "16",
    "--model.coarsest_resolution", "32",
    "--model.finest_resolution", "2048",

    "--model.temporal_partitioning", "adaptive",
    "--model.expansion_factor_threshold", "1.25",
    "--model.camera_embedding_dim", "2",  # This is set to "0" for the numerical comparisons in the paper.

    "--training.max_steps", "50_001",
    "--training.scaler_growth_interval", "100_000",
    "--training.samples_max_batch_size", "640_000",
    "--validation.repeat_cameras", "2",
    "--validation.every_n_steps", "2_500",

    "--training.camera_preset", "siggraph_train",
    "--validation.camera_preset", "siggraph_train_validation",
    "--evaluation.camera_preset", "siggraph_test",
    "--evaluation.coverage", "siggraph_test",

    "--dataset.actor", "Actor01",
    "--dataset.sequence", "Sequence1",
    "--dataset.scale", "4",
    "--dataset.crop_center_square", "true",
    "--dataset.filter_light_bloom", "false",  # Set "true" to avoid light bleeding into the actor.
    "--dataset.frame_numbers", *[str(i) for i in range(*frame_configs)],
    # fmt: on
]
