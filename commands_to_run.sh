nohup ./actorshq/dataset/download_manager.py \
        ./actorshq_access_4x.yaml \
        /smb/archive/corelab_datasets/actorshq \
        --actor Actor04 \
        --sequence Sequence1 \
        --scale 4 > "download_Actor04_seq1_scale4.log" 2>&1 &

./humanrf/run.py \
    --config example_humanrf \
    --workspace /home/zg296/humanrf/example_workspace \
    --dataset.path /smb/archive/corelab_datasets/actorshq

export PYTHONPATH=$PYTHONPATH:/home/zg296/humanrf