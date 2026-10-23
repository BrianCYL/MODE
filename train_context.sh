for env_type in cheetah_vel hopper walker ant_dir;
do
    for data_quality in medium medium-expert;
        do
            python train_context.py \
                --env_type $env_type \
                --context_train_epochs 201 \
                --data_quality $data_quality
        done
done