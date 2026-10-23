for seed in 0 530 1225 326 412;
do
    python downstream_train.py \
        --env_type ${1} \
        --agent_save_folder_name agent \
        --data_quality ${2} \
        --downstream_batch_size 256 \
        --downstream_h_dim 256 \
        --state_normalize True \
        --scheduler True \
        --num_mixtures 8 \
        --start_step 0 \
        --seed $seed 
done