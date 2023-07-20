jobname=${1-'align_emote'}
gpu_id=${2-'0,'}


if [[ ${jobname} == 'align_emote' ]]; then
    current_date=$(date +"%Y-%m-%d-%H-%M")
    ckpt_path='train_logs/align_emote_2023-11-29-19-21/last.pth'
    dataset_names='Mead_M,Mead_W'
    CUDA_VISIBLE_DEVICES=${gpu_id} python train_diffusion_prior.py \
                  --dataset_names ${dataset_names}  \
                  --jobname ${jobname}_${current_date} \
                  --vertice_dim 15069 \
                  --batch_size 256 \
                  --resume_from_ckpt 1 \
                  --ckpt_path ${ckpt_path} \
                  --only_load_caption 1 \
                  --max_lr 0.0001 \
                  --max_epoch 20000
fi
