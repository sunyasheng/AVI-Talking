jobname=${1-'align_emote'}
gpu_id=${2-'0,'}
ckpt_path=${3-'train_logs/align_emote_2023-11-28-10-58/last.pth'}
save_subdir=${4-'align_emote_2023-11-28-10-58_mead_caption_results'}
is_output_gt=${5-0}
is_no_diffusion=${6-0}
is_cal_diversity=${7-0}
is_vis_diversity=${8-0}
is_use_rvd=${9-0}


if [[ ${jobname} == 'align_emote' ]]; then
    current_date=$(date +"%Y-%m-%d-%H-%M")
    test_audio_path='/data/yashengsun/local_storage/paishe_w_cam/proc_emoca/transpose_crop_MVI_0030_002/transpose_crop_MVI_0030_002.wav'
    test_json_path='/data/yashengsun/Proj/TalkingFace/NExT-GPT/code/caption_results/'
    dataset_names='Mead_M,Mead_W'
    CUDA_VISIBLE_DEVICES=${gpu_id} python train_diffusion_prior.py \
                  --dataset_names ${dataset_names} \
                  --jobname ${jobname}_${current_date} \
                  --vertice_dim 15069 \
                  --batch_size 256 \
                  --resume_from_ckpt 1 \
                  --ckpt_path ${ckpt_path} \
                  --only_load_caption 1 \
                  --is_tensorboard_log 0 \
                  --max_lr 0.001 \
                  --test_audio_path ${test_audio_path} \
                  --test_json_path ${test_json_path} \
                  --is_test 1 \
                  --is_talking_instruct 1 \
                  --is_output_gt ${is_output_gt} \
                  --save_subdir ${save_subdir} \
                  --is_no_diffusion ${is_no_diffusion} \
                  --is_cal_diversity ${is_cal_diversity} \
                  --is_vis_diversity ${is_vis_diversity} \
                  --is_use_rvd ${is_use_rvd}
fi
