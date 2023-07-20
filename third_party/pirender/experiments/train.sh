jobname=${1-'mead'}
config_name=${2-'mead_tfhd'}
gpu_id=$3


# for first stage, use --no_resume
# for second stage, use --which_iter to load

# copy the checkpoints from other folder to cur logdir folder for finetuning
# --is_cross_id_loss 1 for identity perserving
#  batch_size=24
batch_size=12
#  CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port 12344 \
CUDA_VISIBLE_DEVICES=${gpu_id} python -m torch.distributed.launch --nproc_per_node=1 \
                            --master_port 12549 train.py \
                            --config ./config/${config_name}.yaml \
                            --name ${jobname} \
                            --batch_size ${batch_size} \
                            --is_cross_id_loss 1 \
                            --which_iter 550000
#                           --no_resume


#if [[ ${jobname} == 'w_mead_crossvideo_mix' ]]; then
#  batch_size=24
##  CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port 12344 \
#  CUDA_VISIBLE_DEVICES=${gpu_id} python -m torch.distributed.launch --nproc_per_node=1 \
#                              --master_port 12849 train.py \
#                              --config ./config/flame_wo_crop.yaml \
#                              --name ${jobname} \
#                              --batch_size ${batch_size} \
#                              --which_iter 610339
##                              --no_resume
#fi


#if [[ ${jobname} == 'cnhd_flame_wo_crop' ]]; then
#  batch_size=24
##  CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port 12344 \
#  CUDA_VISIBLE_DEVICES=${gpu_id} python -m torch.distributed.launch --nproc_per_node=1 \
#                              --master_port 12348 train.py \
#                              --config ./config/flame_wo_crop.yaml \
#                              --name ${jobname} \
#                              --batch_size ${batch_size} \
#                              --which_iter 328004
##                              --no_resume
#fi


#if [[ ${jobname} == 'cross_id_cnhd_flame_wo_crop' ]]; then
#  batch_size=12
##  CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port 12344 \
#  CUDA_VISIBLE_DEVICES=${gpu_id} python -m torch.distributed.launch --nproc_per_node=1 \
#                              --master_port 12348 train.py \
#                              --config ./config/flame_wo_crop.yaml \
#                              --name ${jobname} \
#                              --batch_size ${batch_size} \
#                              --which_iter 328004 \
#                              --is_cross_id_loss 1
##                              --no_resume
#fi

#if [[ ${jobname} == 'mix_flame_wo_crop' ]]; then
#  CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port 12346 train.py \
#                              --config ./config/flame_wo_crop.yaml \
#                              --name ${jobname} \
#                              --no_resume
#fi

#if [[ ${jobname} == 'flame_wo_crop' ]]; then
#  torchrun --nproc_per_node=1 --master_port 12345 train.py \
#                              --config ./config/flame_wo_crop.yaml \
#                              --name ${jobname} #\
##                              --no_resume
#fi
