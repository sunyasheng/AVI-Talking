jobname=$1


#if [[ ${jobname} == 'download_ckpt' ]]; then
#  bash scripts/download_weights.sh
#fi
#
#if [[ ${jobname} == 'download_data' ]]; then
#  bash scripts/download_demo_dataset.sh
#fi


if [[ ${jobname} == 'same_id_reenact' ]]; then
  mkdir -p ./inference_result/${jobname}
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345
#  CUDA_VISIBLE_DEVICES=1,
  CUDA_VISIBLE_DEVICES=0, torchrun --nproc_per_node=1 --master_port 54329 inference_flame.py \
      --config ./config/flame_wo_crop.yaml \
      --name mix_flame_wo_crop \
      --no_resume \
      --output_dir ./inference_result/${jobname}
fi

if [[ ${jobname} == 'cross_id_reenact' ]]; then
  mkdir -p ./inference_result/${jobname}
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345
#  CUDA_VISIBLE_DEVICES=1,
  CUDA_VISIBLE_DEVICES=3, python -m torch.distributed.launch --nproc_per_node=1 \
      --master_port 54329 inference_flame.py \
      --config ./config/flame_wo_crop.yaml \
      --name cross_id_cnhd_flame_wo_crop \
      --output_dir ./inference_result/${jobname} \
      --cross_id
#      --no_resume \
fi

if [[ ${jobname} == 'ori_same_id_reenact' ]]; then
  mkdir -p ./inference_result/${jobname}
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345
#  CUDA_VISIBLE_DEVICES=1,
  CUDA_VISIBLE_DEVICES=1, torchrun --nproc_per_node=1 --master_port 54322 inference.py \
      --config ./config/face_demo.yaml \
      --name face \
      --no_resume \
      --output_dir ./inference_result/${jobname}
fi

#if [[ ${jobname} == 'cross_id_reenact' ]]; then
##  python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345
##  torchrun --nproc_per_node=1 --master_port 12345
#  mkdir -p ./inference_result/${jobname}
#  CUDA_VISIBLE_DEVICES=1, python inference.py \
#    --config ./config/flame_wo_crop.yaml \
#    --name face \
#    --no_resume \
#    --output_dir ./inference_result/${jobname} \
#    --cross_id
#fi
#
#
#if [[ ${jobname} == 'install' ]]; then
#  cd Deep3DFaceRecon_pytorch/nvdiffrast
#  pip install .
#
#  cd Deep3DFaceRecon_pytorch/BFM
#  gdown 1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6
##  cp scripts/face_recon_videos.py ./Deep3DFaceRecon_pytorch
##  cp scripts/extract_kp_videos.py ./Deep3DFaceRecon_pytorch
##  cp scripts/coeff_detector.py ./Deep3DFaceRecon_pytorch
##  cp scripts/inference_options.py ./Deep3DFaceRecon_pytorch/options
#
##  cd ..
##  git clone https://github.com/deepinsight/insightface.git
#
#fi
#
#
#if [[ ${jobname} == '3dmm_coef' ]]; then
#  cd Deep3DFaceRecon_pytorch
#  python coeff_detector.py \
#  --input_dir ../demo_images \
#  --keypoint_dir ../demo_images \
#  --output_dir ../demo_images \
#  --name=model_name \
#  --epoch=20 \
#  --model facerecon
#fi
#
#
#if [[ ${jobname} == 'intuitive_control' ]]; then
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 intuitive_control.py \
#    --config ./config/face_demo.yaml \
#    --name face \
#    --no_resume \
#    --output_dir ./vox_result/face_intuitive \
#    --input_name ./demo_images
#fi
#
#
#if [[ ${jobname} == 'coef_control' ]]; then
#  python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 coef_control.py \
#    --config ./config/face_demo.yaml \
#    --name face \
#    --no_resume \
#    --output_dir ./vox_result/face_intuitive \
#    --input_name ./demo_images
#fi
#
#
#
#if [[ ${jobname} == 'video_coef' ]]; then
#  video_root=/data/yashengsun/Proj/TalkingFace/CelebV-Text/downloaded_celebvtext/processed
#  coef_root=/data/yashengsun/Proj/TalkingFace/CelebV-Text/downloaded_celebvtext/coef
#  cd Deep3DFaceRecon_pytorch
#  python video_coeff_detector.py \
#  --input_dir ../demo_images \
#  --keypoint_dir ../demo_images \
#  --output_dir ../demo_images \
#  --video_root=${video_root} \
#  --coef_root=${coef_root} \
#  --slice_all=64 \
#  --slice_i=0 \
#  --name=model_name \
#  --epoch=20 \
#  --model facerecon
#fi
#
#
