jobname=$1


path_to_videos=/data/yashengsun/local_storage/Mead/M003/fps25_aligned_av
path_to_keypoint=/data/yashengsun/local_storage/Mead/M003/fps25_keypoint
path_to_coef=/data/yashengsun/local_storage/Mead/M003/coef
path_to_lmdb=/data/yashengsun/local_storage/Mead/M003/lmdb

#path_to_videos=/data/yashengsun/Proj/TalkingFace/samples1_align_av/fps25_mp4
#path_to_keypoint=/data/yashengsun/Proj/TalkingFace/samples1_align_av/fps25_keypoint
#path_to_coef=/data/yashengsun/Proj/TalkingFace/samples1_align_av/coef
#path_to_lmdb=/data/yashengsun/Proj/TalkingFace/samples1_align_av/lmdb


#path_to_videos=/data/yashengsun/Proj/TalkingFace/samples_align_av/fps25_mp4
#path_to_keypoint=/data/yashengsun/Proj/TalkingFace/samples_align_av/keypoint
#path_to_coef=/data/yashengsun/Proj/TalkingFace/samples_align_av/coef
#path_to_lmdb=/data/yashengsun/Proj/TalkingFace/samples_align_av/lmdb

#path_to_videos=/data/yashengsun/Proj/data/processed
#path_to_keypoint=/data/yashengsun/Proj/data/keypoint
#path_to_coef=/data/yashengsun/Proj/data/coef
#path_to_lmdb=/data/yashengsun/Proj/data/lmdb

#path_to_videos=/data/yashengsun/Proj/TalkingFace/PIRender/samples/source_video/video
#path_to_keypoint=/data/yashengsun/Proj/TalkingFace/PIRender/samples/source_video/keypoint
#path_to_coef=/data/yashengsun/Proj/TalkingFace/PIRender/samples/source_video/coef
#path_to_lmdb=/data/yashengsun/Proj/TalkingFace/PIRender/samples/source_video/lmdb

#path_to_videos=/data/yashengsun/Proj/TalkingFace/CelebV-Text/downloaded_celebvtext/processed
#path_to_keypoint=/data/yashengsun/Proj/TalkingFace/CelebV-Text/downloaded_celebvtext/keypoint
#path_to_coef=/data/yashengsun/Proj/TalkingFace/CelebV-Text/downloaded_celebvtext/coef
#path_to_lmdb=/data/yashengsun/Proj/TalkingFace/CelebV-Text/downloaded_celebvtext/lmdb

#path_to_videos=/data/yashengsun/local_storage/CelebV-Text/downloaded_celebvtext/fps25_aligned_av
#path_to_keypoint=/data/yashengsun/local_storage/CelebV-Text/downloaded_celebvtext/fps25_keypoint
#path_to_coef=/data/yashengsun/local_storage/CelebV-Text/downloaded_celebvtext/fps25_coef
#path_to_lmdb=/data/yashengsun/local_storage/CelebV-Text/downloaded_celebvtext/fps25_lmdb

if [[ ${jobname} == 'extract_ldmk' ]]; then
  echo 'begin extract_ldmk'
  cd Deep3DFaceRecon_pytorch/
  python extract_kp_videos.py \
    --input_dir ${path_to_videos} \
    --output_dir ${path_to_keypoint} \
    --device_ids 0 \
    --workers 4
fi


if [[ ${jobname} == 'extract_coef' ]]; then
  cd Deep3DFaceRecon_pytorch/
  python face_recon_videos.py \
    --input_dir ${path_to_videos} \
    --keypoint_dir ${path_to_keypoint} \
    --output_dir ${path_to_coef} \
    --inference_batch_size 100 \
    --name=model_name \
    --epoch=20 \
    --model facerecon
#    --name=model_name \
fi


if [[ ${jobname} == 'sliced_extract_ldmk' ]]; then
  echo 'begin extract_ldmk'
  cd Deep3DFaceRecon_pytorch/
  node_i=$2
  python extract_kp_videos.py \
    --input_dir ${path_to_videos} \
    --output_dir ${path_to_keypoint} \
    --device_ids 0,1,2,3,4,5,6,7 \
    --workers 16 \
    --slice_i ${node_i} \
    --slice_all 3
fi

mkdir -p ${path_to_coef}

if [[ ${jobname} == 'sliced_extract_coef' ]]; then
  echo 'begin extract coef'
  cd Deep3DFaceRecon_pytorch/
  nodename=$2
#  slice_all=8
#  st=$((nodename * 8))
#  ed=$((st + 8))
  slice_all=1
  st=0
  ed=1
  for ((i=st; i<ed; i++))
  do
    gpu_id=$((i % 8))
    slice_i=$i
    echo ${slice_i} ${slice_all} ${gpu_id}
    CUDA_VISIBLE_DEVICES=${gpu_id} python face_recon_videos.py \
    --input_dir ${path_to_videos} \
    --keypoint_dir ${path_to_keypoint} \
    --output_dir ${path_to_coef} \
    --inference_batch_size 100 \
    --name=model_name \
    --epoch=20 \
    --slice_all=${slice_all} \
    --slice_i=${slice_i} \
    --model facerecon &
#    break
  done
fi


#if [[ ${jobname} == 'extract_lmdb' ]]; then
#  python scripts/prepare_vox_lmdb.py \
#    --path ${path_to_videos} \
#    --coeff_3dmm_path ${path_to_coef} \
#    --out ${path_to_lmdb}
#fi

