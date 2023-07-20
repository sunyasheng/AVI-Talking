counter=0
for item in "${video_lists[@]}"; do
    counter=$((counter + 1))
    gpu_id=$(( counter % 8))
    item_name=${item%%.*}
    echo "Current item: $item" ${counter} ${gpu_id} ${item_name}

    find ${item_name} -type f -name video_geometry_detail_with_sound.mp4
done

