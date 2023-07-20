import os
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
import glob


def get_data(data_root, is_inference=False, target_folder_name=None, is_wo_audio=False):
    folder_names = os.listdir(data_root)
    print(len(folder_names))
    folder_names = [fn for fn in folder_names if not os.path.isfile(os.path.join(data_root, fn))]
    if target_folder_name is not None:
        folder_names = [target_folder_name]

    data = {}
    for i, folder_name in tqdm(enumerate(folder_names)):
        # print(folder_name, data_root)
        key = folder_name
        res_dict = {}
        wav_path = os.path.join(data_root, folder_name, folder_name+'.wav')
        if (not os.path.exists(wav_path) and (is_wo_audio is False)): continue
        res_dict["name"] = folder_name
        res_dict["wav"] = wav_path
        data[key] = res_dict
        if is_inference is True and i > 4: break
    # print(data.keys())
    return data
