import os
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
import glob


def read_exp(motion_deca_dir, name):
    frames_meta_path = pjoin(motion_deca_dir, name, 'EMOCA_v2_lr_mse_20')
    # import pdb; pdb.set_trace()

    exp_meta_paths = [pjoin(frames_meta_path, name, 'exp.npy') for name in os.listdir(frames_meta_path) if
                      os.path.isdir(os.path.join(frames_meta_path, name)) and 'processed' not in name and name.endswith('_000')]
    exp_meta_paths = sorted(exp_meta_paths)
    # import pdb; pdb.set_trace()
    pose_meta_paths = [p.replace('exp.npy', 'pose.npy') for p in exp_meta_paths]
    shape_meta_paths = [p.replace('exp.npy', 'shape.npy') for p in exp_meta_paths]
    cam_meta_paths = [p.replace('exp.npy', 'cam.npy') for p in exp_meta_paths]
    # import pdb; pdb.set_trace()
    # try:
    exp_meta = [np.load(p) for p in exp_meta_paths]
    exp_meta = np.stack(exp_meta)

    pose_meta = [np.load(p) for p in pose_meta_paths]
    pose_meta = np.stack(pose_meta)

    shape_meta = [np.load(p) for p in shape_meta_paths]
    shape_meta = np.stack(shape_meta)

    cam_meta = [np.load(p) for p in cam_meta_paths]
    cam_meta = np.stack(cam_meta)
    # except Exception as ex:
    #     import traceback
    #     traceback.print_exc()
    #     print(exp_meta_paths)
    #     print(pose_meta_paths)
    #     print(shape_meta_paths)
    #     print(cam_meta_paths)

    return exp_meta, pose_meta, shape_meta, cam_meta


def get_data(data_root, is_inference=False, target_folder_name=None, is_wo_audio=False):
    folder_names = os.listdir(data_root)
    print(len(folder_names))
    folder_names = [fn for fn in folder_names if not os.path.isfile(os.path.join(data_root, fn))]
    if target_folder_name is not None:
        folder_names = [target_folder_name]
    if not os.path.exists(os.path.join(data_root, folder_names[0], 'EMOCA_v2_lr_mse_20')):
        new_folder_names = []
        for folder_name in folder_names:
            # if folder_name == 'zzw_cctv_000036_0/109-114.mp4': continue
            sub_folder_names = os.listdir(os.path.join(data_root, folder_name))
            new_folder_names.extend([folder_name+'/'+sfn for sfn in sub_folder_names])
        # black_list = ['zzw_cctv_000036_0/109-114.mp4']
        black_list = []
        new_folder_names = [fn for fn in new_folder_names if fn not in black_list]
        folder_names = new_folder_names
        # print(new_folder_names)
        print(len(folder_names))

    # import pdb; pdb.set_trace()
    data = {}
    for i, folder_name in tqdm(enumerate(folder_names)):
        # print(folder_name, data_root)
        key = folder_name
        res_dict = {}
        wav_path = os.path.join(data_root, folder_name, folder_name+'.wav')
        if (not os.path.exists(wav_path) and (is_wo_audio is False)): continue
        try:
            res_dict["exp"], res_dict["pose"], res_dict["shape"], res_dict["cam"] = read_exp(data_root, folder_name)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(folder_name, 'has some problems.')
            continue
        res_dict["name"] = folder_name
        res_dict["paths"] = get_detect_paths(data_root, folder_name)
        res_dict["wav"] = wav_path
        # import pdb; pdb.set_trace()
        assert len(res_dict["paths"]) == res_dict["exp"].shape[0], '{}: {} not equal {}'.format(folder_name, len(res_dict["paths"]), res_dict["exp"].shape[0])
        if is_wo_audio is False:
            assert os.path.exists(res_dict["wav"]), '{} does not exist.'.format(res_dict["wav"])
        data[key] = res_dict
        if is_inference is True and i > 4: break
    # print(data.keys())
    return data


def get_detect_paths(motion_deca_dir, name):
    frames_meta_path = pjoin(motion_deca_dir, name, 'EMOCA_v2_lr_mse_20')
    if not '/' in name:
        processed_roots = [pjoin(frames_meta_path, fn_name, name, 'detections') for fn_name in os.listdir(frames_meta_path) if
                          os.path.isdir(os.path.join(frames_meta_path, fn_name)) and 'processed' in fn_name]
        if not os.path.exists(processed_roots[0]):
            processed_roots = [pjoin(frames_meta_path, fn_name, name[-3:], 'detections') for fn_name in
                               os.listdir(frames_meta_path) if
                               os.path.isdir(os.path.join(frames_meta_path, fn_name)) and 'processed' in fn_name]
            # import pdb; pdb.set_trace()
        if not os.path.exists(processed_roots[0]):
            processed_roots = [pjoin(frames_meta_path, fn_name, name+'.mp4', 'detections') for fn_name in
                               os.listdir(frames_meta_path) if
                               os.path.isdir(os.path.join(frames_meta_path, fn_name)) and 'processed' in fn_name]

    else:
        processed_roots = [pjoin(frames_meta_path, fn_name, name.split('/')[-1][:-4], 'detections') for fn_name in
                           os.listdir(frames_meta_path) if
                           os.path.isdir(os.path.join(frames_meta_path, fn_name)) and 'processed' in fn_name]
    processed_roots = sorted(processed_roots)
    # print(processed_roots)
    # assert len(processed_roots) == 1, 'length of processed_roots is not 1'
    processed_root = processed_roots[-1]
    img_paths = sorted(glob.glob(os.path.join(processed_root, '*_000.png')))
    if len(img_paths) ==0: print(processed_root)
    return img_paths
