import glob
import os
import random
import argparse
from tqdm import tqdm
from multiprocessing.pool import Pool

def find_dir_w_suffix(dir_path, suffix='.mp4'):
    res_list = []
    # import pdb; pdb.set_trace();
    for sub_name in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, sub_name)
        # print(sub_dir_path)
        if os.path.isdir(sub_dir_path):
            sub_res_list = find_dir_w_suffix(sub_dir_path, suffix=suffix)
            res_list += sub_res_list
        else:
            if sub_dir_path.endswith(suffix):
                res_list.append(sub_dir_path)
    return res_list


def organize2intructedtalker(gender='W'):
    if gender == 'M':
        Mead_output_root = '/data/yashengsun/local_storage/Mead_M_output'
        new_output_root = '/data/yashengsun/local_storage/Mead_emoca/Mead_M'
        Mead_video_root = '/data/yashengsun/local_storage/Mead_M_fps25/'

    if gender == 'W':
        Mead_output_root = '/data/yashengsun/local_storage/Mead_W_output'
        new_output_root = '/data/yashengsun/local_storage/Mead_emoca/Mead_W'
        Mead_video_root = '/data/yashengsun/local_storage/Mead_W_fps25/'

    mp4_paths = find_dir_w_suffix(Mead_video_root)
    mp4_paths = [p for p in mp4_paths if 'front' in p]

    dir_paths = [p[:-4].replace(Mead_video_root,'') for p in mp4_paths]
    old_dir_paths = [os.path.join(Mead_output_root, p) for p in dir_paths]

    # old_dir_paths = [p for p in old_dir_paths if 'front' in p and os.path.exists(p)]
    # random.shuffle(old_dir_paths)
    # old_dir_paths = old_dir_paths[:50]
    # import pdb; pdb.set_trace()
    if 'Mead_M' in Mead_output_root:
        concat_dir_names = ['M'+p.replace(Mead_output_root+'/', '')[1:].replace('/','_').replace('level_', 'level') for p in old_dir_paths]
    if 'Mead_W' in Mead_output_root:
        concat_dir_names = ['W'+p.replace(Mead_output_root + '/', '')[1:].replace('/', '_').replace('level_', 'level') for p in old_dir_paths]

    new_dir_paths = [os.path.join(new_output_root, n) for n in concat_dir_names]

    for i, (from_dir, to_dir) in enumerate(tqdm(zip(old_dir_paths, new_dir_paths))):
        # print(from_dir, to_dir)
        if not os.path.exists(os.path.join(from_dir, 'EMOCA_v2_lr_mse_20')): continue
        os.makedirs(to_dir, exist_ok=True)
        cmd = 'mv {}/EMOCA_v2_lr_mse_20 {}'.format(from_dir, to_dir)
        print(cmd)
        # import pdb; pdb.set_trace()
        os.system(cmd)

        cmd = 'mv {}.wav {}'.format(from_dir.replace(Mead_output_root, Mead_video_root),
                                    os.path.join(to_dir, os.path.basename(to_dir)+'.wav'))
        print(cmd)
        # import pdb; pdb.set_trace()
        os.system(cmd)
        # if i > 256: break

def mp42mp3(gender='W'):
    if gender == 'W':
        Mead_video_root = '/data/yashengsun/local_storage/Mead_W_fps25/'
    if gender == 'M':
        Mead_video_root = '/data/yashengsun/local_storage/Mead_M_fps25/'
    mp4_paths = find_dir_w_suffix(Mead_video_root)
    wav_paths = [mp4_path.replace('.mp4', '.wav') for mp4_path in mp4_paths]
    for mp4_path, wav_path in tqdm(zip(mp4_paths, wav_paths)):
        if not os.path.exists(wav_path):
            cmd = 'ffmpeg -i {} -ar 16000 {} -loglevel error'.format(mp4_path, wav_path)
            # print(cmd)
            # import pdb; pdb.set_trace()
            os.system(cmd)


def proc2fps25(pair):
    mp4_path, mp4_fps25_path = pair
    # for (mp4_path, mp4_fps25_path) in tqdm(zip(mp4_paths, mp4_fps25_paths)):
    os.makedirs(os.path.dirname(mp4_fps25_path), exist_ok=True)
    cmd = 'ffmpeg -i {} -r 25 -q:v 0 {} -y -loglevel error'.format(mp4_path, mp4_fps25_path)
    # import pdb;pdb.set_trace()
    os.system(cmd)

def mp42fps25(gender='W'):

    # Mead_video_root = '/data/yashengsun/local_storage/Mead_M/'
    # Mead_video_fps25_root = '/data/yashengsun/local_storage/Mead_M_fps25/'
    if gender == 'W':
        Mead_video_root = '/data/yashengsun/local_storage/Mead_W/'
        Mead_video_fps25_root = '/data/yashengsun/local_storage/Mead_W_fps25/'

    if gender == 'M':
        Mead_video_root = '/data/yashengsun/local_storage/Mead_M/'
        Mead_video_fps25_root = '/data/yashengsun/local_storage/Mead_M_fps25/'

    mp4_paths = find_dir_w_suffix(Mead_video_root)
    mp4_paths = [p for p in mp4_paths if 'front' in p]
    mp4_fps25_paths = [p.replace(Mead_video_root, Mead_video_fps25_root) for p in mp4_paths]
    # mp4_fps25_paths = [p for p in mp4_fps25_paths if 'front' in p]
    print('Mead_{}_fps25 include {} frontal instances.'.format(gender, len(mp4_fps25_paths)))
    # import pdb; pdb.set_trace()
    pool = Pool(processes=12)
    pool.map(proc2fps25, list(zip(mp4_paths, mp4_fps25_paths)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mp42mp3')
    parser.add_argument('--gender', type=str, default='M')
    args = parser.parse_args()

    # Note that it requires first converting to fps 25, and then changing mp4 to wav
    if args.mode == 'organize':
        organize2intructedtalker(gender=args.gender)
    elif args.mode == 'mp42fps25':
        mp42fps25(gender=args.gender)
    elif args.mode == 'mp42mp3':
        mp42mp3(gender=args.gender)
