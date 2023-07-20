import glob
import os
import sys


def find_dir_w_suffix(dir_path, suffix='.wav'):
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


def main():
    wav_root = '/data/yashengsun/local_storage/Video_Speech_Actor_wav/Audio_Speech_Actors_01-24'
    wav_paths = find_dir_w_suffix(wav_root, '.wav')
    # import pdb; pdb.set_trace()
    for wav_path in wav_paths:
        wav_16000_path = wav_path.replace('Video_Speech_Actor_wav', 'Video_Speech_Actor_wav_16000')
        os.makedirs(os.path.dirname(wav_16000_path), exist_ok=True)
        cmd = 'ffmpeg -i {} -ar 16000 {} -loglevel error -y'.format(wav_path, wav_16000_path)
        print(cmd)
        # import pdb; pdb.set_trace()
        os.system(cmd)


def copy_wav2dir():
    wav_root = '/data/yashengsun/local_storage/Video_Speech_Actor_wav_16000/Audio_Speech_Actors_01-24'
    dir_root = '/data/yashengsun/local_storage/Video_Speech_Actor_fps25_emote'

    subdirs = os.listdir(dir_root)
    subdirs = [os.path.join(dir_root, p) for p in subdirs]
    all_subsubpaths = []
    for subdir  in subdirs:
        subsubdirs = os.listdir(subdir)
        subsubpaths = [os.path.join(subdir, subsubdir) for subsubdir in subsubdirs]
        all_subsubpaths += subsubpaths

    for subsubpath in all_subsubpaths:
        index_wav_path = subsubpath.replace(dir_root, wav_root)
        index_wav_path = index_wav_path.replace('/02-', '/03-').replace('/01-', '/03-') + '.wav'
        dest_path = os.path.join(subsubpath, os.path.basename(subsubpath)+'.wav')

        cmd = 'cp -r {} {}'.format(index_wav_path, dest_path)
        print(cmd)
        # import pdb; pdb.set_trace()
        os.system(cmd)


if __name__ == '__main__':
    # main()
    copy_wav2dir()
