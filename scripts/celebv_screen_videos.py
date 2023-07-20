## in this file, we select significant videos as training subset

import glob
import os
import pdb
import sys
from tqdm import tqdm
# from moviepy.editor import VideoFileClip
import pickle
import json
from collections import defaultdict
import shutil
import argparse

def get_video_length(video_path):
    """
    Get the duration of a video in seconds.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        float: Video duration in seconds.
    """
    video = VideoFileClip(video_path)
    duration = video.duration
    video.reader.close()
    video.audio.reader.close_proc()
    return duration


def get_video_paths():
    temporal_annotation_path = 'annotations.pkl'
    clip_annotation_path = 'celebvtext_info.json'

    temporal_annotation = pickle.load(open(temporal_annotation_path, 'rb'))
    clip_annotation = json.load(open(clip_annotation_path))
    print(len(temporal_annotation.keys()))
    print(len(clip_annotation.keys()))
    # import pdb; pdb.set_trace()
    # all_actions = set()
    # for seq_i in tqdm(temporal_annotation['act'].keys()):
    #     if '05PGXw-bSWg_' in seq_i:
    #         # import pdb; pdb.set_trace()
    #         print(seq_i)
    #     # print(temporal_annotation['act'][seq_i])
    #     actions = [act_info[0] for act_info in temporal_annotation['act'][seq_i]]
    #     # import pdb; pdb.set_trace()
    #     all_actions = all_actions | set(actions)
    #
    # print(all_actions)

    # valid_action = ['shout', 'sneer', 'squint', 'wink', 'turn', 'cry', 'sniff',  'shake_head', 'make_a_face',
    #                 'look_around', 'nod', 'laugh', 'close_eyes', 'smile', 'blink', 'sigh', 'sneeze',
    #                 'talk', 'weep', 'whisper', 'head_wagging', 'glare', 'frown', 'gaze']


    significant_action = ['wink', 'turn', 'sniff', 'shake_head',
                            'look_around', 'nod', 'laugh', 'close_eyes',
                            'smile', 'blink', 'sigh', #'sneeze',
                            'head_wagging',
                            'glare', 'frown', 'gaze']

    old_data_root = '/data/yashengsun/Proj/TalkingFace/CelebV-Text/downloaded_celebvtext/fps25_aligned_av'
    new_data_root = '/data/yashengsun/Proj/TalkingFace/CelebV-Text/downloaded_celebvtext/new_processed/fps25_aligned_av/'
    old_data_paths = sorted(glob.glob(os.path.join(old_data_root, '*.mp4')))
    new_data_paths = sorted(glob.glob(os.path.join(new_data_root, '*.mp4')))
    print(len(old_data_paths), len(new_data_paths))

    video_paths = old_data_paths + new_data_paths
    return video_paths


def get_specific_video_paths():
    root = '/data/yashengsun/Proj/TalkingFace/CelebV-Text/significant_subset_fps25'
    paths = glob.glob(os.path.join(root, '*.mp4'))
    paths = sorted(paths)
    return paths


def get_actions(temporal_annotation, id_name):
    fix_id_name = id_name
    try:
        actions = [act_info[0] for act_info in temporal_annotation['act'][id_name]]
    except:
        if id_name.lstrip('_').lstrip('-') in temporal_annotation['act']:
            fix_id_name = id_name.lstrip('_').lstrip('-')
        elif '-' + id_name.lstrip('_').lstrip('-') in temporal_annotation['act']:
            fix_id_name = '-' + id_name.lstrip('_').lstrip('-')
        elif '--' + id_name.lstrip('_').lstrip('-') in temporal_annotation['act']:
            fix_id_name = '--' + id_name.lstrip('_').lstrip('-')
        elif '---' + id_name.lstrip('_').lstrip('-') in temporal_annotation['act']:
            fix_id_name = '---' + id_name.lstrip('_').lstrip('-')
        elif '-' + id_name.lstrip('-').lstrip('-') in temporal_annotation['act']:
            fix_id_name = '-' + id_name.lstrip('-').lstrip('-')
        elif '--' + id_name.lstrip('-').lstrip('-') in temporal_annotation['act']:
            fix_id_name = '--' + id_name.lstrip('-').lstrip('-')
        elif '---' + id_name.lstrip('-').lstrip('-') in temporal_annotation['act']:
            fix_id_name = '---' + id_name.lstrip('-').lstrip('-')

        actions = [act_info[0] for act_info in temporal_annotation['act'][fix_id_name]]
    return fix_id_name, actions


# from moviepy.video.io.VideoFileClip import VideoFileClip

def clamp_op(input_path, output_path, start_time, end_time):
    # Load the video clip
    video_clip = VideoFileClip(input_path)

    # Clamp the video to the specified start and end times
    clipped_video = video_clip.subclip(start_time, end_time)

    # Write the clipped video to the output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clipped_video.write_videofile(output_path, codec='libx264', audio_codec='aac')


def clamp_video(have_action, video_path, temporal_annotation, clip_annotation, output_video_path):
    name = os.path.basename(video_path)
    ref_id, actions = get_actions(temporal_annotation, name.replace('.pkl', '').replace('.mp4.mp4',''))
    action_info = temporal_annotation['act'][ref_id]
    action_info = [action_info_i for action_info_i in action_info if action_info_i[0] == have_action ]
    one_action_info = action_info[0]
    clip_info = clip_annotation[ref_id + '.mp4']
    start_sec = clip_info['duration']['start_sec']
    one_action_time_info = one_action_info[1]
    action_start_sec = one_action_time_info[0].split(':')
    action_start_sec = int(action_start_sec[0]) * 3600 + int(action_start_sec[1]) * 60 + int(action_start_sec[2])
    action_start_sec = action_start_sec - start_sec
    action_start_sec = max(int(action_start_sec), 0)
    action_end_sec = action_start_sec + int(one_action_time_info[2])
    # import pdb; pdb.set_trace()

    clamp_op(video_path, output_video_path, action_start_sec, action_end_sec)


def main():
    significant_action = ['wink', 'turn', 'sniff', 'shake_head',
                            'look_around', 'nod', 'laugh', 'close_eyes',
                            'smile', 'blink', 'sigh', #'sneeze',
                            'head_wagging',
                            'glare', 'frown', 'gaze']

    temporal_annotation_path = 'annotations.pkl'
    clip_annotation_path = 'celebvtext_info.json'

    temporal_annotation = pickle.load(open(temporal_annotation_path, 'rb'))
    clip_annotation = json.load(open(clip_annotation_path))

    # video_paths = get_video_paths()
    video_paths = get_specific_video_paths()
    video_names = [os.path.basename(video_path) for video_path in video_paths]

    expected_max_actions = 500
    action_dict = defaultdict(int)
    significant_paths = []
    for i, (video_name, video_path) in tqdm(enumerate(zip(video_names, video_paths))):
        id_name = video_name[:-8]
        fix_id_name = id_name
        try:
            actions = [act_info[0] for act_info in temporal_annotation['act'][id_name]]
        except:
            if id_name.lstrip('_').lstrip('-') in temporal_annotation['act']:
                fix_id_name = id_name.lstrip('_').lstrip('-')
            elif '-' + id_name.lstrip('_').lstrip('-') in temporal_annotation['act']:
                fix_id_name = '-' + id_name.lstrip('_').lstrip('-')
            elif '--' + id_name.lstrip('_').lstrip('-') in temporal_annotation['act']:
                fix_id_name = '--' + id_name.lstrip('_').lstrip('-')
            elif '---' + id_name.lstrip('_').lstrip('-') in temporal_annotation['act']:
                fix_id_name = '---' + id_name.lstrip('_').lstrip('-')
            elif '-' + id_name.lstrip('-').lstrip('-') in temporal_annotation['act']:
                fix_id_name = '-' + id_name.lstrip('-').lstrip('-')
            elif '--' + id_name.lstrip('-').lstrip('-') in temporal_annotation['act']:
                fix_id_name = '--' + id_name.lstrip('-').lstrip('-')
            elif '---' + id_name.lstrip('-').lstrip('-') in temporal_annotation['act']:
                fix_id_name = '---' + id_name.lstrip('-').lstrip('-')
            else:
                print(id_name)
                continue
            actions = [act_info[0] for act_info in temporal_annotation['act'][fix_id_name]]

        try:
            duration = float(clip_annotation[fix_id_name + '.mp4']['duration']['end_sec']) - float(clip_annotation[fix_id_name + '.mp4']['duration']['start_sec'])
        except:
            import traceback
            traceback.print_exc()
            print(id_name)
            continue

        save_root = '/data/yashengsun/local_storage/save_root_{}'.format(expected_max_actions)
        have_actions = list(set(actions).intersection(set(significant_action)))
        if len(have_actions):
            # if duration < 14.0:
            #     print(actions, duration)
            for have_action in have_actions:
                action_dict[have_action] += 1
                if action_dict[have_action] > expected_max_actions:
                    significant_action.remove(have_action)
                significant_paths.append(video_path)
                # save_to_path = os.path.join(save_root, have_action, '{}.mp4'.format(action_dict[have_action]))
                save_to_path = os.path.join(save_root, have_action, video_name)
                try:
                    clamp_video(have_action, video_path, temporal_annotation, clip_annotation, output_video_path=save_to_path)
                    # import pdb; pdb.set_trace()
                except Exception as ex:
                    import traceback
                    traceback.print_exc()
                    # print(video_path, have_action)
                    continue

        # if i % 500 == 0: print(action_dict)
        # if i > 50000: break

    print(len(action_dict))
    import pdb; pdb.set_trace()

    # print(significant_paths)
    # significant_root = '/data/yashengsun/Proj/TalkingFace/CelebV-Text/significant_subset_fps25'
    # for video_path in significant_paths:
    #     shutil.copy(video_path, significant_root)


def delete_irrelevant():
    significant_root = '/data/yashengsun/local_storage/instruct_data/significant_subset_fps25_output'
    dir_names = os.listdir(significant_root)

    from celev_info import action_dict
    effective_names = []
    for v_names in action_dict.values():
        effective_names.extend(v_names)

    effective_significant_root = '/data/yashengsun/local_storage/instruct_data/head_dynamics'
    os.makedirs(effective_significant_root, exist_ok=True)

    # import pdb; pdb.set_trace()
    for i, dir_name in tqdm(enumerate(dir_names)):
        if dir_name in effective_names:
            cmd = 'cp -r {} {}'.format(os.path.join(significant_root, dir_name), effective_significant_root)
            print(cmd)
            os.system(cmd)
            # import pdb; pdb.set_trace()
        else:
            pass
    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='screen')
    args = parser.parse_args()

    if args.mode == 'screen':
        main()

    if args.mode == 'delete':
        delete_irrelevant()