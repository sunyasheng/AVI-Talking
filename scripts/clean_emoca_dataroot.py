import os
from tqdm import tqdm

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


def main():
    root = '/data/yashengsun/local_storage/paishe_w_cam'
    paths = find_dir_w_suffix(root, '.png')
    print(len(paths))
    leave_paths = [p for p in paths if '/detections/' in p]
    for p in tqdm(paths):
        if os.path.exists(p):
            if not p in leave_paths:
                os.system('rm -rf {}/*.png'.format(os.path.dirname(p)))
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
