import torch


def calc_loop_idx(idx, loop_num):
    flag = -1 * ((idx // loop_num % 2) * 2 - 1)
    new_idx = -flag * (flag - 1) // 2 + flag * (idx % loop_num)
    return (new_idx + loop_num) % loop_num


def loopback_frames(img, frame_num):
    loop_num = img.shape[0]
    new_img = []
    for i in range(frame_num):
        new_idx = calc_loop_idx(i, loop_num=loop_num)
        new_img.append(img[new_idx])
    new_img = torch.stack(new_img, dim=0)
    return new_img