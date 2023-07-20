import cv2
import pickle
import torch
import os
import numpy as np
from EMOCA.blendshape_visualizer import Visualizer3DMM


def read_pickle(path):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


def save_frames2video(frames, video_path, fps=25.0):
    frame_width = frames[0].shape[1]
    frame_height = frames[0].shape[0]

    # Initialize the video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        video_writer.write(frame)


class CustomVisualizer:
    def __init__(self, checkpoint_path):
        super(CustomVisualizer, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.visualizer = Visualizer3DMM(checkpoint_path)

    def visualize_exp(self, exp, pose, shape=None, save_root='./results_vis', save_name='render', driven_folder=None):

        rendered_img = self.visualizer(exp, pose, shape)
        rendered_img = rendered_img.permute(0,2,3,1)*255.
        rendered_img_nps = [rendered_img[i].cpu().numpy().astype(np.uint8) for i in range(len(rendered_img))]
        rendered_img_nps = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in rendered_img_nps]
        # import pdb; pdb.set_trace()
        save_path = os.path.join(save_root, save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # torchvision.utils.save_image(rendered_img / 255., '{}.jpg'.format(save_path))

        if driven_folder is None:
            cat_img_nps = rendered_img_nps
        else:
            resize = (rendered_img_nps[0].shape[1], rendered_img_nps[1].shape[0])
            driven_imgs = read_upper_face_folder(driven_folder, len(rendered_img_nps), resize)
            cat_img_nps = [np.hstack([driven_img, render_img]) for driven_img, render_img in zip(driven_imgs, rendered_img_nps)]
        
        save_frames2video(cat_img_nps, save_path + '.mp4')
        print('save results to {}'.format(save_path+'.mp4'))

if __name__ == '__main__':
    # head_visualizer = Visualizer3DMM()
    checkpoint_path = 'BlendshapeVisualizer/checkpoints/BFM/'
    visualizer = CustomVisualizer(checkpoint_path=checkpoint_path)

    dataset_path = 'datadict_train_paishe.pkl'
    data_dict = read_pickle(dataset_path)
    ceoff = data_dict['transpose_crop_MVI_0030_000']['exp'][:200]
    pose = data_dict['transpose_crop_MVI_0030_000']['pose'][:200]
    # import pdb; pdb.set_trace();
    
    # ceoff = np.load(coeff_path)
    ceoff = torch.from_numpy(ceoff)
    pose = torch.from_numpy(pose)
    os.makedirs('save_results', exist_ok=True)
    with torch.no_grad():
        pose[:, :3] = 0.0
        # pose[:,:1] = 0.9 # upper view
        pose[:,1:2] = 1.1 # turn right
        # pose[:,2:3] = 0.9 # rotate left
        visualizer.visualize_exp(ceoff, pose, shape=None, save_name='save_results', driven_folder=None)

    import pdb; pdb.set_trace();