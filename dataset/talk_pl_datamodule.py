import numpy as np
import torch

from third_party.motion_latent_diffusion.mld.data.base import BASEDataModule
from dataset.data_loader import TalkDataset
from omegaconf import OmegaConf
from os.path import join as pjoin

class TalkCollateFn():
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, batch):
        # audiobatch = [b[0] for b in batch]
        # coeffbatch = [b[1] for b in batch]
        posebatch = [b[2] for b in batch]
        # shapebatch = [b[3] for b in batch]
        # cambatch = [b[4] for b in batch]
        # motiondescbatch = [torch.from_numpy(b[5]) for b in batch]
        imgbatch = [b[6] for b in batch]
        ref_imgbatch = [b[7] for b in batch]
        textbatch =[b[9] for b in batch]

        lenbatch = [len(imgbatch[i]) for i in range(len(imgbatch))]

        imgbatches = []
        for i in range(len(imgbatch)):
            imgbatch_new = torch.zeros(size=(self.seq_len,*imgbatch[0].shape[1:])).to(imgbatch[i])
            imgbatch_new[:imgbatch[i].shape[0]] = imgbatch[i]
            imgbatches.append(imgbatch_new)
        imgbatch = torch.stack(imgbatches)

        ref_imgbatches = []
        for i in range(len(ref_imgbatch)):
            ref_imgbatch_new = torch.zeros(size=(self.seq_len,*ref_imgbatch[0].shape[1:])).to(ref_imgbatch[i])
            ref_imgbatch_new[:ref_imgbatch[i].shape[0]] = ref_imgbatch[i]
            ref_imgbatches.append(ref_imgbatch_new)
        ref_imgbatch = torch.stack(ref_imgbatches)

        posebatches = []
        for i in range(len(posebatch)):
            posebatch_new = torch.zeros(size=(self.seq_len,*posebatch[0].shape[1:])).to(posebatch[i])
            posebatch_new[:posebatch[i].shape[0]] = posebatch[i]
            posebatches.append(posebatch_new)
        posebatch = torch.stack(posebatches)
        # print('posebatch shape: ', posebatch.shape)
        adapted_batch = {
            "text": textbatch,
            # "motion": databatchTensor,
            "video": imgbatch, # for emotion contrastive and diffusion
            "ref_video": ref_imgbatch,
            "pose": posebatch[:,:,:3], # for head pose contrastive and diffusion
            "length": lenbatch
        }

        return adapted_batch

class TalkDataModule(BASEDataModule):

    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 # collate_fn=None,
                 phase="train",
                 dataset_config_path='config/mld/talkface/default_mld.yaml',
                 **kwargs):
        self.args = OmegaConf.load(dataset_config_path)
        collate_fn = TalkCollateFn(self.args.seq_length)
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = "talk_dataset"
        self.njoints = 30
        self.is4diffusion = True

        self.Dataset = TalkDataset#(args, is_inference=False)
        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        # self.nfeats = self._sample_set.nfeats
        self.nfeats = 30

    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        split_file = pjoin(
            eval(f"self.cfg.DATASET.{self.name.upper()}.SPLIT_ROOT"),
            self.cfg.EVAL.SPLIT + ".txt",
        )
        return self.Dataset(self.args, is4diffusion=self.is4diffusion)

    def __getattr__(self, item):
        # train_dataset/val_dataset etc cached like properties
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                # todo: config name not consistent
                subset = subset.upper() if subset != "val" else "EVAL"
                split = eval(f"self.cfg.{subset}.SPLIT")
                # import pdb; pdb.set_trace()
                split_file = pjoin(
                    eval(f"self.cfg.DATASET.{self.name.upper()}.SPLIT_ROOT"),
                    eval(f"self.cfg.{subset}.SPLIT") + ".txt",
                )
                self.__dict__[item_c] = self.Dataset(self.args, is4diffusion=self.is4diffusion)
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")
