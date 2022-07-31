import glob
from torch.utils.data import Dataset
from code_GAN.reid.utils.data import transforms
from PIL import Image
import random
import torch
import numpy as np
from scipy import ndimage
from config import cfg

def _pluck(root, query):
    ret = []
    pose_list = [[[] for i in range(8)] for j in range(9)]
    if query:
        path = glob.glob(root + '/*.jpg')
        index = -1
        for fname in path:
            pid = int(fname[-24:-20])
            # if index != pid:
            #     index += 1
            camid = int(fname[-18:-15])
            ret.append((fname, pid, camid, -1))
    else:
        path = glob.glob(root + '/*/*/*.jpg')
        frame2trackID = dict()
        with open(root + '/../test_track.txt') as f:
            for track_id, line in enumerate(f.readlines()):
                curLine = line.strip().split(" ")
                for frame in curLine:
                    frame2trackID[frame] = track_id

        for fname in path:
            pid = int(fname[-24:-20])
            camid = int(fname[-18:-15])
            ret.append((fname, pid, camid, frame2trackID[fname[-24:]]))
    return ret
# /home/ANYCOLOR2434/AICITY2021_Track2_DMT/AIC21/veri_pose/test/008_4_0/3/0038_c008_00008270_0.txt


def get_pose_list(root):
    pose_list = [[[] for i in range(8)] for j in range(9)]
    path = glob.glob(root + '/*/*/*.jpg')

    for fname in path:
        type = int(fname[-28])
        pose = int(fname[-26])
        pose_file = fname[:-4] + '.txt'
        pose_list[type][pose].append(pose_file)
    return pose_list


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, height=256, width=256, pose_aug='no'):
        self.height = height
        self.width = width
        self.dataset = dataset
        self.transform = transform
        self.pose_aug = pose_aug

        normalizer = transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        # normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        if transform is None:
            self.transform = transforms.Compose([
                                 transforms.RectScale(height, width),
                                 transforms.ToTensor(),
                                 normalizer,
                             ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._get_single_item_with_pose(index)

    def _get_single_item(self, index):
        fname, pid, pose, camid, color, type = self.dataset[index]
        fpath = fname
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)
        return img, fname, pid, camid, color, type

    def _get_single_item_with_pose(self, index):
        fname, pid, camid, trackid = self.dataset[index]
        fpath = fname
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)

        return {'origin': img,
                'pid': pid,
                'camid': camid,
                'trackid': trackid,
                'file_name': fname
                }
