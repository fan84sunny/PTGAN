import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import cv2
import tqdm
import sys
from scipy import ndimage
import random
from PIL import Image
# from utils.meter import AverageMeter
from torch.utils.data import DataLoader

from code_GAN.metrics import R1_mAP_eval, R1_mAP_eval_gen
from torch.nn.parallel import DistributedDataParallel
from torch.cuda import amp
import torch.distributed as dist
# from IPython import embed
from code_GAN.PTGAN_dataset import ImageDataset
import model.make_model as make_model
from code_GAN.gan.model import GaussianSmoothing


def frozen_feature_layers(model):
    for name, module in model.named_children():
        if 'base' in name:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def open_all_layers(model):
    for name, module in model.named_children():
        module.train()
        for p in module.parameters():
            p.requires_grad = True


def load_all_landmark(gallery_pose_list):
    landmark_dict = dict()
    for type in gallery_pose_list:
        for gallery_pose in type:
            for file in gallery_pose:
                landmark = []
                with open(file, 'r') as f:
                    landmark_file = f.readlines()
                size = Image.open(file[:-4] + '.jpg').size
                for i, line in enumerate(landmark_file):
                    if i % 2 == 0:
                        h0 = int(float(line) * 224 / size[0])
                        if h0 < 0:
                            h0 = -1
                    else:
                        w0 = int(float(line) * 224 / size[1])
                        if w0 < 0:
                            w0 = -1
                        landmark.append(torch.Tensor([[w0, h0]]))
                landmark = torch.cat(landmark).long()
                # avoid to over fit
                ram = random.randint(0, 19)
                landmark[ram][0] = random.randint(0, 224 - 1)
                landmark[ram][1] = random.randint(0, 224 - 1)
                landmark_dict[file] = landmark
    return landmark_dict


def _load_landmark(path_list):
    landmark_list = []
    for file in path_list:
        landmark = []
        with open(file, 'r') as f:
            landmark_file = f.readlines()
        size = Image.open(file[:-4] + '.jpg').size
        for i, line in enumerate(landmark_file):
            if i % 2 == 0:
                h0 = int(float(line) * 224 / size[0])
                if h0 < 0:
                    h0 = -1
            else:
                w0 = int(float(line) * 224 / size[1])
                if w0 < 0:
                    w0 = -1
                landmark.append(torch.Tensor([[w0, h0]]))
        landmark = torch.cat(landmark).long()
        # avoid to over fit
        ram = random.randint(0, 19)
        landmark[ram][0] = random.randint(0, 224 - 1)
        landmark[ram][1] = random.randint(0, 224 - 1)
        landmark_list.append(landmark)
    return landmark_list


def file2pose_map(landmark_dict, gauss_sigma=5):
    pose_map_dict = dict()
    for landmark_key in tqdm.tqdm(landmark_dict):
        landmark = landmark_dict[landmark_key]
        maps = []
        randnum = landmark.size(0) + 1
        gauss_sigma = random.randint(gauss_sigma - 1, gauss_sigma + 1)
        for i in range(landmark.size(0)):
            map = np.zeros([224, 224])
            if landmark[i, 0] != -1 and landmark[i, 1] != -1 and i != randnum:
                map[landmark[i, 0], landmark[i, 1]] = 1
                map = ndimage.filters.gaussian_filter(map, sigma=gauss_sigma)
                map = map / map.max()
            maps.append(map)
        maps = np.stack(maps, axis=0)
        pose_map_dict[landmark_key] = maps
        print(sys.getsizeof(maps) / 1024)
    return pose_map_dict


def _generate_pose_map(landmark_list, gauss_sigma=5):
    map_list = []
    for landmark in landmark_list:
        maps = []
        randnum = landmark.size(0) + 1
        gauss_sigma = random.randint(gauss_sigma - 1, gauss_sigma + 1)
        for i in range(landmark.size(0)):
            map = np.zeros([224, 224])
            if landmark[i, 0] != -1 and landmark[i, 1] != -1 and i != randnum:
                map[landmark[i, 0], landmark[i, 1]] = 1
                map = ndimage.filters.gaussian_filter(map, sigma=gauss_sigma)
                map = map / map.max()
            maps.append(map)
        maps = np.stack(maps, axis=0)
        map_list.append(maps)
    return map_list


def do_inference(cfg, query_loader, gallery_loader, gallery_pose_list):
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    device = "cuda"
    # if device:
    #     if torch.cuda.device_count() > 1:
    #         print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
    #         model = nn.DataParallel(model)
    #     model.to(device)

    q = query_loader.dataset
    g = gallery_loader.dataset
    val_list = q.dataset + g.dataset
    val_loader = DataLoader(ImageDataset(val_list, height=384, width=384), batch_size=12, num_workers=4,
                            pin_memory=False)
    num_query = len(q)

    # Compute Original_query to Original_gallery distance matrix
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    if cfg.TEST.EVAL:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING,
                                dataset=cfg.DATASETS.NAMES, reranking_track=cfg.TEST.RE_RANKING_TRACK)
    else:
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                           reranking=cfg.TEST.RE_RANKING, reranking_track=cfg.TEST.RE_RANKING_TRACK)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
    #         model = nn.DataParallel(model)

    for n_iter, input in enumerate(val_loader):
        with torch.no_grad():
            # img_paths = imgpath[::2]
            evaluator.update2((input['pid'], input['camid'], input['trackid']))
    distmat = np.load('/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AICITY2021_Track2_DMT-main/logs/stage2/101a_384/veri_gan/gen_distmat.npy')
    ori_cmc, ori_mAP, _, pids, _, _, _ = evaluator.calcu_R1_mAP(distmat)
    logger.info("mAP: {:.1%}".format(ori_mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, ori_cmc[r - 1]))
