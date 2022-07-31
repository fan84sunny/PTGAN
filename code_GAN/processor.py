import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# import cv2
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
                h0 = int(float(line) * 224/size[0])
                if h0 < 0:
                    h0 = -1
            else:
                w0 = int(float(line) * 224/size[1])
                if w0 < 0:
                    w0 = -1
                landmark.append(torch.Tensor([[w0, h0]]))
        landmark = torch.cat(landmark).long()
        # avoid to over fit
        ram = random.randint(0, 19)
        landmark[ram][0] = random.randint(0, 224-1)
        landmark[ram][1] = random.randint(0, 224-1)
        landmark_list.append(landmark)
    return landmark_list


def file2pose_map(landmark_dict, gauss_sigma=5):
    pose_map_dict = dict()
    for landmark_key in tqdm.tqdm(landmark_dict):
        landmark = landmark_dict[landmark_key]
        maps = []
        randnum = landmark.size(0)+1
        gauss_sigma = random.randint(gauss_sigma-1, gauss_sigma+1)
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
        randnum = landmark.size(0)+1
        gauss_sigma = random.randint(gauss_sigma-1, gauss_sigma+1)
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


def do_inference(cfg, model, query_loader, gallery_loader, gallery_pose_list):

    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    device = "cuda"
    # if device:
    #     if torch.cuda.device_count() > 1:
    #         print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
    #         model = nn.DataParallel(model)
    #     model.to(device)

    model.eval()
    # feature_dis = np.zeros((len(query_loader.dataset), len(gallery_loader.dataset)), dtype=np.float)
    Smoothing = GaussianSmoothing(20, 21, 5)
    Smoothing = nn.DataParallel(Smoothing).to(device)
    model = model.to(device)
    reid_model = make_model(cfg, num_class=1678)
    reid_model.load_param(cfg.TEST.WEIGHT)
    reid_model.to("cuda")
    reid_model.eval()
    q = query_loader.dataset
    g = gallery_loader.dataset
    val_list = q.dataset + g.dataset
    val_loader = DataLoader(ImageDataset(val_list, height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1]), batch_size=12, num_workers=4, pin_memory=False)
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

    img_path_list = []

    for n_iter, input in enumerate(val_loader):
        with torch.no_grad():
            # img_paths = imgpath[::2]
            img = input['origin'].to(device)
            if cfg.TEST.FLIP_FEATS == 'on':
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                        f1 = reid_model(img)
                    else:
                        f2 = reid_model(img)
                feat = f2 + f1
            else:
                feat = reid_model(img)
            evaluator.update((feat.clone(), input['pid'], input['camid'], input['trackid']))
            img_path_list.extend(input['file_name'])
    distmat, P, neg_vec = evaluator.compute(fic=cfg.TEST.FIC, fac=cfg.TEST.FAC, rm_camera=cfg.TEST.RM_CAMERA,
                                save_dir=cfg.OUTPUT_DIR, crop_test=cfg.TEST.CROP_TEST, la=cfg.TEST.LA)
    ori_cmc, ori_mAP, _, pids, _, _, _ = evaluator.calcu_R1_mAP(distmat)

    logger.info("Ori Validation Results - Epoch: ")
    logger.info("mAP: {:.1%}".format(ori_mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, ori_cmc[r - 1]))

    # _, _, _, pids, _, _, _ = evaluator.calcu_R1_mAP(distmat)
    np.save(os.path.join(cfg.OUTPUT_DIR, "original.npy"), distmat)
    # Compute Gener_query to Original_gallery distance matrix

    evaluator_gen = R1_mAP_eval_gen(P, neg_vec, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING,
                            dataset=cfg.DATASETS.NAMES, reranking_track=cfg.TEST.RE_RANKING_TRACK)

    gen_distmat = np.zeros(distmat.shape)

    start = time.time()
    landmark_dict = load_all_landmark(gallery_pose_list)
    # pose_map_dict = file2pose_map(landmark_dict)
    print(time.time()-start)
    count = 0
    with torch.no_grad():
        for batch, gallery_data in enumerate(tqdm.tqdm(gallery_loader)):
            gallery_img = gallery_data['origin'].to(device)
            pose, type = model.get_pose_type(gallery_img)
            gallery_poseid = torch.argmax(pose, dim=1)
            gallery_typeid = torch.argmax(type, dim=1)
            if gallery_typeid == 4 and gallery_poseid == 2:
                gallery_typeid += 1
                count += 1
            # print(gallery_poseid)
            # print(gallery_poseid.shape)
            img = gallery_img

            #### this place is generate gallery img #####
            pose_file = gallery_pose_list[gallery_typeid[0]][gallery_poseid[0]][0]
            landmark_tensor = landmark_dict[pose_file].view(1, 20, 2)
            landmark_tensor = landmark_tensor.to(device)
            pose_maps = Smoothing(landmark_tensor)
            img = model.generate(img, pose_maps)
            #######
            img = F.interpolate(img, size=cfg.INPUT.SIZE_TEST[0])
            # if cfg.MODEL.NAME == 'transformer':
            #     img = F.interpolate(img, size=cfg.INPUT.SIZE_TEST[0])
            for i in range(2):
                if i == 1:
                    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                    img = img.index_select(3, inv_idx)
                    f1 = reid_model(img)
                else:
                    f2 = reid_model(img)

            gallery_features = dict()
            gallery_features['gf'] = f2 + f1
            gallery_features['g_camids'] = gallery_data['camid']
            gallery_features['gallery_tids'] = gallery_data['trackid']
            evaluator_gen.reset(gallery_features)

            for query_data in query_loader:
                query_img = query_data['origin'].to(device)
                pose_file = [gallery_pose_list[gallery_typeid[0]][gallery_poseid[0]][0] for _ in range(query_img.shape[0])]

                # print(query_img)
                # start = time.time()
                landmark_tensor = torch.zeros((query_img.shape[0], 20, 2))
                # landmark = [landmark_dict[file] for file in pose_file]
                for c, file in enumerate(pose_file):
                    landmark_tensor[c] = landmark_dict[file]
                landmark_tensor = landmark_tensor.to(device)
                # landmark = _load_landmark(pose_file)
                pose_maps = Smoothing(landmark_tensor)
                # pose_maps = [pose_map_dict[file] for file in pose_file]
                # pose_maps = torch.Tensor(pose_maps).to(device)
                # print('load pose map: ', time.time() - start)

                gen_query2gal = model.generate(query_img, pose_maps)

                # np_imgs = gen_query2gal.detach().cpu().numpy()
                # for img1 in np_imgs:
                #     img1 = img1 * 0.5 + 0.5
                #     q = np.transpose(img1, (1, 2, 0))
                #     q = Image.fromarray((q * 255).astype('uint8')).convert('RGB')
                #     q.show()

                img = gen_query2gal.to(device)
                img = F.interpolate(img, size=cfg.INPUT.SIZE_TEST[0])

                # if cfg.MODEL.NAME == 'transformer':
                #     img = F.interpolate(img, size=cfg.INPUT.SIZE_TEST[0])
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                        f1 = reid_model(img)
                    else:
                        f2 = reid_model(img)
                feat = f1 + f2
                evaluator_gen.update((feat.clone(), query_data['camid'], query_data['trackid']))
            # print(gen_distmat[..., batch].shape)
            k = evaluator_gen.compute()
            # print(k.shape)
            gen_distmat[..., batch] = k[..., 0]
        # np.save(os.path.join(cfg.OUTPUT_DIR, "gen_distmat.npy"), distmat)
        gen_cmc, gen_mAP, _, gen_pids, _, _, _ = evaluator.calcu_R1_mAP(gen_distmat)
        logger.info("GEN Validation Results - Epoch: ")
        logger.info("mAP: {:.1%}".format(gen_mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, gen_cmc[r - 1]))
        np.save(os.path.join(cfg.OUTPUT_DIR, "gen_distmat.npy"), gen_distmat)

        g_pids = pids[num_query:]
        q_pids = pids[:num_query]
        with open('/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AICITY2021_Track2_DMT-main/logs/stage2/101a_384/veri_gan/pid_gen.txt', 'a+') as f:
            f.write(",")
            for i, garrley in enumerate(g_pids):
                f.write(str(garrley))
                if i < len(g_pids) - 1:
                    f.write(',')
            f.write('\n')
            for i, query in enumerate(q_pids):
                f.write(str(query))
                for dis in gen_distmat[i]:
                    f.write(',' + str(dis))
                f.write('\n')

        distmat += gen_distmat * 0.5
        np.save(os.path.join(cfg.OUTPUT_DIR, "TEST_DIST_MAT_py.npy"), distmat)
        cmc, mAP, distmat, pids, camids, qf, gf = evaluator.calcu_R1_mAP(distmat)

        logger.info(cfg.OUTPUT_DIR)

        logger.info("Ori Validation Results - Epoch: ")
        logger.info("mAP: {:.1%}".format(ori_mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, ori_cmc[r - 1]))

        logger.info("ori+gen Validation Results - Epoch: ")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        print(count)
        """
        # show image
        
        np_imgs = gen_query2gal.detach().numpy()
        np_ori_imgs = query_img.detach().numpy()
        
        for img1, img2, name in zip(np_imgs, np_ori_imgs, query_data['file_name']):
            img1 = img1 * 0.5 + 0.5
            q = np.transpose(img1, (1, 2, 0))
            q = Image.fromarray((q * 255).astype('uint8')).convert('RGB')

            img2 = img2 * 0.5 + 0.5
            g = np.transpose(img2, (1, 2, 0))
            g = Image.fromarray((g*255).astype('uint8')).convert('RGB')
            print(name)
            g.show()
            q.show()
        """
