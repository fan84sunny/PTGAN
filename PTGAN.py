import os, sys
import os.path as osp
import os

from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
# from processor import do_inference
from utils.logger import setup_logger


import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from code_GAN.PTGAN_dataset import ImageDataset, _pluck, get_pose_list
from code_GAN.reid import datasets
from code_GAN.reid.utils.data.preprocessor import Preprocessor
from code_GAN.reid.utils.data.sampler import RandomPairSampler
from code_GAN.reid.utils.data import transforms as T
from code_GAN.reid.evaluators import CascadeEvaluator
from code_GAN.processor import do_inference
from code_GAN.gan.options import Options
from code_GAN.gan.utils.visualizer import Visualizer
from code_GAN.gan.model import Model

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data(data_dir, workers):
    query_root = osp.join(data_dir, 'query')
    train_root = osp.join(data_dir, 'train')
    gallery_root = osp.join(data_dir, 'test')
    query_data = _pluck(query_root, True)
    gallery_data = _pluck(gallery_root, False)
    gallery_pose_list = get_pose_list(train_root)
    query_dataset = ImageDataset(query_data)
    gallery_dataset = ImageDataset(gallery_data)
    # use combined trainval set for training as default
    query_loader = DataLoader(query_dataset, batch_size=64, num_workers=8, pin_memory=True)
    gallery_loader = DataLoader(gallery_dataset, batch_size=1, num_workers=1, pin_memory=True)

    return query_loader, gallery_loader, gallery_pose_list


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)
    # model = make_model(cfg, num_class=num_classes)
    # model.load_param(cfg.TEST.WEIGHT)

    query_loader, gallery_loader, gallery_pose_list = get_data('/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AIC21/veri_pose', 4)

    model = Model()
    model.reset_model_status()
    do_inference(cfg, model, query_loader, gallery_loader, gallery_pose_list)


if __name__ == '__main__':
    main()
