import argparse
import os, sys

import code_GAN.gan.utils.util as util
from code_GAN.reid import models
from code_GAN.reid import datasets


class Options(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--stage', type=int, default=1, help='training stage [1|2]')
        self.parser.add_argument('-d', '--dataset', type=str, default='train', choices=datasets.names())
        # paths
        self.parser.add_argument('--dataroot', type=str, default='../veri_pose/',
                                 help='root path to datasets')
        self.parser.add_argument('--checkpoints', type=str, default='./weights/', help='root path to save models')
        self.parser.add_argument('--name', type=str, default='GAN_stage_2', help='directory to save models')
        self.parser.add_argument('--netE_pretrain', type=str,
                                 default='/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AICITY2021_Track2_DMT-main/code_GAN/gan/weights/model_130.pth')
        self.parser.add_argument('--netG_pretrain', type=str,
                                 default='/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AICITY2021_Track2_DMT-main/code_GAN/weights/GAN_stage_2/17_net_G.pth')
        self.parser.add_argument('--netDp_pretrain', type=str,
                                 default='/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AICITY2021_Track2_DMT-main/code_GAN/weights/GAN_stage_2/17_net_Dp.pth')
        self.parser.add_argument('--netDi_pretrain', type=str,
                                 default='/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AICITY2021_Track2_DMT-main/code_GAN/weights/GAN_stage_2/17_net_Di.pth')
        # self.parser.add_argument('--netE_pretrain', type=str, default='weights/Encoder/model_140.pth')
        # self.parser.add_argument('--netG_pretrain', type=str, default='weights/GAN_stage_2/17_net_G.pth')
        # self.parser.add_argument('--netDp_pretrain', type=str, default='weights/GAN_stage_2/17_net_Dp.pth')
        # self.parser.add_argument('--netDi_pretrain', type=str, default='weights/GAN_stage_2/17_net_Di.pth')
        # model structures
        self.parser.add_argument('--arch', type=str, default='resnet50', choices=models.names())
        self.parser.add_argument('--norm', type=str, default='batch', help='batch normalization')
        self.parser.add_argument('--drop', type=float, default=0.0, help='dropout for the netG')
        self.parser.add_argument('--connect_layers', type=int, default=0, help='skip connections num for netG')
        self.parser.add_argument('--fuse_mode', type=str, default='cat', help='fuse reid and pose feature [cat|add]')
        self.parser.add_argument('--id_class', type=int, default=575, help='ID class for train dataset')
        self.parser.add_argument('--pose_feature_size', type=int, default=256, help='length of feature vector for pose')
        self.parser.add_argument('--noise_feature_size', type=int, default=128, help='noise dimension')
        self.parser.add_argument('--pose_aug', type=str, default='gauss', help='posemap augmentation [no|erase|gauss]')
        # dataloader setting
        self.parser.add_argument('-b', '--batch_size', type=int, default=12, help='input batch size')
        self.parser.add_argument('-j', '--workers', default=8, type=int, help='num threads for loading data')
        self.parser.add_argument('--width', type=int, default=224, help='input image width')
        self.parser.add_argument('--height', type=int, default=224, help='input image height')
        # optimizer setting
        self.parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=20, help='decay learning rate to zero')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
        self.parser.add_argument('--save_step', type=int, default=1, help='frequency of saving weights')
        self.parser.add_argument('--eval_step', type=int, default=10, help='frequency of evaluate')
        # visualization setting
        self.parser.add_argument('--display_port', type=int, default=8098, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=1,
                                 help='window id of the web display, set 0 for non-usage of visdom')
        self.parser.add_argument('--display_winsize', type=int, default=224,  help='display window size')
        self.parser.add_argument('--env', type=str, default='GAN_train-stage-2', help='display env name')

        self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of showing training results')
        self.parser.add_argument('--display_single-pane-ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain number'
                                      'of images per row.')
        self.parser.add_argument('--update_html-freq', type=int, default=100, help='saving training results to html')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save training results to [opt.checkpoints]/name/web/')
        self.parser.add_argument('--print_freq', type=int, default=10,
                                 help='frequency of showing training results on console')
        # training schedule
        self.parser.add_argument('--lambda_recon', type=float, default=100.0, help='loss weight of loss_recon')
        self.parser.add_argument('--lambda_orthogonal', type=float, default=0.0, help='loss weight of loss_orthogonal')
        self.parser.add_argument('--smooth_label', action='store_true', default=True, help='smooth label for GANloss')

        self.opt = self.parser.parse_args()
        self.show_opt()

    def parse(self):
        return self.opt

    def show_opt(self):
        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
