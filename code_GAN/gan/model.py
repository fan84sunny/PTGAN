from collections import OrderedDict
from .utils import util
from .networks import get_norm_layer, init_weights, CustomPoseGenerator, NLayerDiscriminator, remove_module_key, \
    set_bn_fix, get_scheduler, print_network, OrthogonalEncoder, IDDiscriminator
# from ..gan.networks import get_norm_layer, init_weights, CustomPoseGenerator, NLayerDiscriminator, remove_module_key, \
#     set_bn_fix, get_scheduler, print_network, OrthogonalEncoder, IDDiscriminator
# from ..gan.gan_losses import GANLoss
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.kernel_size = kernel_size
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp((-((mgrid - mean) / std) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, landmark):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        maps = torch.zeros(len(landmark), 20, 224, 224).to("cuda")
        value = torch.min(landmark, dim=2).values
        flag = value != -1
        index = landmark[flag].cpu().detach().numpy().astype(np.int)
        # index = index
        # maps[flag][:, index[:, 0], index[:, 1]] = 1
        # maps[flag][:, index[:, 0], index[:, 1]] = tmp
        tmp = maps[flag]
        for i, (x, y) in enumerate(index):
            tmp[i, x, y] = 1
        maps[flag] = tmp
        maps = self._generate_pose_map_tensor(maps)
        m = torch.max(maps.view(len(landmark), 20, -1), dim=2).values
        maps[flag] = maps[flag] / m[flag][:, None, None]
        # maps = maps / maps.max()
        # maps = np.stack(maps, axis=0)
        # map_list.append(maps)
        return maps

    def _generate_pose_map_tensor(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups, padding=10)
        # map_list = []
        # # landmark = torch.Tensor(landmark_list)
        # for landmark in landmark_list:
        #     maps = []
        #     randnum = landmark.size(0)+1
        #     # gauss_sigma = random.randint(gauss_sigma-1, gauss_sigma+1)
        #     for i in range(landmark.size(0)):
        #         map = torch.zeros(224, 224)
        #         print(landmark.shape)
        #         print(landmark[i, 0])
        #         if landmark[i, 0] != -1 and landmark[i, 1] != -1 and i != randnum:
        #             map[landmark[i, 0], landmark[i, 1]] = 1
        #             map = Smoothing(map)
        #             map = map / map.max()
        #         maps.append(map)
        #     maps = np.stack(maps, axis=0)
        #     map_list.append(maps)
        # return map_list


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # self.opt = opt
        # self.save_dir = os.path.join(opt.checkpoints, opt.name)
        self.norm_layer = get_norm_layer(norm_type="batch")
        self.device = "cuda"

        self._init_models()
        print('---------- Networks initialized -------------')
        print_network(self.net_E)
        print_network(self.net_G)
        print_network(self.net_Di)
        print_network(self.net_Dp)
        print('-----------------------------------------------')

    def _init_models(self):
        self.net_G = CustomPoseGenerator(256, 2048, 128,
                                         dropout=0.0, norm_layer=self.norm_layer, fuse_mode='cat',
                                         connect_layers=0)
        self.net_E = OrthogonalEncoder()
        self.net_Di = IDDiscriminator(575)
        self.net_Dp = NLayerDiscriminator(3+20, norm_layer=self.norm_layer)

        self._load_state_dict(self.net_E, '/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AICITY2021_Track2_DMT-main/code_GAN/weights/GAN_stage_2/17_net_E.pth')
        self._load_state_dict(self.net_G, '/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AICITY2021_Track2_DMT-main/code_GAN/weights/GAN_stage_2/17_net_G.pth')
        self._load_state_dict(self.net_Di, '/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AICITY2021_Track2_DMT-main/code_GAN/weights/GAN_stage_2/17_net_Di.pth')
        self._load_state_dict(self.net_Dp, '/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AICITY2021_Track2_DMT-main/code_GAN/weights/GAN_stage_2/17_net_Dp.pth')

        self.net_E = nn.DataParallel(self.net_E).to(self.device)
        self.net_G = nn.DataParallel(self.net_G).to(self.device)
        self.net_Di = nn.DataParallel(self.net_Di).to(self.device)
        self.net_Dp = nn.DataParallel(self.net_Dp).to(self.device)

    def reset_model_status(self):
        self.net_G.eval()
        self.net_Dp.eval()
        self.net_E.eval()
        self.net_Di.eval()

    def _load_state_dict(self, net, path):
        state_dict = remove_module_key(torch.load(path))
        net.load_state_dict(state_dict)

    def enocder(self, img):
        noise = torch.randn(img.shape[0], 128)
        # noise = noise.to('cuda')
        outputs = self.net_E(img)
        id_feature = outputs[1].view(outputs[0].size(0), outputs[0].size(1), 1, 1)
        return noise, id_feature, torch.argmax(outputs[2], dim=1)

    def generate(self, query_img, target_pose):
        noise, id_feature, q_pose = self.enocder(query_img)
        # pose_feature, id_feature, pose, id, color, type
        fake = self.net_G(target_pose, id_feature, noise.view(noise.size(0), noise.size(1), 1, 1))
        return fake

    def get_pose_type(self, img):
        # pose_feature, id_feature, pose, id, color, type
        outputs = self.net_E(img)
        return outputs[2], outputs[5]

    def get_current_visuals(self):
        input = util.tensor2im(self.origin)
        target = util.tensor2im(self.target)
        fake = util.tensor2im(self.fake)
        map = self.posemap.sum(1)
        map[map>1] = 1
        map = util.tensor2im(torch.unsqueeze(map, 1))
        return OrderedDict([('input', input), ('posemap', map), ('fake', fake), ('target', target)])
