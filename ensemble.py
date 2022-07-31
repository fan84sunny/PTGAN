import numpy as np
import os
'''
@inproceedings{luo2021empirical,
 title={An Empirical Study of Vehicle Re-Identification on the AI City Challenge},
 author={Luo, Hao and Chen, Weihua and Xu Xianzhe and Gu Jianyang and Zhang, Yuqi and Chong Liu and Jiang Qiyi and He, Shuting and Wang, Fan and Li, Hao},
 booktitle={Proc. CVPR Workshops},
 year={2021}
}
'''
distmat_paths = [
        './logs/stage2/resnext101a_384/v1/dist_mat.npy',
        './logs/stage2/resnext101a_384/v2/dist_mat.npy',

        './logs/stage2/101a_384/v1/dist_mat.npy',
        './logs/stage2/101a_384/v2/dist_mat.npy',

        './logs/stage2/101a_384_recrop/v1/dist_mat.npy',
        './logs/stage2/101a_384_recrop/v2/dist_mat.npy',

        './logs/stage2/101a_384_spgan/v1/dist_mat.npy',
        './logs/stage2/101a_384_spgan/v2/dist_mat.npy',

        './logs/stage2/densenet169a_384/v1/dist_mat.npy',
        './logs/stage2/densenet169a_384/v2/dist_mat.npy',

        './logs/stage2/s101_384/v1/dist_mat.npy',
        './logs/stage2/s101_384/v2/dist_mat.npy',

        './logs/stage2/se_resnet101a_384/v1/dist_mat.npy',
        './logs/stage2/se_resnet101a_384/v2/dist_mat.npy',

        './logs/stage2/transreid_256/v1/dist_mat.npy',
        './logs/stage2/transreid_256/v2/dist_mat.npy',
        ]
data = np.load(distmat_paths[0])
distmat = np.zeros((data.shape[0],data.shape[1]))
for i in distmat_paths:
    distmat += np.load(i)

sort_distmat_index = np.argsort(distmat, axis=1)

print('The shape of distmat is: {}'.format(distmat.shape))
save_path = './track2_veri.txt'
# with open(save_path,'w') as f:
#     for item in sort_distmat_index:
#         for i in range(99):
#             f.write(str(item[i] + 1).zfill(6)+'.jpg' + ' ')
#         f.write(str(item[i+1] + 1).zfill(6)+'.jpg' + '\n')

