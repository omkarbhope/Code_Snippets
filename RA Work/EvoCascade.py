import torch.nn as nn
import numpy as np

class CascadeModule(nn.Module):
    def __init__(self, cascade):
        super(CascadeModule, self).__init__()
        self.cascade = nn.ModuleList(cascade)
        self.re_order_indices= [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]

    def normalize(self, skeleton, re_order=None):
        norm_skel = skeleton.copy()
        if re_order is not None:
            norm_skel = norm_skel[re_order].reshape(32)
        norm_skel = norm_skel.reshape(16, 2)
        mean_x = np.mean(norm_skel[:,0])
        std_x = np.std(norm_skel[:,0])
        mean_y = np.mean(norm_skel[:,1])
        std_y = np.std(norm_skel[:,1])
        denominator = (0.5*(std_x + std_y))
        norm_skel[:,0] = (norm_skel[:,0] - mean_x)/denominator
        norm_skel[:,1] = (norm_skel[:,1] - mean_y)/denominator
        norm_skel = norm_skel.reshape(32)         
        return norm_skel
        
    def forward(self, data):
        # data = self.normalize(data)
        num_stages = len(self.cascade)
        # for legacy code that does not have the num_blocks attribute
        for i in range(len(self.cascade)):
            self.cascade[i].num_blocks = len(self.cascade[i].res_blocks)
        prediction = self.cascade[0](data)
        # prediction for later stages
        for stage_idx in range(1, num_stages):
            prediction += self.cascade[stage_idx](data)
        return prediction
