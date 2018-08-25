import torch
from torch import nn
from torch.nn import functional as F
import utils
import numpy as np
import random


class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = utils.cuda(
                torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss


class MultiDiceLoss:
    def __init__(self, num_classes=1):
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = 0.
        smooth = 1.
        for cls in range(1, self.num_classes):
            jaccard_target = (targets == cls).float()
            jaccard_output = outputs[:, cls]
            intersection = (jaccard_output * jaccard_target).sum()
            coefficient = (2. * intersection + smooth) / (jaccard_output.sum() + jaccard_target.sum() + smooth)
            loss += 1. - coefficient
        return loss


'''
Implementation based on paper
Mahendran, A., Thewlis, J., & Vedaldi, A. (n.d.). Cross Pixel Optical Flow Similarity for Self-Supervised Learning.
'''


# class CrossPixelSimilarityLoss_origin:
#     def __init__(self, sigma=0.0036, sampling_size=512):
#         self.sigma = sigma
#         self.sampling_size = sampling_size
#         self.epsilon = 1.0e-15
#
#     def __call__(self, embeddings, flows):
#         assert (flows.shape[1] == 2)
#
#         ## Build optical flow correlation matrix
#         flows_flatten = flows.view(flows.shape[0], 2, -1)
#         # Spatially random sampling (512 samples)
#         random_locations = np.array(random.sample(range(flows_flatten.shape[2]), self.sampling_size))
#         flows_sample = torch.index_select(flows_flatten, 2, torch.from_numpy(random_locations).long().cuda())
#         # Broadcasting can handle these "1" dimension size
#         # norm = torch.norm(torch.unsqueeze(flows_sample, dim=-1).permute(0, 3, 2, 1) -
#         #                            torch.unsqueeze(flows_sample, dim=-1).permute(0, 2, 3, 1), p=2, dim=3,
#         #                            keepdim=False)
#         # print(torch.mean(norm))
#
#         k_f = torch.exp(torch.norm(torch.unsqueeze(flows_sample, dim=-1).permute(0, 3, 2, 1) -
#                                    torch.unsqueeze(flows_sample, dim=-1).permute(0, 2, 3, 1), p=2, dim=3,
#                                    keepdim=False))
#         # column-wise normalization
#         # eye = torch.unsqueeze(torch.eye(k_f.shape[1]), dim=0).cuda()
#
#
#         # exp_k_f = torch.exp(k_f)
#         # eye = torch.unsqueeze(torch.eye(exp_k_f.shape[1]), dim=0).cuda()
#         # mask = torch.ones_like(exp_k_f) - eye
#         # masked_exp_k_f = torch.mul(mask, exp_k_f) + eye
#         # s_f = masked_exp_k_f / torch.sum(masked_exp_k_f, dim=1, keepdim=True)
#
#         ## Build embeddings correlation matrix
#         embeddings_flatten = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
#         # Spatially random sampling (512 samples)
#         embeddings_sample = torch.index_select(embeddings_flatten, 2, torch.from_numpy(random_locations).long().cuda())
#
#         embeddings_sample_norm = torch.norm(embeddings_sample, p=2, dim=1, keepdim=True)
#         # k_theta = 0.25 * (self.epsilon + torch.matmul(embeddings_sample.permute(0, 2, 1), embeddings_sample)) / (
#         #         self.epsilon + torch.matmul(embeddings_sample_norm.permute(0, 2, 1), embeddings_sample_norm))
#         # # column-wise normalization
#         # exp_k_theta = torch.exp(k_theta)
#         # masked_exp_k_theta = torch.mul(mask, exp_k_theta) + torch.tensor(np.exp(-0.75)).cuda() * eye
#         # s_theta = masked_exp_k_theta / torch.sum(masked_exp_k_theta, dim=1, keepdim=True)
#
#         ## Cross entropy
#         loss = -torch.sum(torch.mul(s_f, torch.log(s_theta)))
#         return loss

class CrossPixelSimilarityLoss:
    def __init__(self, sigma=0.0036, sampling_size=512, norm_epsilon=1.0):
        self.sigma = sigma
        self.sampling_size = sampling_size
        self.epsilon = 1.0e-15
        self.norm_epsilon = norm_epsilon

    def __call__(self, embeddings, flows):
        assert (flows.shape[1] == 2)

        ## Build optical flow correlation matrix
        flows_flatten = flows.view(flows.shape[0], 2, -1)
        # Spatially random sampling (512 samples)
        random_locations = np.array(random.sample(range(flows_flatten.shape[2]), self.sampling_size))
        flows_sample = torch.index_select(flows_flatten, 2, torch.from_numpy(random_locations).long().cuda())
        # Broadcasting can handle these "1" dimension size
        k_f = self.epsilon + torch.norm(torch.unsqueeze(flows_sample, dim=-1).permute(0, 3, 2, 1) -
                                        torch.unsqueeze(flows_sample, dim=-1).permute(0, 2, 3, 1), p=2, dim=3,
                                        keepdim=False)
        # print_info(k_f)
        # column-wise normalization
        s_f = k_f / torch.sum(k_f, dim=1, keepdim=True)

        ## Build embeddings correlation matrix
        embeddings_flatten = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
        # Spatially random sampling (512 samples)
        embeddings_sample = torch.index_select(embeddings_flatten, 2, torch.from_numpy(random_locations).long().cuda())

        embeddings_sample_norm = torch.norm(embeddings_sample, p=2, dim=1, keepdim=True)
        k_theta = 1.0 + self.epsilon - (torch.matmul(embeddings_sample.permute(0, 2, 1), embeddings_sample)) / (
            torch.matmul(embeddings_sample_norm.permute(0, 2, 1) + self.norm_epsilon,
                         embeddings_sample_norm + self.norm_epsilon))
        # print_info(k_theta)
        # column-wise normalization
        s_theta = k_theta / torch.sum(k_theta, dim=1, keepdim=True)

        ## Cross entropy
        loss = -torch.mean(torch.mul(s_f, torch.log(s_theta)))

        return loss


def print_info(tensor):
    print(torch.max(tensor), torch.min(tensor), torch.mean(tensor))
