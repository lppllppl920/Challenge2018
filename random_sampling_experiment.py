import torch
import cv2
from pathlib import Path
from albumentations.torch.functional import img_to_tensor
import random
import numpy as np

image_path = Path("G:/Johns Hopkins University/Challenge/davinci_surgical_video/video_1")
image = cv2.imread(str(image_path / "frame_00000.png"))

tensor = img_to_tensor(image)
print(tensor.shape)

vector = tensor.view(3, -1)


## Spatially random sampling (512 samples)
random_locations = np.array(random.sample(range(vector.shape[1]), 512))
sample = torch.index_select(vector, 1, torch.from_numpy(random_locations).long())
print(sample)


sample_norm = torch.norm(sample, p=2, dim=0, keepdim=True)
print(sample_norm.shape)

epsilon = 1.0e-15
sample_kernel_theta = (epsilon + torch.mm(sample.permute(1, 0), sample)) / \
                (epsilon + torch.mm(sample_norm.permute(1, 0), sample_norm))
print(sample_kernel_theta)

# expanded_sample_norm = sample_norm.expand(sample_norm.shape[1], -1)
# squared_expanded_sample_norm = torch.mul(expanded_sample_norm, expanded_sample_norm)
#
# squared_expanded_sample_norm + squared_expanded_sample_norm.permute(1, 0)
sigma = 1.0
expanded_sample = torch.unsqueeze(sample, dim=0).permute(0, 2, 1)
temp = torch.norm(expanded_sample - expanded_sample.permute(1, 0, 2), p=2, dim=2, keepdim=True) ** 2 / (-2.0 * sigma ** 2)
sample_kernel_f = torch.exp(temp)
sample_kernel_f = torch.squeeze(sample_kernel_f)
print(sample_kernel_f)


exp_sample_kernel_f = torch.exp(sample_kernel_f)
print(exp_sample_kernel_f)
mask = torch.ones_like(exp_sample_kernel_f) - torch.eye(exp_sample_kernel_f.shape[0])

masked_sample_kernel_f = torch.mul(mask, exp_sample_kernel_f) + torch.eye(exp_sample_kernel_f.shape[0])
print(masked_sample_kernel_f)

column_normalized_sample_kernel_f = masked_sample_kernel_f / \
                                    torch.sum(masked_sample_kernel_f, dim=0, keepdim=True).\
                                        expand(masked_sample_kernel_f.shape[0], -1)

print(column_normalized_sample_kernel_f)

print(torch.sum(column_normalized_sample_kernel_f, dim=0))



