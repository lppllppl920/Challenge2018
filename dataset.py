import torch
import numpy as np
import cv2

from torch.utils.data import Dataset
from albumentations.torch.functional import img_to_tensor

import utils

class Challenge2018ColorizationDataset(Dataset):
    def __init__(self, image_file_names, to_augment=False, transform=None):
        self.image_file_names = image_file_names
        self.to_augment = to_augment
        self.transform = transform

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img_file_name = str(self.image_file_names[idx])
        mask = load_image(img_file_name)
        image = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        ## Normalize mask image (color) to [-1.0, 1.0] (Augmentation later won't normalize mask image)
        mask = normalize(mask, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255)
        image = np.repeat(np.expand_dims(image, axis=-1), repeats=3, axis=-1)

        if self.to_augment:
            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]

        return img_to_tensor(image), img_to_tensor(mask)


class Challenge2018Dataset(Dataset):
    def __init__(self, image_file_names, json_file_name, to_augment=False, transform=None):
        self.image_file_names = image_file_names
        self.to_augment = to_augment
        self.transform = transform
        self.class_color_table = utils.read_json(json_file_name)

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img_file_name = str(self.image_file_names[idx])
        mask_file_name = img_file_name.replace("left_frames", "labels")

        image = load_image(img_file_name)
        mask = load_mask(mask_file_name, self.class_color_table)

        if self.to_augment:
            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]

        return img_to_tensor(image), torch.from_numpy(mask).long()


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, class_color_table):
    mask = cv2.imread(str(path))
    shape = mask.shape

    # one_hot_mask_vector = np.zeros((shape[0] * shape[1], 11), dtype=np.float32)
    mask_vector = np.zeros((shape[0] * shape[1], 1))
    mask_image_vector = np.reshape(mask, (-1, 3))
    for class_id in range(11):
        # one_hot_vector = np.zeros((11,), dtype=np.float32)
        # one_hot_vector[class_id] = 1.0
        indices = np.where(np.all(mask_image_vector == np.uint8(class_color_table[class_id]), axis=-1))
        mask_vector[indices] = class_id
        # one_hot_mask_vector[indices] = one_hot_vector

    mask = np.reshape(mask_vector, (shape[0], shape[1]))

    return mask


def normalize(img, mean, std, max_pixel_value=255.0):
    img = img.astype(np.float32) / max_pixel_value

    img -= np.ones(img.shape) * mean
    img /= np.ones(img.shape) * std
    return img

