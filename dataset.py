import torch
import numpy as np
import cv2

from torch.utils.data import Dataset
from albumentations.torch.functional import img_to_tensor

import utils
import random


class VideoOpticalFlowDataset(Dataset):
    def __init__(self, image_file_names, to_augment=True, transform=None, img_width=640, img_height=480, factor=0.1):
        self.image_file_names = image_file_names
        self.to_augment = to_augment
        self.transform = transform
        self.scale = factor * np.sqrt(img_height ** 2 + img_width ** 2)

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img_file_name = str(self.image_file_names[idx])
        if idx != 0 and idx != len(self.image_file_names) - 1:
            plus = random.uniform(0, 1)
            if plus <= 0.5:
                adjacent_img_file_name = str(self.image_file_names[idx - 1])
            else:
                adjacent_img_file_name = str(self.image_file_names[idx + 1])
        else:
            if idx == 0:
                adjacent_img_file_name = str(self.image_file_names[idx + 1])
            else:
                adjacent_img_file_name = str(self.image_file_names[idx - 1])

        img_1 = cv2.imread(str(img_file_name))
        img_2 = cv2.imread(str(adjacent_img_file_name))

        ## The position of a flow vector is based on the grid of img_1
        flow = optical_flow_estimate(img_1, img_2)

        if self.to_augment:
            data = {"image": img_1, "mask": flow}
            augmented = self.transform(**data)
            img_1, flow = augmented["image"], augmented["mask"]

        return img_to_tensor(img_1), img_to_tensor(flow / self.scale)


class ChallengeTestDataset(Dataset):
    def __init__(self, image_file_names, transform=None):
        self.image_file_names = image_file_names
        self.transform = transform
    def __len__(self):
        return len(self.image_file_names)
    def __getitem__(self, idx):
        image_file_name = str(self.image_file_names[idx])
        image = cv2.imread(image_file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image #img_to_tensor(image)



class Challenge2018OpticalFlowDataset(Dataset):
    def __init__(self, image_file_names, to_augment=True, transform=None, img_width=1280, img_height=1024, factor=0.05, p=0.5):
        self.image_file_names = image_file_names
        self.to_augment = to_augment
        self.transform = transform
        self.scale = factor * np.sqrt(img_height ** 2 + img_width ** 2)
        self.p = p

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img_left_file_name = str(self.image_file_names[idx])
        img_right_file_name = img_left_file_name.replace("left_frames", "right_frames")

        image_left = cv2.imread(str(img_left_file_name))
        image_right = cv2.imread(str(img_right_file_name))

        prob = random.uniform(0, 1)
        if prob <= self.p:
            flow = optical_flow_estimate(image_right, image_left)
            if self.to_augment:
                data = {"image": image_left, "mask": flow}
                augmented = self.transform(**data)
                image, flow = augmented["image"], augmented["mask"]
                return img_to_tensor(image), img_to_tensor(flow / self.scale)
            else:
                return img_to_tensor(image_left), img_to_tensor(flow / self.scale)

        else:
            flow = optical_flow_estimate(image_left, image_right)
            if self.to_augment:
                data = {"image": image_right, "mask": flow}
                augmented = self.transform(**data)
                image, flow = augmented["image"], augmented["mask"]
                return img_to_tensor(image), img_to_tensor(flow / self.scale)
            else:
                return img_to_tensor(image_right), img_to_tensor(flow / self.scale)


class Challenge2018TemporalOpticalFlowDataset(Dataset):
    def __init__(self, image_file_names, to_augment=True, transform=None, img_width=1280, img_height=1024, factor=0.05, p=0.5):
        self.image_file_names = image_file_names
        self.to_augment = to_augment
        self.transform = transform
        self.scale = factor * np.sqrt(img_height ** 2 + img_width ** 2)
        self.p = p
        self.dataset_length = len(self.image_file_names)
        self.img_width = img_width
        self.img_height = img_height
    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img_left_file_name_1 = str(self.image_file_names[idx])
        if idx < self.dataset_length - 1:
            ## We need to handle the boundary case where names from two different sequences meet
            img_left_file_name_2 = str(self.image_file_names[idx + 1])
            index = img_left_file_name_1.find(".png")
            substring = img_left_file_name_1[index - 3:index]
            if substring == "000":
                img_left_file_name_2 = str(self.image_file_names[idx - 1])
        else:
            img_left_file_name_2 = str(self.image_file_names[idx - 1])

        # Double the amount of data for training
        prob = random.uniform(0, 1)
        if prob <= self.p:
            img_left_file_name_1.replace("left_frames", "right_frames")
            img_left_file_name_2.replace("left_frames", "right_frames")

        image_1 = cv2.imread(str(img_left_file_name_1))
        image_2 = cv2.imread(str(img_left_file_name_2))

        if(image_1.shape[0] != self.img_height or image_2.shape[1] != self.img_width):
            image_1 = cv2.resize(image_1, dsize=(self.img_width, self.img_height))
            image_2 = cv2.resize(image_2, dsize=(self.img_width, self.img_height))

        prob = random.uniform(0, 1)

        ## Change the order for calculating optical flows
        if prob <= 0.5:
            flow = optical_flow_estimate(image_1, image_2)
            if self.to_augment:
                image_1, flow = self.transform(image_1, flow)
            return img_to_tensor(image_1), img_to_tensor(flow / self.scale)
        else:
            flow = optical_flow_estimate(image_2, image_1)
            if self.to_augment:
                image_2, flow = self.transform(image_2, flow)
            return img_to_tensor(image_2), img_to_tensor(flow / self.scale)



class Challenge2018ColorizationDataset(Dataset):
    def __init__(self, image_file_names, to_augment=True, transform=None):
        self.image_file_names = image_file_names
        self.to_augment = to_augment
        self.transform = transform

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img_file_name = str(self.image_file_names[idx])
        mask = load_image(img_file_name)
        image = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        image = np.repeat(np.expand_dims(image, axis=-1), repeats=3, axis=-1)

        if self.to_augment:
            image, mask = self.transform(image, mask)

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


def optical_flow_estimate(img_1, img_2):
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray_1, gray_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow