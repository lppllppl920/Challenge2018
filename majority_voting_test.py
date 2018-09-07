import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
import random
import tqdm
import cv2
import numpy as np
from torch import nn

import utils
from models import UNet_Colorization, DynamicGNoise, Discriminator, UNet16Colorization, UNet11Colorization
from dataset import ChallengeTestDataset
from utils import get_test_color_file_names, init_net, write_event
from torchsummary import summary
from transforms import (ResizeImage,
                        ImageRealOnly,
                        NormalizeImage)


if __name__ == '__main__':
    # device = torch.device("cuda")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fold = 1
    batch_size = 8
    num_workers = 8
    root = Path("/home/xingtong/")
    data_root = root / "miccai_challenge_2018_training_data"
    json_file_name = str(data_root / "labels.json")

    loadSize = 266
    fineSize = 256
    crop_stride = 2

    test_transform = ImageRealOnly([ResizeImage(h=loadSize, w=loadSize),
                                NormalizeImage()])
    test_file_names = get_test_color_file_names(root=data_root)
    test_dataset = ChallengeTestDataset(image_file_names=test_file_names, transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    # Building Generator
    netG = UNet_Colorization(num_classes=11, filters_base=16)
    # utils.init_net(netG)
    # summary(netG, input_size=(3, fineSize, fineSize))

    model_root = None
    try:
        model_root = root / "models_colorization"
        model_root.mkdir(mode=0o777, parents=False)
    except OSError:
        print("path exists")

    G_model_path_1 = model_root / 'latest_net_G.pth'
    netG.load_state_dict(torch.load(str(G_model_path_1)))
    netG.cuda()

    batch_size = 8

    color_table = utils.read_json(json_file_name)
    try:
        netG.eval()
        for i, color_inputs in enumerate(test_loader):

            # # print(color_inputs.shape)
            # color_inputs.to(device)
            # color = color_inputs.data.cpu().numpy()[0]
            # cv2.imshow("color", color * 0.5 + 0.5)
            # cv2.waitKey()
            ## Crop in a deterministic way and forward these batches of images to the network
            for idx in range(color_inputs.shape[0]):
                cropped_image_list = np.zeros((((loadSize - fineSize) // crop_stride + 1) ** 2, fineSize, fineSize, 3), dtype=np.float32)
                cropped_mask_list = np.zeros((((loadSize - fineSize) // crop_stride + 1) ** 2, fineSize, fineSize, 3),
                                              dtype=np.float32)
                start_pos_list = []
                img = color_inputs[idx]
                count = 0
                for w in range((loadSize - fineSize) // crop_stride + 1):
                    for h in range((loadSize - fineSize) // crop_stride + 1):
                        cropped_image_list[count] = img[h:h+fineSize, w:w+fineSize]
                        start_pos_list.append([h * crop_stride, w * crop_stride])
                        count += 1
                # print("image", cropped_image_list.shape)

                cropped_image_list_swapped = np.moveaxis(cropped_image_list, source=[0, 1, 2, 3], destination=[0, 2, 3, 1])

                batch_num = int(np.ceil(cropped_image_list_swapped.shape[0] / 8.0))
                for j in range(batch_num):
                    input_cropped_images = torch.from_numpy(cropped_image_list_swapped[j * batch_size : min(cropped_image_list_swapped.shape[0], (j + 1) * batch_size)])
                    input_cropped_images = input_cropped_images.to(device)
                    output_cropped_masks = netG(input_cropped_images)
                    cropped_mask_list[j * batch_size : min(cropped_image_list_swapped.shape[0], (j + 1) * batch_size)] = \
                        np.moveaxis(output_cropped_masks.data.cpu().numpy(), source=[0, 1, 2, 3], destination=[0, 3, 1, 2])

                print("mask", cropped_mask_list.shape)
                color = cropped_image_list[0] * 0.5 + 0.5
                mask = cropped_mask_list[0] * 0.5 + 0.5
                color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

                cv2.imshow("color", color)
                cv2.imshow("mask", mask)

                one_hot_masks = utils.convert_color_mask_to_one_hot_mask(255 * (cropped_mask_list * 0.5 + 0.5), color_table)
                print(color_table)
                final_color_mask = utils.crop_majority_voting(start_pos_list, one_hot_masks, [loadSize, loadSize], [fineSize, fineSize], color_table)

                final_color_mask = cv2.cvtColor(final_color_mask, cv2.COLOR_RGB2BGR)
                cv2.imshow("final", final_color_mask)
                cv2.waitKey()

                # print("mask", cropped_mask_list.shape)
                # color = cropped_image_list[0] * 0.5 + 0.5
                # mask = cropped_mask_list[0] * 0.5 + 0.5
                # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                #
                # cv2.imshow("color", color)
                # cv2.imshow("mask", mask)
                # cv2.waitKey()





    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print('done.')
        exit()
