import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
import random
import tqdm
import cv2
import numpy as np
from validation import validation_multi
from torch import nn

from loss import CrossPixelSimilarityLoss
from models import UNet_softmax
from dataset import Challenge2018OpticalFlowDataset
from utils import get_color_file_names
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose
)
from torchsummary import summary
import utils


def train_transform(p=1):
    return Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        Normalize(p=1, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ], p=p)


def val_transform(p=1):
    return Compose([
        Normalize(p=1, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ], p=p)


if __name__ == '__main__':
    device = torch.device("cuda")

    fold = 0
    batch_size = 3
    num_workers = 3
    root = Path("G:\Johns Hopkins University\Challenge")
    # root = Path("/home/xingtong/")
    data_root = root / "miccai_challenge_2018_training_data"
    # root = Path("/home/xingtong/miccai_challenge_2018_training_data")
    train_file_names, val_file_names = get_color_file_names(fold=fold, root=data_root)

    train_dataset = Challenge2018OpticalFlowDataset(image_file_names=train_file_names,
                                                    to_augment=True, transform=train_transform(), img_height=1024,
                                                    img_width=1280, factor=0.05)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    lr = 1.0e-3
    n_epochs = 100
    add_log = False
    add_output = True
    model = UNet_softmax(num_classes=11, filters_base=6, input_channels=3, add_output=add_output)
    utils.init_net(model)
    summary(model, input_size=(3, 1024, 1280))

    optimizer = Adam(model.parameters(), lr=lr)

    model_root = root / "models_left_right_frames_flow"
    model_root.mkdir(mode=0o777, parents=False, exist_ok=True)

    results_root = root / "results_left_right_frames_flow"
    results_root.mkdir(mode=0o777, parents=False, exist_ok=True)

    model_path = model_root / 'model_{fold}.pt'.format(fold=fold)

    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    report_each = 10
    validate_each = 1
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')

    cross_pixel_loss = CrossPixelSimilarityLoss(sigma=0.1, sampling_size=64, norm_epsilon=1.0e-3)

    scale = 0.05 * np.sqrt(1024 ** 2 + 1280 ** 2)

    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        try:
            mean_loss = 0
            for i, (colors, flows) in enumerate(train_loader):
                colors = colors.to(device)
                flows = flows.to(device)
                # cpu_flows = flows.data.cpu().numpy()
                # images = colors.data.cpu().numpy()
                # cpu_flows = np.moveaxis(cpu_flows, [0, 1, 2, 3], [0, 3, 1, 2])
                # images = np.moveaxis(images, [0, 1, 2, 3], [0, 3, 1, 2])
                #
                # cv2.imshow('flow HSV', utils.draw_hsv(cpu_flows[0] * scale))
                # cv2.imshow("color", images[0] * 0.5 + 0.5)
                # cv2.waitKey()
                embeddings = model(colors)
                loss = cross_pixel_loss(embeddings, flows)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))

            tq.set_postfix(loss='{:.5f}'.format(np.mean(losses)))
            tq.close()
            save(epoch + 1)

        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')

