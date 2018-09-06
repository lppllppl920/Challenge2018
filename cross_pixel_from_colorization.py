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
from loss import CrossPixelSimilarityLoss
from models import UNetfromColorization
from dataset import Challenge2018TemporalOpticalFlowDataset
from utils import get_color_file_names, init_net, write_event
from torchsummary import summary
from transforms import (DualCompose,
                        Resize,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        VerticalFlip,
                        MaskOnly,
                        RandomNoise,
                        NormalizeImage,
                        RandomBrightnessDual,
                        RandomContrastDual)

if __name__ == '__main__':
    device = torch.device("cuda")

    fold = 0
    batch_size = 1
    num_workers = 3
    root = Path("/home/xingtong/")
    data_root = root / "miccai_challenge_2018_training_data"
    json_file_name = str(data_root / "labels.json")
    train_file_names = get_color_file_names(fold=fold, root=data_root)

    lr = 1.0e-4
    n_epochs = 200
    # scale = 4
    img_width = 1280
    img_height = 1024
    factor = 0.05
    train_transform = DualCompose([
        # Resize(w=img_width, h=img_height),
        HorizontalFlip(),
        VerticalFlip(),
        ImageOnly(
            [RandomBrightnessDual(limit=0.3),
             RandomContrastDual(limit=0.3)
             ]),
        ImageOnly([NormalizeImage()])])
    # RandomSaturation(limit=0.3)
    valid_transform = DualCompose([
        # Resize(w=img_width, h=img_height),
        ImageOnly([NormalizeImage()])
    ])

    train_dataset = Challenge2018TemporalOpticalFlowDataset(image_file_names=train_file_names,
                                                     to_augment=True, transform=train_transform, img_width=img_width, img_height=img_height, factor=factor, p=0.5)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    scale = factor * np.sqrt(img_height ** 2 + img_width ** 2)

    try:
        model_root = root / "models_optflow"
        model_root.mkdir(mode=0o777, parents=False)
    except OSError:
        print("path exists")

    try:
        results_root = root / "results_optflow"
        results_root.mkdir(mode=0o777, parents=False)
    except OSError:
        print("path exists")

    model_path = model_root / 'model_{fold}.pt'.format(fold=fold)

    net = UNetfromColorization(num_classes=11, filters_base=16)
    utils.init_net(net)
    summary(net, input_size=(3, img_height, img_width))

    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        net.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': net.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    ##TODO: Remove below lines
    first_time = False
    if first_time:
        prev_root = root / "models_colorization"
        embedding_model_path = prev_root / 'G_model_1.pt'
        net.load_state_dict(torch.load(str(embedding_model_path))['model'])

    # Optimizer
    optimizer = Adam(net.parameters(), lr=lr)

    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')

    cross_pixel_loss = CrossPixelSimilarityLoss(sigma=0.1, sampling_size=64)
    for epoch in range(epoch, n_epochs + 1):
        net.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        try:
            mean_loss = 0
            for i, (colors, flows) in enumerate(train_loader):
                colors = colors.to(device)
                flows = flows.to(device)

                cpu_flows = flows.data.cpu().numpy()
                images = colors.data.cpu().numpy()
                cpu_flows = np.moveaxis(cpu_flows, [0, 1, 2, 3], [0, 3, 1, 2])
                images = np.moveaxis(images, [0, 1, 2, 3], [0, 3, 1, 2])

                cv2.imshow('flow HSV', utils.draw_hsv(cpu_flows[0] * scale))
                cv2.imshow("color", images[0] * 0.5 + 0.5)
                cv2.waitKey()

                embeddings = net(colors)
                loss = cross_pixel_loss(embeddings, flows)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                # if(i == len(train_loader) - 1):
                # if(i == 0):
                # images = utils.draw_embeddings(colors, embeddings, 5)
                # utils.write_images(images, root=results_root, file_prefix="embedding_epoch_" + str(epoch) + "_" + str(i) + "_")
                # # cv2.waitKey(100)
                # cv2.waitKey()
            tq.set_postfix(loss='{:.5f}'.format(np.mean(losses)))
            tq.close()
            save(epoch + 1)

        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')