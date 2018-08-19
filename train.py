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

from loss import LossMulti, MultiDiceLoss
from models import UNet11, LinkNet34, UNet, UNet16, AlbuNet
from dataset import Challenge2018Dataset
from utils import get_color_file_names, write_event, read_json
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose
)


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
    json_file_name = str(data_root / "labels.json")
    # root = Path("/home/xingtong/miccai_challenge_2018_training_data")
    # json_file_name = "/home/xingtong/miccai_challenge_2018_training_data/labels.json"
    train_file_names, val_file_names = get_color_file_names(fold=fold, root=data_root)

    train_dataset = Challenge2018Dataset(image_file_names=train_file_names,
                                         json_file_name=json_file_name,
                                         to_augment=True, transform=train_transform())

    val_dataset = Challenge2018Dataset(image_file_names=val_file_names[:50],
                                       json_file_name=json_file_name,
                                       to_augment=True, transform=val_transform())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    lr = 1.0e-4
    n_epochs = 600
    add_log = True
    model = UNet(num_classes=11, filters_base=6, input_channels=3, add_log=add_log)

    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=None).cuda()

    optimizer = Adam(model.parameters(), lr=lr)

    try:
        model_root = root / "models"
        model_root.mkdir(mode=0o777, parents=False)
    except OSError:
        print("path exists")

    try:
        results_root = root / "results"
        results_root.mkdir(mode=0o777, parents=False)
    except OSError:
        print("path exists")

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
    valid_each = 4
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []

    if(add_log == False):
        criterion = MultiDiceLoss(num_classes=11)
    else:
        criterion = LossMulti(num_classes=11, jaccard_weight=0.5)
    class_color_table = read_json(json_file_name)
    first_time = True
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(train_loader):
                # images = inputs.data.cpu().numpy()
                # targets = targets.data.cpu().numpy()
                # print(targets.shape)
                # images = np.moveaxis(images, [0, 1, 2, 3], [0, 3, 1, 2])
                # cv2.imshow("color", images[0] * 0.5 + 0.5)
                # cv2.waitKey()
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                # if i and i % report_each == 0:
                #     write_event(log, step, loss=mean_loss)

            # write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)

            if(epoch % valid_each == 0):
                valid_metrics = validation_multi(model, criterion, val_loader, 11, first_time, class_color_table,
                                             results_root)
                first_time = False
                valid_loss = valid_metrics['valid_loss']
                valid_losses.append(valid_loss)
                # write_event(log, step, **valid_metrics)

        except KeyboardInterrupt:
                tq.close()
                print('Ctrl+C, saving snapshot')
                save(epoch)
                print('done.')
