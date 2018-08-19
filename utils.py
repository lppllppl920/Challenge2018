import json
import numpy as np
from pathlib import Path
from datetime import datetime

import random
import cv2

import torch
import tqdm



def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()

def read_json(file_path = "G:\Johns Hopkins University\Challenge\miccai_challenge_2018_training_data\labels.json"):
    with open(file_path) as data_file:
        data = json.load(data_file)

    class_color_table = np.zeros((len(data), 3), dtype=np.float32)
    ## Convert RGB to BGR to follow the colorspace convetion of OpenCV
    for i in range(len(data)):
        temp = data[i]["color"]
        class_color_table[i] = [temp[2], temp[1], temp[0]]

    return class_color_table


def get_color_file_names(fold=0,
                         root=Path("G:\Johns Hopkins University\Challenge\miccai_challenge_2018_training_data")):
    folds = {0: [1, 3],
             1: [2, 5],
             2: [4, 8],
             3: [6, 7]}

    train_file_names = []
    val_file_names = []
    for i in range(1, 8):
        if i in folds[fold]:
            val_file_names += list((root / ('seq_' + str(i)) / 'left_frames').glob('frame*'))
        else:
            train_file_names += list((root / ('seq_' + str(i)) / 'left_frames').glob('frame*'))
    return train_file_names, val_file_names


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write(u'\n')
    log.flush()


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None,
          num_classes=None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
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
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs = cuda(inputs)

                with torch.no_grad():
                    targets = cuda(targets)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, num_classes)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


def write_images(class_images, class_color_table, root, prefix, frame_count):
    class_shape = class_images.shape
    table_shape = class_color_table.shape

    for batch_index in range(class_shape[0]):
        class_image = class_images[batch_index]
        class_image_vector = np.reshape(class_image, (-1, 1))
        rgb_class_image_vector = np.zeros((class_image_vector.shape[0], 3))
        for class_id in range(table_shape[0]):
            indices = np.where(np.all(class_image_vector == class_id, axis=-1))
            rgb_class_image_vector[indices] = class_color_table[class_id]

        rgb_class_image = np.reshape(rgb_class_image_vector, (class_shape[1], class_shape[2], 3))
        cv2.imwrite(str(root / prefix) + str(frame_count) + ".png", np.uint8(rgb_class_image))
        frame_count += 1

    return frame_count
