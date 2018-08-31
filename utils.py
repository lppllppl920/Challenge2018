import json
import numpy as np
from pathlib import Path
from datetime import datetime

import random
import cv2

import torch
import tqdm


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

def get_color_file_names_both_cam(fold=0,
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
            val_file_names += list((root / ('seq_' + str(i)) / 'right_frames').glob('frame*'))
        else:
            train_file_names += list((root / ('seq_' + str(i)) / 'left_frames').glob('frame*'))
            train_file_names += list((root / ('seq_' + str(i)) / 'right_frames').glob('frame*'))
    return train_file_names, val_file_names


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(unicode(json.dumps(data, sort_keys=True)))
    log.write(unicode('\n'))
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


def init_net(net):
    assert(torch.cuda.is_available())
    net = net.cuda()
    glorot_weight_zero_bias(net)
    return net


def glorot_weight_zero_bias(model):
    """
    Initalize parameters of all modules
    by initializing weights with glorot  uniform/xavier initialization,
    and setting biases to zero.
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                torch.nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def draw_embeddings(colors, embeddings, seed=0):
    embeddings_cpu = embeddings.data.cpu().numpy()
    colors_cpu = colors.data.cpu().numpy()
    np.random.seed(seed)
    projection = np.random.randn(3, embeddings_cpu.shape[1])

    images = []
    for i in range(embeddings_cpu.shape[0]):
        temp = np.moveaxis(embeddings_cpu[i], [0, 1, 2], [2, 0, 1])
        temp = np.expand_dims(temp, axis=-1)
        print(temp.shape)
        projected = np.squeeze(np.matmul(projection, temp))
        print(projected.shape)
        projected_display = np.uint8(255 * (projected - np.max(projected)) / (np.max(projected) - np.min(projected)))
        display_color = np.uint8(255 * (np.moveaxis(colors_cpu[i], [0, 1, 2], [2, 0, 1]) * 0.5 + 0.5))
        display = cv2.hconcat((display_color, projected_display))
        images.append(display)
        cv2.imshow("embeddings_" + str(i), display)

    return images


def write_images(images, root, file_prefix="embeddings"):
    for i, image in enumerate(images):
        cv2.imwrite(str(root / (file_prefix + "_{:03d}.png").format(i)), image)

    return
