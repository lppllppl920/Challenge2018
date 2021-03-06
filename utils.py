import json
import numpy as np
from pathlib import Path
from datetime import datetime

import random
import cv2

import torch
import tqdm


def read_json(file_path):
    with open(file_path) as data_file:
        data = json.load(data_file)

    class_color_table = np.zeros((len(data), 3), dtype=np.float32)
    ## Convert RGB to BGR to follow the colorspace convetion of OpenCV
    for i in range(len(data)):
        temp = data[i]["color"]
        class_color_table[i] = [temp[0], temp[1], temp[2]]

    return class_color_table


def get_color_file_names(fold=1,
                         root=Path("G:\Johns Hopkins University\Challenge\miccai_challenge_2018_training_data")):

    if(fold > 0):
        train_file_names = []
        val_file_names = []
        for i in range(1, 16):
            if i == fold:
                val_file_names += list((root / ('seq_' + str(i)) / 'left_frames').glob('frame*'))
            else:
                train_file_names += list((root / ('seq_' + str(i)) / 'left_frames').glob('frame*'))
        train_file_names.sort(), val_file_names.sort()
        return train_file_names, val_file_names
    else:
        ## No validation
        train_file_names = []
        for i in range(1, 16):
            train_file_names += list((root / ('seq_' + str(i)) / 'left_frames').glob('frame*'))
            train_file_names.sort()
        return train_file_names

def get_color_file_names_both_cam(root=Path("G:\Johns Hopkins University\Challenge\miccai_challenge_2018_training_data")):

    train_file_names = []
    val_file_names = []
    for i in range(1, 16):
        # if i == fold:
        #     val_file_names += list((root / ('seq_' + str(i)) / 'left_frames').glob('frame*'))
        #     val_file_names += list((root / ('seq_' + str(i)) / 'right_frames').glob('frame*'))
        # else:
        train_file_names += list((root / ('seq_' + str(i)) / 'left_frames').glob('frame*'))
        train_file_names += list((root / ('seq_' + str(i)) / 'right_frames').glob('frame*'))

    for i in range(18, 22):
        val_file_names += list((root / ('seq_' + str(i)) / 'left_frames').glob('frame*'))
        val_file_names += list((root / ('seq_' + str(i)) / 'right_frames').glob('frame*'))


    train_file_names.sort(), val_file_names.sort()
    return train_file_names, val_file_names


def get_test_color_file_names(root):
    test_file_names = []
    for i in range(18, 22):
        test_file_names += list((root / ('seq_' + str(i)) / 'left_frames').glob('frame*'))
    test_file_names.sort()
    return test_file_names

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


# def write_images(class_images, class_color_table, root, prefix, frame_count):
#     class_shape = class_images.shape
#     table_shape = class_color_table.shape
#
#     for batch_index in range(class_shape[0]):
#         class_image = class_images[batch_index]
#         class_image_vector = np.reshape(class_image, (-1, 1))
#         rgb_class_image_vector = np.zeros((class_image_vector.shape[0], 3))
#         for class_id in range(table_shape[0]):
#             indices = np.where(np.all(class_image_vector == class_id, axis=-1))
#             rgb_class_image_vector[indices] = class_color_table[class_id]
#
#         rgb_class_image = np.reshape(rgb_class_image_vector, (class_shape[1], class_shape[2], 3))
#         cv2.imwrite(str(root / prefix) + str(frame_count) + ".png", np.uint8(rgb_class_image))
#         frame_count += 1
#
#     return frame_count


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
    print(projection)
    images = []
    for i in range(embeddings_cpu.shape[0]):
        temp = np.moveaxis(embeddings_cpu[i], [0, 1, 2], [2, 0, 1])
        temp = np.expand_dims(temp, axis=-1)
        projected = np.log(1.0 + np.abs(np.squeeze(np.matmul(projection, temp))))
        projected_display = np.uint8(255 * (projected - np.max(projected, axis=(0, 1), keepdims=True)) /
                                     (np.max(projected, axis=(0, 1), keepdims=True) - np.min(projected, axis=(0, 1), keepdims=True)))
        display_color = np.uint8(255 * (np.moveaxis(colors_cpu[i], [0, 1, 2], [2, 0, 1]) * 0.5 + 0.5))
        display = cv2.hconcat((display_color, projected_display))
        images.append(display)
        cv2.imshow("embeddings_" + str(i), display)

    return images


def write_images(images, root, file_prefix="embeddings"):
    for i, image in enumerate(images):
        cv2.imwrite(str(root / (file_prefix + "_{:03d}.png").format(i)), image)
    return

## TODO:
## Provide a list of starting position, a list of cropped outputs, original image size and cropped image size
## We assume the cropped_image_list is BxHxWx11 where the position of determined class is 1
## color table is in RGB format
def crop_majority_voting(start_pos_list, cropped_image_list, org_size, crop_size, color_table):
    assert(len(org_size) == 2)
    assert(len(crop_size) == 2)
    assert(color_table.shape[0] == 11)
    voting_ground = np.zeros((org_size[0], org_size[1], 11), dtype=np.float32)
    final_color_map = np.zeros((org_size[0], org_size[1], 3), dtype=np.float32)

    for i in range(len(start_pos_list)):
        start_pos = start_pos_list[i]
        cropped_image = cropped_image_list[i]
    # for start_pos, cropped_image in zip(start_pos_list, cropped_image_list):
        voting_ground[start_pos[0]:start_pos[0] + crop_size[0], start_pos[1]:start_pos[1] + crop_size[1], :] += cropped_image
    voting_result_map = np.argmax(voting_ground, axis=-1)
    for i in range(11):
        final_color_map[voting_result_map == i] = color_table[i]

    print(final_color_map[voting_result_map == 4])
    return final_color_map / 255.0

## color table is in RGB format
## masks are in RGB format
def convert_color_mask_to_one_hot_mask(masks, color_table):

    one_hot_masks = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 11), dtype=np.float32)
    for i in range(11):
        one_hot_masks[:, :, :, i] = np.sum(np.abs(masks - color_table[i]), axis=-1)

    one_hot_map = np.argmin(one_hot_masks, axis=-1)
    # cv2.imshow("gray", np.uint8(one_hot_map[0] * 20))
    # cv2.waitKey()
    for i in range(11):
        temp = np.zeros((11,), dtype=np.float32)
        temp[i] = 1.0
        one_hot_masks[one_hot_map == i] = temp

    return one_hot_masks