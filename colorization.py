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
from dataset import Challenge2018ColorizationDataset
from utils import get_color_file_names_both_cam, init_net, write_event
from torchsummary import summary
from transforms import (DualCompose,
                        Resize,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        VerticalFlip,
                        MaskOnly,
                        RandomNoise,
                        RandomCrop,
                        RandomBrightnessDual,
                        RandomContrastDual,
                        RandomSaturationDual,
                        RandomColorDual)

if __name__ == '__main__':
    device = torch.device("cuda")

    fold = 1
    batch_size = 8
    num_workers = 8
    root = Path("/home/xingtong/")
    data_root = root / "miccai_challenge_2018_training_data"
    json_file_name = str(data_root / "labels.json")
    train_file_names, val_file_names = get_color_file_names_both_cam(root=data_root)

    lr = 2.0e-4
    gaussian_std = 0.05
    n_epochs = 600
    # scale = 4

    img_width = 256 #1280 // scale
    img_height = 256 #1024 // scale
    offset = 30
    loss_ratio = 0.5
    train_transform = DualCompose([
        Resize(w=img_width + offset, h=img_height + offset),
        RandomCrop(size=(img_height, img_width)),
        HorizontalFlip(),
        VerticalFlip(),
        RandomColorDual(limit=0.3, prob=1.0),
        # RandomBrightnessDual(limit=0.3),
        # RandomContrastDual(limit=0.3),
        # RandomSaturationDual(limit=0.3),
        Normalize(normalize_mask=True)])

    valid_transform = DualCompose([
        Resize(w=img_width, h=img_height),
        Normalize(normalize_mask=True)
    ])

    train_dataset = Challenge2018ColorizationDataset(image_file_names=train_file_names,
                                                     to_augment=True, transform=train_transform)

    val_dataset = Challenge2018ColorizationDataset(image_file_names=val_file_names[::20],
                                                   to_augment=True, transform=valid_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Building Generator
    netG = UNet_Colorization(num_classes=11, filters_base=16)
    utils.init_net(netG)
    summary(netG, input_size=(3, img_height, img_width))

    # Building Discriminator
    netD = Discriminator(input_nc=3, img_height=img_height, img_width=img_width, filter_base=8, num_block=5)
    utils.init_net(netD)
    summary(netD, input_size=(3, img_height, img_width))

    # Optimizer
    # G_optimizer = Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=lr, betas=(0.5, 0.999))
    G_optimizer = Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    try:
        model_root = root / "models_colorization"
        model_root.mkdir(mode=0o777, parents=False)
    except OSError:
        print("path exists")

    try:
        results_root = root / "results_colorization"
        results_root.mkdir(mode=0o777, parents=False)
    except OSError:
        print("path exists")

    model_path = model_root / 'model_{fold}.pt'.format(fold=fold)

    # Read existing weights for both G and D models
    G_model_path = model_root / 'G_model_{fold}.pt'.format(fold=fold)
    D_model_path = model_root / 'D_model_{fold}.pt'.format(fold=fold)
    if G_model_path.exists() and D_model_path.exists():
        state = torch.load(str(G_model_path))
        netG.load_state_dict(state['model'])

        state = torch.load(str(D_model_path))
        epoch = state['epoch']
        step = state['step']
        best_mean_error = state['error']
        netD.load_state_dict(state['model'])

        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0
        best_mean_error = 0.0

    G_model_path_1 = model_root / 'compatible_G_model_{fold}_entire_dataset.pt'.format(fold=fold)
    torch.save(netG.state_dict(), str(G_model_path_1))
'''
    save = lambda ep, model, model_path, error: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'error': error
    }, str(model_path))

    dataset_length = len(train_loader)
    adding_noise = True
    validate_each = 1
    log = model_root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mse_gan = nn.MSELoss()
    mse_L2 = nn.MSELoss()

    best_mean_rec_loss = 100
    for epoch in range(epoch, n_epochs + 1):

        mean_D_loss = 0
        mean_G_loss = 0
        mean_recover_loss = 0
        D_losses = []
        G_losses = []
        recover_losses = []

        netG.train()
        netD.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        noise_model = DynamicGNoise(shape1=img_height, shape2=img_width,
                                    std=gaussian_std - (gaussian_std / n_epochs) * epoch)

        try:
            for i, (gray_inputs, color_inputs) in enumerate(train_loader):
                color_inputs, gray_inputs = color_inputs.to(device), gray_inputs.to(device)

                # # ## display
                # gray = gray_inputs.data.cpu().numpy()[0]
                # color = color_inputs.data.cpu().numpy()[0]
                # gray = np.moveaxis(gray, source=[0, 1, 2], destination=[2, 0, 1])
                # color = np.moveaxis(color, source=[0, 1, 2], destination=[2, 0, 1])
                # print(np.max(gray), np.max(color))
                # cv2.imshow("gray", gray * 0.5 + 0.5)
                # cv2.imshow("color", color * 0.5 + 0.5)
                # cv2.waitKey()

                # Update Discriminator
                D_optimizer.zero_grad()

                pred_colors = netG(gray_inputs)
                if adding_noise:
                    C_real = netD(noise_model(color_inputs))
                    C_fake = netD(noise_model(pred_colors.detach()))
                    # C_real = netD(torch.cat((noise_model(color_inputs), gray_inputs), dim=1))
                    # C_fake = netD(torch.cat((noise_model(pred_colors.detach()), gray_inputs), dim=1))
                else:
                    C_real = netD(color_inputs)
                    C_fake = netD(pred_colors.detach())
                    # C_real = netD(torch.cat((color_inputs, gray_inputs), dim=1))
                    # C_fake = netD(torch.cat((pred_colors.detach(), gray_inputs), dim=1))

                # mean_C_real = torch.mean(C_real, dim=0, keepdim=True).expand_as(C_real)
                # mean_C_fake = torch.mean(C_fake, dim=0, keepdim=True).expand_as(C_fake)
                mean_C_real = torch.mean(C_real, dim=0, keepdim=True).expand_as(C_real).detach()
                mean_C_fake = torch.mean(C_fake, dim=0, keepdim=True).expand_as(C_fake).detach()
                loss1 = mse_gan(C_real - mean_C_fake, torch.tensor(1.0).cuda().expand_as(C_real))
                loss2 = mse_gan(C_fake - mean_C_real, torch.tensor(-1.0).cuda().expand_as(C_fake))
                # loss3 = mse_L2(pred_color, color)
                loss = 0.5 * (loss1 + loss2)
                loss.backward()
                D_losses.append(loss.item())
                D_optimizer.step()

                # Updating Generator
                G_optimizer.zero_grad()

                pred_colors = netG(gray_inputs)
                C_real = netD(color_inputs)
                C_fake = netD(pred_colors)
                # C_real = netD(torch.cat((color_inputs, gray_inputs), dim=1))
                # C_fake = netD(torch.cat((pred_colors, gray_inputs), dim=1))

                # mean_C_real = torch.mean(C_real, dim=0, keepdim=True).expand_as(C_real)
                # mean_C_fake = torch.mean(C_fake, dim=0, keepdim=True).expand_as(C_fake)
                mean_C_real = torch.mean(C_real, dim=0, keepdim=True).expand_as(C_real).detach()
                mean_C_fake = torch.mean(C_fake, dim=0, keepdim=True).expand_as(C_fake).detach()
                loss1 = mse_gan(C_fake - mean_C_real, torch.tensor(1.0).cuda().expand_as(C_fake))
                loss2 = mse_gan(C_real - mean_C_fake, torch.tensor(-1.0).cuda().expand_as(C_real))
                loss3 = mse_L2(pred_colors, color_inputs)
                loss = (1.0 - loss_ratio) * 0.5 * (loss1 + loss2) + loss_ratio * loss3
                loss.backward()
                G_losses.append((0.5 * (loss1 + loss2)).item())
                recover_losses.append(loss3.item())
                G_optimizer.step()

                step += 1
                tq.update(batch_size)
                mean_D_loss = np.mean(D_losses)
                mean_G_loss = np.mean(G_losses)
                mean_recover_loss = np.mean(recover_losses)
                tq.set_postfix(
                    loss=' D={:.5f}, G={:.5f}, Rec={:.5f}'.format(mean_D_loss, mean_G_loss, mean_recover_loss))

                if i == dataset_length - 2:
                    color_inputs_cpu = color_inputs.data.cpu().numpy()
                    pred_color_cpu = pred_colors.data.cpu().numpy()
                    color_imgs = []
                    pred_color_imgs = []
                    for j in range(batch_size):
                        color_img = color_inputs_cpu[j]
                        pred_color_img = pred_color_cpu[j]

                        color_img = np.moveaxis(color_img, source=[0, 1, 2], destination=[2, 0, 1])
                        pred_color_img = np.moveaxis(pred_color_img, source=[0, 1, 2], destination=[2, 0, 1])

                        color_img = cv2.resize(color_img, dsize=(300, 300))
                        pred_color_img = cv2.resize(pred_color_img, dsize=(300, 300))
                        color_imgs.append(color_img)
                        pred_color_imgs.append(pred_color_img)

                    final_color = color_imgs[0]
                    final_pred_color = pred_color_imgs[0]
                    for j in range(batch_size - 1):
                        final_color = cv2.hconcat((final_color, color_imgs[j + 1]))
                        final_pred_color = cv2.hconcat((final_pred_color, pred_color_imgs[j + 1]))

                    final = cv2.vconcat((final_color, final_pred_color))
                    final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(results_root / 'generated_mask_{epoch}.png'.format(epoch=epoch)),
                                np.uint8(255 * (final * 0.5 + 0.5)))
                    cv2.imshow("generated", final * 0.5 + 0.5)
                    cv2.waitKey(10)

            if epoch % validate_each == 0:
                rec_losses = []
                counter = 0
                for j, (color_inputs, gray_inputs) in enumerate(val_loader):
                    netG.eval()
                    color_inputs, gray_inputs = color_inputs.to(device), gray_inputs.to(device)

                    pred_color_inputs = netG(gray_inputs)
                    pred_color_inputs_cpu = pred_color_inputs.data.cpu().numpy()
                    color_inputs_cpu = color_inputs.data.cpu().numpy()
                    with torch.no_grad():
                        rec_losses.append(mse_L2(pred_color_inputs, color_inputs).item())

                mean_rec_loss = np.mean(rec_losses)

                tq.set_postfix(
                    loss='validation Rec={:.5f}'.format(mean_rec_loss))
                # if mean_rec_loss < best_mean_rec_loss:
                ## TODO: indentation
                counter = 0
                for j, (gray_inputs, color_inputs) in enumerate(val_loader):
                    netG.eval()
                    color_inputs, gray_inputs = color_inputs.to(device), gray_inputs.to(device)
                    pred_color_inputs = netG(gray_inputs)
                    pred_color_inputs_cpu = pred_color_inputs.data.cpu().numpy()
                    color_inputs_cpu = color_inputs.data.cpu().numpy()
                    for idx in range(color_inputs_cpu.shape[0]):
                        color = color_inputs_cpu[idx]
                        pred_color = pred_color_inputs_cpu[idx]
                        color = np.moveaxis(color, source=[0, 1, 2], destination=[2, 0, 1])
                        pred_color = np.moveaxis(pred_color, source=[0, 1, 2], destination=[2, 0, 1])

                        result = cv2.cvtColor(
                            cv2.hconcat((np.uint8(255 * (color * 0.5 + 0.5)),
                                         np.uint8(255 * (pred_color * 0.5 + 0.5)))),
                            cv2.COLOR_BGR2RGB)
                        cv2.imwrite(str(results_root / 'validation_{counter}.png'.format(counter=counter)), result)
                        counter += 1

                # Save both models
                best_mean_rec_loss = mean_rec_loss
                save(epoch, netD, D_model_path, best_mean_rec_loss)
                save(epoch, netG, G_model_path, best_mean_rec_loss)
                print("Found better model in terms of validation loss: {}".format(best_mean_rec_loss))

            write_event(log, step, Rec_error=mean_recover_loss)
            write_event(log, step, Dloss=mean_D_loss)
            write_event(log, step, Gloss=mean_G_loss)
            tq.close()
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            tq.close()
            print('Ctrl+C, saving snapshot')
            print('done.')
            exit()

'''
