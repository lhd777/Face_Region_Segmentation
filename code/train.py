import argparse
import logging
import os
import sys


import torch.nn as nn
from torch import optim
from tqdm import tqdm


from dataset import *
from metrics import *
from networks import *
from losses import *
from utils import *

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split


def train_net(net,
              device,
              epochs=20,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              classes=15):
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                size_length = masks_pred.size()[2] * masks_pred.size()[3]
                mask_p = masks_pred.view(batch_size, classes, size_length)
                mask_t = true_masks.view(batch_size, classes, size_length)
                # modify different criterions
                # loss = criterion(mask_p, mask_t)
                loss = criterion.forward(mask_p.float(), mask_t.float())
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


if __name__ == '__main__':
    dir_img = '../data/imgs/'
    dir_mask = './data/masks/'
    dir_checkpoint = './checkpoint/'

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    setup_seed(2020)
    # NestUnet 需要输入图像尺寸为16的整数倍
    net = build_model(model_name='NestUnet', input_channels=3, num_classes=15)
    criterion = DiceLoss()
    #     if net.n_classes > 1:
    #         criterion = nn.CrossEntropyLoss()
    #     else:
    #         criterion = nn.BCEWithLogitsLoss()
    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=25,
                  batch_size=1,
                  lr=0.001,
                  device=device,
                  img_scale=0.25,
                  val_percent=0.1,
                  classes=15)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
