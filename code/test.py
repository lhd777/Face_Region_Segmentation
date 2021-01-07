import os
from PIL import Image, ImageDraw
import imgviz

from dataset import *
from metrics import *
from networks import *
from losses import *
from utils import *

import torch
from torch.utils.data import DataLoader, random_split

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    dir_img = '../data/imgs/'
    dir_mask = './data/masks/'
    dir_checkpoint = './checkpoint/'

    net = build_model(model_name='NestUnet', input_channels=3, num_classes=15)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    net.load_state_dict(torch.load("../model/Unet++.pth"))
    batch_size = 1
    setup_seed(2020)
    dataset_ = BasicDataset(dir_img, dir_mask, scale=0.25)
    n_val = int(len(dataset_) * 0.1)
    n_train = len(dataset_) - n_val
    train, val = random_split(dataset_, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    f1_, acc_, mIoU_ = [], [], []
    for batch in val_loader:
        imgs = batch['image']
        true_mask = batch['mask']
        idx = batch['id']
        image = (np.squeeze(imgs.numpy()).transpose((1, 2, 0)) * 255).astype(np.uint8)
        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_pred = torch.squeeze(torch.max(net(imgs), 1)[1]).numpy().astype(np.uint8)
        true_mask_ = torch.squeeze(torch.max(true_mask, 1)[1]).numpy().astype(np.uint8)

        # 拉伸到一维计算mIoU
        true_mask_ = true_mask_.reshape(-1)
        mask_pred_ = mask_pred.reshape(-1)

        metric = EvaluationMetric()
        f1, acc, mIoU = metric.evaluate(true_mask_, mask_pred_)
        print(f'f1:{f1}; acc:{acc}; mIoU:{mIoU}')
        f1_.append(f1)
        acc_.append(acc)
        mIoU_.append(mIoU)
        lbl_viz = imgviz.label2rgb(
            label=mask_pred, img=imgviz.asgray(image), loc="rb"
        )
        Image.fromarray(lbl_viz).save(f'./Dataset2/NestUnet_pred_dice/val/{idx[0]}.png')
    f1, acc, mIoU = np.mean(f1_), np.mean(acc_), np.mean(mIoU_)
    print(f'MEAN f1:{f1}; acc:{acc}; mIoU:{mIoU}')
