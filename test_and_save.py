# -*- coding: utf-8 -*-

# import cv2

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import lpips
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from PIL import Image
import numpy as np
import math
import time
import os
import datetime
import random
import shutil
from options.test_options import TestOptions
from models import create_model
import torchvision.transforms.functional as F
import torch
import torchvision

loss_fn_vgg = lpips.LPIPS(net='vgg', version=0.1)
transf = torchvision.transforms.Compose(
      [torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
#transf = torchvision.transforms.ToTensor()
# transform = transforms.Compose([torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) ])
l1_loos = torch.nn.L1Loss()


def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def calculate_ssim(img1, img2):
    img1 = img1.squeeze(0)
    img2 = img2.squeeze(0)

    ssim=compare_ssim(np.array(img1), np.array(img2), win_size=3,  multichannel=True)
    return ssim


def calculate_lplps(pre, gt):

    pre=pre.squeeze(0)
    pre = transf(pre)
    pre=pre.to(torch.float32).unsqueeze(0)
    pre = pre.to(torch.float32)

    gt = gt.squeeze(0)
    gt = transf(gt)
    gt = gt.to(torch.float32).unsqueeze(0)
    lpip = loss_fn_vgg(pre.float(), gt.float()).item()

    return lpip


def calculate_l1_loos(gt, pre):
    # pre = torch.tensor(pre,dtype=torch.float32)
    # gt = torch.tensor(gt,dtype=torch.float32)
    pre=pre.squeeze(0)
    gt = gt.squeeze(0)

    pre = transf(pre)
    gt = transf(gt)

    loos = l1_loos(pre,gt).item()

    return loos


import glob


def load_flist(flist):
    # np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
    if isinstance(flist, list):
        return flist
    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            except:
                return [flist]
    return []


def postprocess(img):
    img = (img + 1) / 2 * 255
    img = img.permute(0, 2, 3, 1)
    img = img.int().cpu().numpy().astype(np.uint8)
    return img


# load test data
val_image = '/opt/data/private/paris_eval'

# Model and version
opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0  # test code only supports num_threads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
model = create_model(opt)  # create a model given opt.model and other options
model.setup(opt)  # regular setup: load and print networks; create schedulers

model.eval()

val_mask_suffix = ['mask01', 'mask12', 'mask23', 'mask34', 'mask45', 'mask56']
save_dir_suffix = ['010', '1020', '2030', '3040', '4050', '5060']

for suffix_idx in range(6):

    val_mask = '/opt/data/private/mask_class/' + val_mask_suffix[suffix_idx]

    save_dir = './results/LGNet-' + save_dir_suffix[suffix_idx]

    test_image_flist = load_flist(val_image)
    print(len(test_image_flist))
    test_mask_flist = load_flist(val_mask)
    print(len(test_mask_flist))

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(os.path.join(save_dir, 'comp'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'masked'), exist_ok=True)

    psnr = []
    ssim = []
    l1 = []
    lpips = []
    mask_num = len(test_mask_flist)
    # iteration through datasets
    for idx in range(len(test_image_flist)):
        img = Image.open(test_image_flist[idx]).resize((256, 256)).convert('RGB')
        mask = Image.open(test_mask_flist[idx % mask_num]).resize((256, 256)).convert('L')

        masks = F.to_tensor(mask)
        images = F.to_tensor(img) * 2 - 1.
        images = images.unsqueeze(0)
        masks = masks.unsqueeze(0)

        data = {'A': images, 'B': masks, 'A_paths': ''}
        model.set_input(data)
        with torch.no_grad():
            model.forward()

        orig_imgs = postprocess(model.images)
        mask_imgs = postprocess(model.masked_images1)
        comp_imgs = postprocess(model.merged_images3)

        orig_img = orig_imgs.copy()
        comp_img = comp_imgs.copy()
        psnr_tmp = calculate_psnr(comp_img, orig_img)
        psnr.append(psnr_tmp)

        #ssim_tmp = calculate_ssim(comp_img, orig_img)
        ssim_tmp=calculate_ssim(comp_img,orig_img)
        #print("ssmi:", ssim_tmp)
        ssim.append(ssim_tmp)

        l1_loos_tmp = calculate_l1_loos(comp_img, orig_img)
        #print("l1_loos_tmp", l1_loos_tmp)
        l1.append(l1_loos_tmp)

        # lpip.append(util_of_lpips(net='alex').calc_lpips(np.asarray(comp_imgs),
        #                                              np.asarray(orig_imgs)).squeeze().detach().numpy())
        lplp_tmp = calculate_lplps(comp_img, orig_img)
        #print("lplp", lplp_tmp)
        lpips.append(lplp_tmp)

        names = test_image_flist[idx].split('/')
        Image.fromarray(comp_imgs[0]).save(save_dir + '/comp/' + names[-1].split('.')[0] + '_comp.png')
        Image.fromarray(mask_imgs[0]).save(save_dir + '/masked/' + names[-1].split('.')[0] + '_mask.png')

    print('Finish in {}'.format(save_dir))
    print('The avg psnr is', np.mean(np.array(psnr)))
    print('The avg ssim is', np.mean(np.array(ssim)))
    print('The avg l1 is', np.mean(np.array(l1)))
    print('The avg lpips is', np.mean(np.array(lpips)))
