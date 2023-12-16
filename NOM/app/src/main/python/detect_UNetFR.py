import os
from os.path import dirname, join
import torch
import torch.nn.functional as F

from unet import UNet
from dice_score import multiclass_dice_coeff, dice_coeff

import numpy as np
import matplotlib.pyplot as plt
import imageio
from com.chaquo.python import Python


# testing purpose
import psutil



def loadFiles_plus(path_im, keyword = ""):
    re_fs = []
    re_fullfs = []

    files = os.listdir(path_im)
    files = sorted(files)

    for file in files:
        if file.find(keyword) != -1:
            re_fs.append(file)
            re_fullfs.append(path_im + "/" + file)

    return re_fs, re_fullfs


def model2File(model, save_path):

    state_dict = model.state_dict()
    save_dict = {}

    for i in state_dict.keys():
        save_dict[i] = state_dict[i].detach().to_sparse()

    torch.save(save_dict, save_path)


def file2Model(model, save_path):

    save_dict = torch.load(save_path)
    state_dict = {}

    for i in save_dict.keys():
        state_dict[i] = save_dict[i].to_dense()
    model.load_state_dict(state_dict)

    return model


def countParams(model):
    totalnum = 0
    totalbias = 0
    for name, layer in model.named_modules():
        try:
            num = torch.sum(layer.weight.data.abs() > 0).item()
            totalnum += num
        except:
            pass

        try:
            num = torch.sum(layer.bias.data.abs() > 0).item()
            totalbias += num
        except:
            pass

    return totalnum, totalbias


def cleanUNet(model):
    for name, layer in model.named_modules():
        try:
            idx = layer.IDROP.abs() < 0.0000000000000000000001
            layer.weight.data[idx] = 0

            layer.IDROP = None
        except:
            pass

        try:
            layer.ZMAT = None
        except:
            pass

        try:
            layer.IMAT = None
        except:
            pass


    return None




def main():

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    tar = join(dirname(__file__), "unet_fr.tar.xz")
    tar_file = "tar -xf " + tar
    os.system(tar_file)
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    cleanUNet(net)
    unet_pt = join(dirname(__file__), "unet_fr.pt")
    net = file2Model(net, unet_pt)
    net.to(device=device)
    # net.half()

    numweight, numbias = countParams(net)

    print("number of nonzero parameters in weight:", numweight)
    print("number of nonzero parameters in bias  :", numbias)


    input_dir = join(dirname(__file__), "testimgs/input")
    mask_dir = join(dirname(__file__), "testimgs/mask")

    fs_im, fullfs_im = loadFiles_plus(input_dir, 'png')
    fs_gt, fullfs_gt = loadFiles_plus(mask_dir, 'png')

    dice_score = 0

    # file directory
    file_dir = str(Python.getPlatform().getApplication().getFilesDir())
    # mem init
    pt_mem = join(dirname(file_dir), 'mem_usage.txt') #x5
    f = open(pt_mem, "w")
    f.write("Memory usage: \n")
    f.close()
    mem_usage = []

    plt.figure(figsize=(12, 4))
    for i in range(len(fullfs_im)):

        # ram check
        # Getting % usage of virtual_memory (3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB (4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
        mem_usage.append(psutil.virtual_memory()[3]/1000000000)

        img = torch.tensor(imageio.imread(fullfs_im[i]), dtype=torch.float32)/255.0
        img = img.half()

        print("env path:", os.environ['HOME'])
        # This line goes with the tensor cut part
        # img = img[:,:-1,:]
        # print(img.shape)
        # input("eee")
        lab = torch.tensor(imageio.imread(fullfs_gt[i]), dtype=torch.float32)/255.0

        print("segmenting files:", fullfs_im[i])

        img = img.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        lab = lab.unsqueeze(0).to(device=device, dtype=torch.long)

        '''
        ###########################
        # run the tensor cut here #
        # Assuming image_tensor is of shape [B, C, H, W]
        _, C, H, W = img.shape
        center_y, center_x = H // 2, W // 2

        # Cut the tensor into four pieces
        upper_left = img[:, :, 0:center_y, 0:center_x]
        upper_right = img[:, :, 0:center_y, center_x:W]
        lower_left = img[:, :, center_y:H, 0:center_x]
        lower_right = img[:, :, center_y:H, center_x:W]

        # Process each quadrant with the net (Placeholder for your neural network processing)
        upper_left_processed = net(upper_left)
        upper_right_processed = net(upper_right)
        lower_left_processed = net(lower_left)
        lower_right_processed = net(lower_right)

        # Combine the processed quadrants
        top_half = torch.cat((upper_left_processed, upper_right_processed), dim=3)
        bottom_half = torch.cat((lower_left_processed, lower_right_processed), dim=3)
        mask_pred = torch.cat((top_half, bottom_half), dim=2)

        # revise the size
        last_column = mask_pred[:, :, :, -1].unsqueeze(-1)  # Extract and add a new dimension to fit
        mask_pred = torch.cat((mask_pred, last_column), dim=3)
        ###########################
        '''


        mask_pred = net(img)

        # print(mask_pred.shape)
        print("done1")

        showmask = mask_pred.argmax(dim=1).squeeze()
        print("done2")
        showgt = lab.squeeze()
        print("done3")
        lab = F.one_hot(lab, net.n_classes).permute(0, 3, 1, 2).float()
        print("done4")
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        print("done5")
        dice_score += multiclass_dice_coeff(mask_pred[:, 1:], lab[:, 1:], reduce_batch_first=False)
        print("done6")

        # plt.subplot(1, 3, 1)
        # plt.imshow(img.squeeze().permute(1, 2, 0).detach().cpu().numpy())
        # plt.title("img")
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(showgt.detach().cpu().numpy())
        # plt.title("Groundtruth")
        #
        # plt.subplot(1, 3, 3)
        # plt.imshow(showmask.detach().cpu().numpy())
        # plt.title("Binary Mask")

        out0 = join(dirname(file_dir), 'output/result' + str(i) + '_0.png')
        print(out0)
        out1 = join(dirname(file_dir), 'output/result' + str(i) + '_1.png')
        print(out1)
        out2 = join(dirname(file_dir), 'output/result' + str(i) + '_2.png')
        print(out2)
        # out0 = join(dirname(__file__), 'output_test/result' + str(i) + '_0.png')
        # print(out0)
        # out1 = join(dirname(__file__), 'output_test/result' + str(i) + '_1.png')
        # print(out1)
        # out2 = join(dirname(__file__), 'output_test/result' + str(i) + '_2.png')
        # print(out2)
        plt.imsave(out0, img.squeeze().permute(1, 2, 0).detach().cpu().numpy())
        plt.imsave(out1, showgt.detach().cpu().numpy())
        plt.imsave(out2, showmask.detach().cpu().numpy())

        plt.pause(0.1)

        f = open(pt_mem, "a")
        f.write("Main usage {}: ".format(i))
        f.write(str(mem_usage))
        f.write("\n")
        f.close()




    print("average Dice Score:", dice_score.item()/len(fullfs_im))

    f = open(pt_mem, "a")
    f.write("Dice Score: {}".format(dice_score.item()/len(fullfs_im)))
    f.write("\n")
    f.close()

    return

main()
