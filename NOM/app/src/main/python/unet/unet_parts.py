import torch
import torch.nn as nn
import torch.nn.functional as F
# import frereg as fr
import freq_reg._freqreg_eva as fr

from com.chaquo.python import Python
import psutil
from os.path import dirname, join


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            fr.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, minrate=0.001, droprate=0.1),
            nn.BatchNorm2d(mid_channels, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            fr.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, minrate=0.001, droprate=0.1),
            nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

        self.file_dir = str(Python.getPlatform().getApplication().getFilesDir())
        self.pt_path_double_conv1 = join(dirname(self.file_dir), 'tensor/tensor_double_conv1.pt') #x1
        # self.pt_path_double_conv1 = join(dirname(__file__), 'output_tensors/tensor_double_conv1.pt') #x1

    def forward(self, x):
        torch.save(x, self.pt_path_double_conv1)
        del x
        # return self.double_conv(x)
        return self.double_conv(torch.load(self.pt_path_double_conv1))


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

        self.file_dir = str(Python.getPlatform().getApplication().getFilesDir())
        self.pt_path_down1 = join(dirname(self.file_dir), 'tensor/tensor_down1.pt') #x1
        # self.pt_path_down1 = join(dirname(__file__), 'output_tensors/tensor_down1.pt') #x1

    def forward(self, x):
        torch.save(x, self.pt_path_down1)
        del x
        # return self.maxpool_conv(x)
        return self.maxpool_conv(torch.load(self.pt_path_down1))


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # print('RAM Used (GB) for the Up Bilinear:', psutil.virtual_memory()[3]/1000000000)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            # print('RAM Used (GB) for the Up DoubleConv:', psutil.virtual_memory()[3]/1000000000)
        else:
            self.up = fr.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, minrate=0.001, droprate=0.1, bias=False)
            # print('RAM Used (GB) for the Up ConvTranspose:', psutil.virtual_memory()[3]/1000000000)
            self.conv = DoubleConv(in_channels, out_channels)
            # print('RAM Used (GB) for the Up DoubleConv:', psutil.virtual_memory()[3]/1000000000)

        self.file_dir = str(Python.getPlatform().getApplication().getFilesDir())
        self.pt_path_up1 = join(dirname(self.file_dir), 'tensor/tensor_up1.pt') #x1
        self.pt_path_up2 = join(dirname(self.file_dir), 'tensor/tensor_up2.pt') #x2
        # self.pt_path_up1 = join(dirname(__file__), 'output_tensors/tensor_up1.pt') #x1
        # self.pt_path_up2 = join(dirname(__file__), 'output_tensors/tensor_up2.pt') #x2

    def forward(self, x1, x2):
        x1 = self.up(x1)

        print('RAM Used (GB) for the output x1 and x2 in Up:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        torch.save(x1, self.pt_path_up1)
        torch.save(x2, self.pt_path_up2)
        del x1
        del x2
        # print(x1)
        print('RAM Used (GB) after save x1 and x2 in Up:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])



        diffY = torch.load(self.pt_path_up2).size()[2] - torch.load(self.pt_path_up1).size()[2]
        diffX = torch.load(self.pt_path_up2).size()[3] - torch.load(self.pt_path_up1).size()[3]

        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(torch.load(self.pt_path_up1), [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        
        print('RAM Used (GB) for the output pad in Up:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        torch.save(x1, self.pt_path_up1)
        del x1
        # print(x1)
        print('RAM Used (GB) after save pad in Up:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])

        x = torch.cat([torch.load(self.pt_path_up2), torch.load(self.pt_path_up1)], dim=1)
        # x = torch.cat([x2, x1], dim=1)

        print('RAM Used (GB) for the output x.cat in Up:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        torch.save(x, self.pt_path_up1)
        del x
        # print(x1)
        print('RAM Used (GB) after save x.cat in Up:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])

        return self.conv(torch.load(self.pt_path_up1))
        # return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = fr.Conv2d(in_channels, out_channels, kernel_size=1, minrate=0.001, droprate=0.1, bias=False)

        self.file_dir = str(Python.getPlatform().getApplication().getFilesDir())
        self.pt_path_out_conv1 = join(dirname(self.file_dir), 'tensor/tensor_out_conv1.pt') #x1

    def forward(self, x):
        torch.save(x, self.pt_path_out_conv1)
        del x
        return self.conv(torch.load(self.pt_path_out_conv1))
        # return self.conv(x)
