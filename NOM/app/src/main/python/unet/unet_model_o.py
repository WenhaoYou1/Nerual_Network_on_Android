from .unet_parts_o import *
from com.chaquo.python import Python
import psutil
from os.path import dirname, join
import torch


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.file_dir = str(Python.getPlatform().getApplication().getFilesDir())
        self.pt_mem = join(dirname(self.file_dir), 'mem_usage.txt') #x5
        self.count = 0

    def forward(self, x):

        self.mem_usage = []
        f = open(self.pt_mem, "a")
        f.write("Run {}: \n".format(self.count))
        self.count += 1
        
        x1 = self.inc(x)
        self.mem_usage.append(psutil.virtual_memory()[3]/1000000000)
        print(psutil.virtual_memory()[3]/1000000000)
        x2 = self.down1(x1)
        self.mem_usage.append(psutil.virtual_memory()[3]/1000000000)
        print(psutil.virtual_memory()[3]/1000000000)
        x3 = self.down2(x2)
        self.mem_usage.append(psutil.virtual_memory()[3]/1000000000)
        print(psutil.virtual_memory()[3]/1000000000)
        x4 = self.down3(x3)
        self.mem_usage.append(psutil.virtual_memory()[3]/1000000000)
        print(psutil.virtual_memory()[3]/1000000000)
        x5 = self.down4(x4)
        self.mem_usage.append(psutil.virtual_memory()[3]/1000000000)
        print(psutil.virtual_memory()[3]/1000000000)
        x = self.up1(x5, x4)
        self.mem_usage.append(psutil.virtual_memory()[3]/1000000000)
        print(psutil.virtual_memory()[3]/1000000000)
        x = self.up2(x, x3)
        self.mem_usage.append(psutil.virtual_memory()[3]/1000000000)
        print(psutil.virtual_memory()[3]/1000000000)
        x = self.up3(x, x2)
        self.mem_usage.append(psutil.virtual_memory()[3]/1000000000)
        print(psutil.virtual_memory()[3]/1000000000)
        x = self.up4(x, x1)
        self.mem_usage.append(psutil.virtual_memory()[3]/1000000000)
        print(psutil.virtual_memory()[3]/1000000000)
        logits = self.outc(x)
        self.mem_usage.append(psutil.virtual_memory()[3]/1000000000)
        print(psutil.virtual_memory()[3]/1000000000)

        f.write("UNet Model: ")
        f.write(str(self.mem_usage))
        f.write("\n")
        f.close()
        return logits

    # def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.down4 = torch.utils.checkpoint(self.down4)
    #     self.up1 = torch.utils.checkpoint(self.up1)
    #     self.up2 = torch.utils.checkpoint(self.up2)
    #     self.up3 = torch.utils.checkpoint(self.up3)
    #     self.up4 = torch.utils.checkpoint(self.up4)
    #     self.outc = torch.utils.checkpoint(self.outc)
