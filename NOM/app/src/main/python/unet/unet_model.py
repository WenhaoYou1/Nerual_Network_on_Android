from .unet_parts import *
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
        self.pt_path0 = join(dirname(self.file_dir), 'tensor/tensor0.pt') #x
        self.pt_path1 = join(dirname(self.file_dir), 'tensor/tensor1.pt') #x1
        self.pt_path2 = join(dirname(self.file_dir), 'tensor/tensor2.pt') #x2
        self.pt_path3 = join(dirname(self.file_dir), 'tensor/tensor3.pt') #x3
        self.pt_path4 = join(dirname(self.file_dir), 'tensor/tensor4.pt') #x4
        self.pt_path5 = join(dirname(self.file_dir), 'tensor/tensor5.pt') #x5

        self.pt_mem = join(dirname(self.file_dir), 'mem_usage.txt') #x5
        self.count = 0

        # self.pt_path0 = join(dirname(__file__), 'output_tensors/tensor0.pt') #x
        # self.pt_path1 = join(dirname(__file__), 'output_tensors/tensor1.pt') #x1
        # self.pt_path2 = join(dirname(__file__), 'output_tensors/tensor2.pt') #x2
        # self.pt_path3 = join(dirname(__file__), 'output_tensors/tensor3.pt') #x3
        # self.pt_path4 = join(dirname(__file__), 'output_tensors/tensor4.pt') #x4
        # self.pt_path5 = join(dirname(__file__), 'output_tensors/tensor5.pt') #x5

    def forward(self, x):

        self.mem_usage_before = []
        self.mem_usage_after = []
        f = open(self.pt_mem, "a")
        f.write("Run {}: \n".format(self.count))
        self.count += 1

        x1 = self.inc(x)
        del x
        # print("x1")
        # print(x1)
        print('RAM Used (GB) for the output inc x1:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_before.append(psutil.virtual_memory()[3]/1000000000)
        torch.save(x1, self.pt_path1)
        del x1
        # print(x1)
        print('RAM Used (GB) after save inc x1:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_after.append(psutil.virtual_memory()[3]/1000000000)

        #################################################################################
        x2 = self.down1(torch.load(self.pt_path1))

        # print("x2")
        # print(x2)
        print('RAM Used (GB) for the output down1 x2:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_before.append(psutil.virtual_memory()[3]/1000000000)
        torch.save(x2, self.pt_path2)
        del x2
        # print(x2)
        print('RAM Used (GB) after save down1 x2:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_after.append(psutil.virtual_memory()[3]/1000000000)

        # exit()
        #################################################################################
        x3 = self.down2(torch.load(self.pt_path2))
        
        print('RAM Used (GB) for the output down2 x3:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_before.append(psutil.virtual_memory()[3]/1000000000)
        torch.save(x3, self.pt_path3)
        del x3
        # print(x3)
        print('RAM Used (GB) after save down2 x3:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_after.append(psutil.virtual_memory()[3]/1000000000)

        #################################################################################
        x4 = self.down3(torch.load(self.pt_path3))

        print('RAM Used (GB) for the output down3 x4:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_before.append(psutil.virtual_memory()[3]/1000000000)
        torch.save(x4, self.pt_path4)
        del x4
        # print(x4)
        print('RAM Used (GB) after save down3 x4:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_after.append(psutil.virtual_memory()[3]/1000000000)

        #################################################################################
        x5 = self.down4(torch.load(self.pt_path4))

        print('RAM Used (GB) for the output down4 x5:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_before.append(psutil.virtual_memory()[3]/1000000000)
        torch.save(x5, self.pt_path5)
        del x5
        # print(x5)
        print('RAM Used (GB) after save down4 x5:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_after.append(psutil.virtual_memory()[3]/1000000000)

        #################################################################################
        x = self.up1(torch.load(self.pt_path5), torch.load(self.pt_path4))
        # print(x)

        print('RAM Used (GB) for the output up1 x:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_before.append(psutil.virtual_memory()[3]/1000000000)
        torch.save(x, self.pt_path0)
        del x
        # print(x)
        print('RAM Used (GB) after save up1 x:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_after.append(psutil.virtual_memory()[3]/1000000000)

        #################################################################################
        # print(torch.load(self.pt_path0))
        x = self.up2(torch.load(self.pt_path0), torch.load(self.pt_path3))

        print('RAM Used (GB) for the output up2 x:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_before.append(psutil.virtual_memory()[3]/1000000000)
        torch.save(x, self.pt_path0)
        del x
        # print(x)
        print('RAM Used (GB) after save up2 x:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_after.append(psutil.virtual_memory()[3]/1000000000)

        #################################################################################
        x = self.up3(torch.load(self.pt_path0), torch.load(self.pt_path2))

        print('RAM Used (GB) for the output up3 x:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_before.append(psutil.virtual_memory()[3]/1000000000)
        torch.save(x, self.pt_path0)
        del x
        # print(x)
        print('RAM Used (GB) after save up3 x:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_after.append(psutil.virtual_memory()[3]/1000000000)

        #################################################################################

        # memory check
        x = self.up4(torch.load(self.pt_path0), torch.load(self.pt_path1))

        print('RAM Used (GB) for the output up4 x:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_before.append(psutil.virtual_memory()[3]/1000000000)
        torch.save(x, self.pt_path0)
        del x
        # print(x)
        print('RAM Used (GB) after save up4 x:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_after.append(psutil.virtual_memory()[3]/1000000000)

        #################################################################################
        logits = self.outc(torch.load(self.pt_path0))

        #print(logits)

        print('RAM Used (GB) for the output outc logits:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_before.append(psutil.virtual_memory()[3]/1000000000)
        torch.save(logits, self.pt_path0)
        del logits
        # print(x)
        print('RAM Used (GB) after save outc logits:', psutil.virtual_memory()[3]/1000000000)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        self.mem_usage_after.append(psutil.virtual_memory()[3]/1000000000)

        #print("load")
        #print(self.pt_path0)

        f.write("UNet Model before: ")
        f.write(str(self.mem_usage_before))
        f.write("\n")
        f.write("UNet Model after: ")
        f.write(str(self.mem_usage_after))
        f.write("\n")
        f.close()

        return torch.load(self.pt_path0)

    '''
    # these methods need to be generalized for several inputs
    def load(self, parameter = 1):
        pass
    def save(self, input1 = None, input2 = None, parameter = 1):
        pass
    '''

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
