import torch
import torch.nn as nn
from torch.nn import functional as F
from config import rec_filter, ac_function_rec


class UNet3d(nn.Module):
    def __init__(self,
                 in_channels: int,):
        super(UNet3d, self).__init__()

        self.in_dim = in_channels
        self.out_dim = in_channels
        self.num_filter = rec_filter
        act_fn = nn.LeakyReLU(inplace=True)

        #%% UNet body
        self.down_1 = self.double_conv_block(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_2 = self.double_conv_block(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_3 = self.double_conv_block(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.bridge = self.double_conv_block(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.trans_1 = self.up_conv(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1 = self.double_conv_block(self.num_filter * 12, self.num_filter * 4, act_fn)

        self.trans_2 = self.up_conv(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.up_2 = self.double_conv_block(self.num_filter * 6, self.num_filter * 2, act_fn)

        self.trans_3 = self.up_conv(self.num_filter * 2, self.num_filter * 2, act_fn)
        self.up_3 = self.double_conv_block(self.num_filter * 3, self.num_filter, act_fn)

        self.out = self.out_block(self.num_filter, self.out_dim)

    def conv_block(self, in_dim, out_dim, act_fn):
        model = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_dim),
            act_fn,
        )
        return model

    def up_conv(self, in_dim, out_dim, act_fn):
        model = nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(out_dim),
            act_fn,
        )
        return model

    def double_conv_block(self, in_dim, out_dim, act_fn):
        model = nn.Sequential(
            self.conv_block(in_dim, out_dim, act_fn),
            self.conv_block(out_dim, out_dim, act_fn),
        )
        return model

    def out_block(self, in_dim, out_dim):
        if ac_function_rec == "Sigmoid":
            model = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        else:
            model = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
            )
        return model

    def forward(self, x):

        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        bridge = self.bridge(pool_3)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_3], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_2], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_1], dim=1)
        up_3 = self.up_3(concat_3)

        out = self.out(up_3)

        return [out]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        target = args[1]

        F1_loss = F.l1_loss(recons, target)
        return {'loss': F1_loss}
