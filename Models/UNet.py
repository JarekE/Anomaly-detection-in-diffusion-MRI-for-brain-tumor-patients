import torch
import torch.nn as nn
from torch.nn import functional as F
from config import rec_filter, ac_function_rec


class UNet3d(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        act_fn = nn.LeakyReLU(inplace=True)

        # Reconstruction Network
        self.in_dim_rec = in_channels
        self.out_dim_rec = 64
        self.num_filter_rec = rec_filter

        self.down_1_rec = self.double_conv_block(self.in_dim_rec, self.num_filter_rec, act_fn)
        self.pool_1_rec = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_2_rec = self.double_conv_block(self.num_filter_rec, self.num_filter_rec * 2, act_fn)
        self.pool_2_rec = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_3_rec = self.double_conv_block(self.num_filter_rec * 2, self.num_filter_rec * 4, act_fn)
        self.pool_3_rec = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.bridge_rec = self.double_conv_block(self.num_filter_rec * 4, self.num_filter_rec * 8, act_fn)

        self.trans_1_rec = self.up_conv(self.num_filter_rec * 8, self.num_filter_rec * 8, act_fn)
        self.up_1_rec = self.double_conv_block(self.num_filter_rec * 12, self.num_filter_rec * 4, act_fn)

        self.trans_2_rec = self.up_conv(self.num_filter_rec * 4, self.num_filter_rec * 4, act_fn)
        self.up_2_rec = self.double_conv_block(self.num_filter_rec * 6, self.num_filter_rec * 2, act_fn)

        self.trans_3_rec = self.up_conv(self.num_filter_rec * 2, self.num_filter_rec * 2, act_fn)
        self.up_3_rec = self.double_conv_block(self.num_filter_rec * 3, self.num_filter_rec, act_fn)

        self.out_rec = self.out_block_rec(self.num_filter_rec, self.out_dim_rec)

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

    def out_block_rec(self, in_dim, out_dim):
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


    def reconstructive(self, x):
        down_1 = self.down_1_rec(x)
        pool_1 = self.pool_1_rec(down_1)
        down_2 = self.down_2_rec(pool_1)
        pool_2 = self.pool_2_rec(down_2)
        down_3 = self.down_3_rec(pool_2)
        pool_3 = self.pool_3_rec(down_3)

        bridge = self.bridge_rec(pool_3)

        trans_1 = self.trans_1_rec(bridge)
        concat_1 = torch.cat([trans_1, down_3], dim=1)
        up_1 = self.up_1_rec(concat_1)
        trans_2 = self.trans_2_rec(up_1)
        concat_2 = torch.cat([trans_2, down_2], dim=1)
        up_2 = self.up_2_rec(concat_2)
        trans_3 = self.trans_3_rec(up_2)
        concat_3 = torch.cat([trans_3, down_1], dim=1)
        up_3 = self.up_3_rec(concat_3)

        out = self.out_rec(up_3)
        return out


    def forward(self, input_data):
        rec = self.reconstructive(input_data)

        return [rec]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        reconstruction = args[0]
        target = args[1]

        reconstruction_loss = F.mse_loss(reconstruction, target)
        loss = reconstruction_loss
        return {'loss': loss, 'RL': reconstruction_loss.detach()}