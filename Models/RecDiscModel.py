import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
from kornia.core import Tensor
from kornia.testing import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from config import rec_filter, ac_function, ac_function_rec, loss_weight, positiv_weight, leaky_relu

"""

NEW CLASS
Uses AE in reconstruction and UNET in discrimination

"""

class RecDisc(nn.Module):
    def __init__(self, in_channels: int, in_channels_unet: int):
        super().__init__()

        if leaky_relu == "True":
            act_fn = nn.LeakyReLU(inplace=True)
        else:
            act_fn = nn.ReLU(inplace=True)

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
        self.up_1_rec = self.double_conv_block(self.num_filter_rec * 8, self.num_filter_rec * 4, act_fn)

        self.trans_2_rec = self.up_conv(self.num_filter_rec * 4, self.num_filter_rec * 4, act_fn)
        self.up_2_rec = self.double_conv_block(self.num_filter_rec * 4, self.num_filter_rec * 2, act_fn)

        self.trans_3_rec = self.up_conv(self.num_filter_rec * 2, self.num_filter_rec * 2, act_fn)
        self.up_3_rec = self.double_conv_block(self.num_filter_rec * 2, self.num_filter_rec, act_fn)

        self.out_rec = self.out_block_rec(self.num_filter_rec, self.out_dim_rec)

        # Discrimination Network
        self.in_dim = in_channels_unet
        self.out_dim = 1
        self.num_filter = 16

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

    def out_block(self, in_dim, out_dim):
        if ac_function == "Sigmoid":
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
        up_1 = self.up_1_rec(trans_1)
        trans_2 = self.trans_2_rec(up_1)
        up_2 = self.up_2_rec(trans_2)
        trans_3 = self.trans_3_rec(up_2)
        up_3 = self.up_3_rec(trans_3)

        out = self.out_rec(up_3)
        return out

    def discriminative(self, x):
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
        return out

    def forward(self, input_data):

        rec = self.reconstructive(input_data)
        # Concatenate input and reconstructed image
        conc = torch.cat((rec, input_data), dim=1)
        disc = self.discriminative(conc)

        return [disc, rec, input_data]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        # [disc, rec, anomaly_data, input, rec_map, z]
        discriminative_image = args[0]
        reconstruction = args[1]
        input = args[3]
        reconstructive_map = args[4]

        # Calculate reconstruction loss
        reconstruction_loss = F.mse_loss(input, reconstruction)

        # discriminative_loss = F.l1_loss(discriminative_image, reconstructive_map)
        #kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        #discriminative_loss = binary_focal_loss_with_logits(discriminative_image, reconstructive_map, **kwargs)
        discriminative_loss = F.binary_cross_entropy_with_logits(discriminative_image, reconstructive_map, pos_weight=torch.tensor(positiv_weight))

        loss = loss_weight * reconstruction_loss + discriminative_loss

        return {'loss': loss, 'RL': reconstruction_loss.detach(), 'DL': discriminative_loss.detach()}


"""

NEW CLASS
Uses Unets in both tasks 

"""


class RecDiscUnet(nn.Module):
    def __init__(self, in_channels: int, in_channels_unet: int):
        super().__init__()

        if leaky_relu == "True":
            act_fn = nn.LeakyReLU(inplace=True)
        else:
            act_fn = nn.ReLU(inplace=True)

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

        # Discrimination Network
        self.in_dim = in_channels_unet
        self.out_dim = 1
        self.num_filter = 16

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

    def out_block(self, in_dim, out_dim):
        if ac_function == "Sigmoid":
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

    def discriminative(self, x):
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
        return out

    def forward(self, input_data):

        rec = self.reconstructive(input_data)
        # Concatenate input and reconstructed image
        conc = torch.cat((rec, input_data), dim=1)
        disc = self.discriminative(conc)

        return [disc, rec, input_data]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        # [disc, rec, anomaly_data, input, rec_map, z]
        discriminative_image = args[0]
        reconstruction = args[1]
        input = args[3]
        reconstructive_map = args[4]

        # Calculate reconstruction loss
        reconstruction_loss = F.mse_loss(input, reconstruction)

        # discriminative_loss = F.l1_loss(discriminative_image, reconstructive_map)
        #kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        #discriminative_loss = binary_focal_loss_with_logits(discriminative_image, reconstructive_map, **kwargs)
        discriminative_loss = F.binary_cross_entropy_with_logits(discriminative_image, reconstructive_map, pos_weight=torch.tensor(positiv_weight))

        loss = loss_weight * reconstruction_loss + discriminative_loss

        return {'loss': loss, 'RL': reconstruction_loss.detach(), 'DL': discriminative_loss.detach()}


def binary_focal_loss_with_logits(
    input: Tensor,
    target: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: Optional[float] = None,
    pos_weight: Optional[Tensor] = None,
) -> Tensor:

    KORNIA_CHECK_SHAPE(input, ["B", "C", "*"])
    KORNIA_CHECK(
        input.shape[0] == target.shape[0],
        f'Expected input batch_size ({input.shape[0]}) to match target batch_size ({target.shape[0]}).',
    )

    if pos_weight is None:
        pos_weight = torch.ones(input.shape[-1], device=input.device, dtype=input.dtype)

    KORNIA_CHECK_IS_TENSOR(pos_weight)
    KORNIA_CHECK(input.shape[-1] == pos_weight.shape[0], "Expected pos_weight equals number of classes.")

    probs_pos = input.sigmoid()
    probs_neg = torch.sigmoid(-input)

    loss_tmp = (
        -alpha * pos_weight * probs_neg.pow(gamma) * target * input.sigmoid().log()
        - (1 - alpha) * torch.pow(probs_pos, gamma) * (1.0 - target) * (-input).sigmoid().log()
    )

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss