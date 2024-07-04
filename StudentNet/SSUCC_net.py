import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

from cbam import SpatialGate

'''
'''
def update_module(module, new_in_channels, default_in_channels=3):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """
    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)

'''
'''
class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv = nn.Sequential(
            nn.Conv2d(3*channels, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = F.interpolate(self.pool1(x), x.shape[2:])
        x2 = F.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)  # (1, 3C, H, W)
        fusion = self.conv(concat)

        return fusion

'''
'''
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        if in_channels < r:
            self.excitation = nn.Sequential(
                nn.Linear(in_channels, in_channels * r, bias=False),
                nn.ReLU(inplace=False),
                nn.Linear(in_channels * r, in_channels, bias=False),
                nn.Sigmoid()
            )
        else:
            self.excitation = nn.Sequential(
                nn.Linear(in_channels, in_channels // r, bias=False),
                nn.ReLU(inplace=False),
                nn.Linear(in_channels // r, in_channels, bias=False),
                nn.Sigmoid()
            )

    def forward(self, modality1, modality2, cross_modality=None):
        if cross_modality is not None:
            x = torch.cat((modality1, modality2, cross_modality), dim=1)
        else:
            x = torch.cat((modality1, modality2), dim=1)
        batch, channels, _, _ = x.shape
        theta = self.squeeze(x).view(batch, channels)
        theta = self.excitation(theta).view(batch, channels, 1, 1)
        x = x * theta
        if cross_modality is not None:
            modality1_features, modality2_features, cm_features = torch.chunk(x, 3, dim=1)
            x = modality1_features + modality2_features + cm_features
        else:
            modality1_features, modality2_features = torch.chunk(x, 2, dim=1)
            x = modality1_features + modality2_features
        return x

'''
'''
class FuseBlock(nn.Module):
    def __init__(self, channel, is_first_block=False):
        super(FuseBlock, self).__init__()

        self.modality1_msc = MSC(channel)
        self.modality2_msc = MSC(channel)
        self.cross_modality_msc = MSC(channel)

        self.spatial_attn_block = SpatialGate()

        if is_first_block:
            self.update_cross_modality_block = SEBlock(channel*2)
        else:
            self.update_cross_modality_block = SEBlock(channel*3)

        self.modality1_distribute_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.modality2_distribute_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, modality1, modality2, cross_modality=None):

        modality1_m = self.modality1_msc(modality1)
        modality2_m = self.modality2_msc(modality2)

        modality1_spatial_attn = self.spatial_attn_block(modality1_m-modality2_m)
        
        if cross_modality is None:
            new_cross_modality = self.update_cross_modality_block(modality1_spatial_attn * modality1_m, modality2_m)
        else:
            cross_modality_m = self.cross_modality_msc(cross_modality)
            new_cross_modality = self.update_cross_modality_block(modality1_spatial_attn * modality1_m, 
                                                                  modality2_m, 
                                                                  cross_modality_m)

        s_modality1 = self.modality1_distribute_conv(new_cross_modality - modality1_m)
        modality1_distribute_gate = torch.sigmoid(s_modality1)

        s_modality2 = self.modality2_distribute_conv(new_cross_modality - modality2_m)
        modality2_distribute_gate = torch.sigmoid(s_modality2)

        new_modality1 = modality1 + (new_cross_modality - modality1_m) * modality1_distribute_gate
        new_modality2 = modality2 + (new_cross_modality - modality2_m) * modality2_distribute_gate

        return new_modality1, new_modality2, new_cross_modality

'''
'''
class SSUCCNet(nn.Module):
    def __init__(self, encoder_name, encoder_weights, classes):
        super(SSUCCNet, self).__init__()
        self.modality1_stream = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=4,
            classes=classes,
        )
        if encoder_name == 'mit_b4': update_module(self.modality1_stream.encoder.patch_embed1.proj, new_in_channels=4)

        self.modality2_stream = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=2,
            classes=classes,
        )
        if encoder_name == 'mit_b4': update_module(self.modality2_stream.encoder.patch_embed1.proj, new_in_channels=2)

        self.cross_modality_stream = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
        )
        
        out_channels = self.modality1_stream.encoder.out_channels[2:] # stage 1 ~ 4
        self.fuse_block_stage1 = FuseBlock(channel=out_channels[0], is_first_block=True)
        self.fuse_block_stage2 = FuseBlock(channel=out_channels[1])
        self.fuse_block_stage3 = FuseBlock(channel=out_channels[2])
        self.fuse_block_stage4 = FuseBlock(channel=out_channels[3])

        self.cr_head = nn.Sequential(*[
            nn.Conv2d(16, 64, 3, padding=1, stride=1),
            nn.Conv2d(64, 4, 3, padding=1, stride=1)
        ])
        
    def forward(self, modality1, modality2, output_shape=None):

        # input
        modality1_x = modality1
        modality2_x = modality2

        # get [B, C, H, W]
        modality1_B, modality1_C, modality1_H, modality1_W = modality1_x.shape
        modality2_B, modality2_C, modality2_H, modality2_W = modality2_x.shape
        assert modality1_B == modality2_B
        assert modality1_H == modality2_H
        assert modality1_W == modality2_W

        # assign value to B
        B = modality1_B

        # get initial value
        dummy1 = torch.empty([B, 0, modality1_H, modality1_W], dtype=modality1_x.dtype, device=modality1_x.device)
        dummy2 = torch.empty([B, 0, modality1_H // 2, modality1_W // 2], dtype=modality1_x.dtype, device=modality1_x.device)
        modality1_features = [dummy1, dummy2]
        modality2_features = [dummy1, dummy2]
        cross_modality_features = [dummy1, dummy2]

        # stage 1
        modality1_x, modality1_H, modality1_W = self.modality1_stream.encoder.patch_embed1(modality1_x)
        for i, blk in enumerate(self.modality1_stream.encoder.block1):
            modality1_x = blk(modality1_x, modality1_H, modality1_W)
        modality1_x = self.modality1_stream.encoder.norm1(modality1_x)
        modality1_x = modality1_x.reshape(B, modality1_H, modality1_W, -1).permute(0, 3, 1, 2).contiguous()

        modality2_x, modality2_H, modality2_W = self.modality2_stream.encoder.patch_embed1(modality2_x)
        for i, blk in enumerate(self.modality2_stream.encoder.block1):
            modality2_x = blk(modality2_x, modality2_H, modality2_W)
        modality2_x = self.modality2_stream.encoder.norm1(modality2_x)
        modality2_x = modality2_x.reshape(B, modality2_H, modality2_W, -1).permute(0, 3, 1, 2).contiguous()

        modality1_x, modality2_x, cross_modality_x = self.fuse_block_stage1(modality1_x, modality2_x, None)
        modality1_features.append(modality1_x)
        modality2_features.append(modality2_x)
        cross_modality_features.append(cross_modality_x)

        # stage 2
        modality1_x, modality1_H, modality1_W = self.modality1_stream.encoder.patch_embed2(modality1_x)
        for i, blk in enumerate(self.modality1_stream.encoder.block2):
            modality1_x = blk(modality1_x, modality1_H, modality1_W)
        modality1_x = self.modality1_stream.encoder.norm2(modality1_x)
        modality1_x = modality1_x.reshape(B, modality1_H, modality1_W, -1).permute(0, 3, 1, 2).contiguous()

        modality2_x, modality2_H, modality2_W = self.modality2_stream.encoder.patch_embed2(modality2_x)
        for i, blk in enumerate(self.modality2_stream.encoder.block2):
            modality2_x = blk(modality2_x, modality2_H, modality2_W)
        modality2_x = self.modality2_stream.encoder.norm2(modality2_x)
        modality2_x = modality2_x.reshape(B, modality2_H, modality2_W, -1).permute(0, 3, 1, 2).contiguous()

        cross_modality_x, cross_modality_H, cross_modality_W = self.cross_modality_stream.encoder.patch_embed2(cross_modality_x)
        for i, blk in enumerate(self.cross_modality_stream.encoder.block2):
            cross_modality_x = blk(cross_modality_x, cross_modality_H, cross_modality_W)
        cross_modality_x = self.cross_modality_stream.encoder.norm2(cross_modality_x)
        cross_modality_x = cross_modality_x.reshape(B, cross_modality_H, cross_modality_W, -1).permute(0, 3, 1, 2).contiguous()

        modality1_x, modality2_x, cross_modality_x = self.fuse_block_stage2(modality1_x, modality2_x, cross_modality_x)
        modality1_features.append(modality1_x)
        modality2_features.append(modality2_x)
        cross_modality_features.append(cross_modality_x)

        # stage 3
        modality1_x, modality1_H, modality1_W = self.modality1_stream.encoder.patch_embed3(modality1_x)
        for i, blk in enumerate(self.modality1_stream.encoder.block3):
            modality1_x = blk(modality1_x, modality1_H, modality1_W)
        modality1_x = self.modality1_stream.encoder.norm3(modality1_x)
        modality1_x = modality1_x.reshape(B, modality1_H, modality1_W, -1).permute(0, 3, 1, 2).contiguous()

        modality2_x, modality2_H, modality2_W = self.modality2_stream.encoder.patch_embed3(modality2_x)
        for i, blk in enumerate(self.modality2_stream.encoder.block3):
            modality2_x = blk(modality2_x, modality2_H, modality2_W)
        modality2_x = self.modality2_stream.encoder.norm3(modality2_x)
        modality2_x = modality2_x.reshape(B, modality2_H, modality2_W, -1).permute(0, 3, 1, 2).contiguous()

        cross_modality_x, cross_modality_H, cross_modality_W = self.cross_modality_stream.encoder.patch_embed3(cross_modality_x)
        for i, blk in enumerate(self.cross_modality_stream.encoder.block3):
            cross_modality_x = blk(cross_modality_x, cross_modality_H, cross_modality_W)
        cross_modality_x = self.cross_modality_stream.encoder.norm3(cross_modality_x)
        cross_modality_x = cross_modality_x.reshape(B, cross_modality_H, cross_modality_W, -1).permute(0, 3, 1, 2).contiguous()

        modality1_x, modality2_x, cross_modality_x = self.fuse_block_stage3(modality1_x, modality2_x, cross_modality_x)
        modality1_features.append(modality1_x)
        modality2_features.append(modality2_x)
        cross_modality_features.append(cross_modality_x)

        # stage 4
        modality1_x, modality1_H, modality1_W = self.modality1_stream.encoder.patch_embed4(modality1_x)
        for i, blk in enumerate(self.modality1_stream.encoder.block4):
            modality1_x = blk(modality1_x, modality1_H, modality1_W)
        modality1_x = self.modality1_stream.encoder.norm4(modality1_x)
        modality1_x = modality1_x.reshape(B, modality1_H, modality1_W, -1).permute(0, 3, 1, 2).contiguous()

        modality2_x, modality2_H, modality2_W = self.modality2_stream.encoder.patch_embed4(modality2_x)
        for i, blk in enumerate(self.modality2_stream.encoder.block4):
            modality2_x = blk(modality2_x, modality2_H, modality2_W)
        modality2_x = self.modality2_stream.encoder.norm4(modality2_x)
        modality2_x = modality2_x.reshape(B, modality2_H, modality2_W, -1).permute(0, 3, 1, 2).contiguous()

        cross_modality_x, cross_modality_H, cross_modality_W = self.cross_modality_stream.encoder.patch_embed4(cross_modality_x)
        for i, blk in enumerate(self.cross_modality_stream.encoder.block4):
            cross_modality_x = blk(cross_modality_x, cross_modality_H, cross_modality_W)
        cross_modality_x = self.cross_modality_stream.encoder.norm4(cross_modality_x)
        cross_modality_x = cross_modality_x.reshape(B, cross_modality_H, cross_modality_W, -1).permute(0, 3, 1, 2).contiguous()

        modality1_x, modality2_x, cross_modality_x = self.fuse_block_stage4(modality1_x, modality2_x, cross_modality_x)
        modality1_features.append(modality1_x)
        modality2_features.append(modality2_x)
        cross_modality_features.append(cross_modality_x)

        cross_modality_decoder_out = self.cross_modality_stream.decoder(*cross_modality_features)
        cross_modality_pred = self.cross_modality_stream.segmentation_head(cross_modality_decoder_out)

        cr_out = self.modality1_stream.decoder(*modality1_features)
        cr_pred = self.cr_head(cr_out)

        if output_shape is not None:
            cross_modality_pred = F.interpolate(cross_modality_pred, size=output_shape, mode='bilinear', align_corners=False)
        
        feats = {
            "cross_modal_decoder_out": cross_modality_decoder_out
        }

        return cross_modality_pred, cr_pred, feats

'''
'''
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modality1 = torch.randn(1, 4, 160, 160, device=device)
    modality2 = torch.randn(1, 2, 160, 160, device=device)

    net = SSUCCNet(encoder_name='mit_b4', encoder_weights='imagenet', classes=7)
    net.to(device)
    net.eval()

    pred = net(modality1, modality2)
    print(pred.shape)
