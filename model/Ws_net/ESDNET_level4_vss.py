"""
Implementation of ESDNet for image demoireing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parameter import Parameter
from modules.mamba import *
from modules.modules import *
class ESDNET(nn.Module):
    def __init__(self,
                 en_feature_num=48,
                 en_inter_num=32,
                 de_feature_num=48,
                 de_inter_num=32,
                 sam_number=1,
                 ):
        super(ESDNET, self).__init__()
        self.encoder = Encoder(feature_num=en_feature_num, inter_num=en_inter_num, sam_number=sam_number)
        self.decoder = Decoder(en_num=en_feature_num, feature_num=de_feature_num, inter_num=de_inter_num,
                               sam_number=sam_number)
        self.CBAM = CBAM(8*en_feature_num)
    def forward(self, x):
        ori_shape = x[0,0,:,:].shape
        y,y_1, y_2, y_3,y_4 = self.encoder(x)
        y_4 = y_4+self.CBAM(y_4)
        out_1 = self.decoder(y,y_1, y_2, y_3,y_4)
        return out_1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, sam_number):
        super(Decoder, self).__init__()
        self.preconv_3 = conv_relu(8 * en_num*2, feature_num*4, 3, padding=1)
        self.decoder_4 = VSSLayer(dim=8 * feature_num, depth=1,d_state=16,drop_path=[0.2])
        self.preconv_2 = conv_relu(4 * en_num*2, feature_num*2, 3, padding=1)
        self.decoder_3 = Decoder_Level(feature_num*4, inter_num, sam_number)

        # self.preconv_2 = conv_relu(2 * en_num*2, feature_num*1, 3, padding=1)
        self.decoder_2 = Decoder_Level(feature_num*2, inter_num, sam_number)

        self.preconv_1 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level(feature_num, inter_num, sam_number)

        self.head = conv(in_channel=feature_num*2, out_channel=1, kernel_size=3, padding=1)
    def forward(self, y,y_1, y_2, y_3,y_4):
        x_4 = y_4
        x_4 = self.decoder_4(x_4.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous() # 384 32 32
        x_3 = torch.cat([y_3, x_4], dim=1) # 768 32 32
        x_3 = self.preconv_3(x_3) # 192 32 32
        x_3 = self.decoder_3(x_3) # 192 64 64

        x_2 = torch.cat([y_2, x_3], dim=1) # 384 64 64
        x_2 = self.preconv_2(x_2) # 96 64 64
        x_1 = self.decoder_2(x_2) # 96 128 128

        x_1 = torch.cat([y_1, x_1], dim=1) # 192 128 128
        x_1 = self.preconv_1(x_1)  # 48 128 128
        x = self.decoder_1(x_1) #48 256 256
        out = self.head(torch.cat([x,y],1)) # 1 * 256 256
        return out


class Encoder(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Encoder, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(1, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.encoder_1 = Encoder_Level(feature_num, inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, sam_number=sam_number)
        self.encoder_4 = VSSLayer(dim=8 * feature_num, depth=1,d_state=16,drop_path=[0.2])
    def forward(self, x):

        #x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)  # 48 256 256
        out_feature_1 = self.encoder_1(x) # 96 128 128
        out_feature_2 = self.encoder_2(out_feature_1) #192 64 64
        out_feature_3 = self.encoder_3(out_feature_2) #384 32 32

        out_feature_4 = self.encoder_4(out_feature_3.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous() #384 32 32

        #16 48 128 128
        #16 96 64 64
        #16 192 32 32
        # 16 384 16 16
        return x,out_feature_1, out_feature_2, out_feature_3,out_feature_4


class Encoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number):
        super(Encoder_Level, self).__init__()
        self.rdb = RDB(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)

        if level < 4:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.level = level

    def forward(self, x):
        out_feature = self.rdb(x)
        for sam_block in self.sam_blocks:
            out_feature = sam_block(out_feature)
        if self.level < 4:
            out_feature = self.down(out_feature)
            return out_feature
        return out_feature


class Decoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Decoder_Level, self).__init__()
        self.rdb = RDB(feature_num, (1, 2, 1), inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)
        # self.conv = conv(in_channel=feature_num, out_channel=4, kernel_size=3, padding=1)

    def forward(self, x, feat=True):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.rdb(x)
        for sam_block in self.sam_blocks:
            x = sam_block(x)
        # out = self.conv(x)
        # out = F.pixel_shuffle(out, 2)

        # if feat:
        #     feature = F.interpolate(x, scale_factor=2, mode='bilinear')
        #     return out, feature
        # else:
        return x


class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class SAM(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(SAM, self).__init__()
        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.fusion = CSAF(3 * in_channel)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)
        y = x + y

        return y


class CSAF(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(CSAF, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):

        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4

        return x


class RDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(RDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)
        return t + x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out
