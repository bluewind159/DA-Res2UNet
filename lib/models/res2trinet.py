import torch
from torch import nn, Tensor
from .splat import SplAtConv2d
from dropblock import DropBlock2D
from .modules import STNLayer
class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)
        
class Res2WNet(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(Res2WNet, self).__init__()
        n1 = 64
        #print('Res2Net64+drop')
        print('Res2WNet')
        self.drop=DropBlock2D(block_size=7, drop_prob=0.1)
        
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.step1 = DoubleConv(in_channel, filters[0])
        self.step2 = DoubleConv(filters[0], filters[1])
        self.step3 = DoubleConv(filters[1], filters[2])
        self.step4 = DoubleConv(filters[2], filters[3])
        self.step5 = DoubleConv(filters[3], filters[4])
        self.splat = SPAConv(filters[4], filters[4])
        self.splat_w = SPAConv(filters[4], filters[4])
        #self.splat =  SplAtConv2d(
        #        filters[4], filters[4], kernel_size=3,
        #        stride=1, padding=1, bias=False, dropblock_prob=0.0, radix=1),
        self.step6 = Up(filters[4]*2+filters[3]*2, filters[3])
        self.step7 = Up(filters[3]+filters[2]*2, filters[2])
        self.step8 = Up(filters[2]+filters[1]*2, filters[1])
        self.step9 = Up(filters[1]+filters[0]*2, filters[0])
        self.step10 = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )
        self.step1w = DoubleConv(in_channel, filters[0])
        self.step2w = DoubleConv(filters[0], filters[1])
        self.step3w = DoubleConv(filters[1], filters[2])
        self.step4w = DoubleConv(filters[2], filters[3])
        self.step5w = DoubleConv(filters[3], filters[4])
        
        self.mp = nn.MaxPool2d(2)
        self.pam = _PositionAttentionModule(filters[4])
        self.cam = _ChannelAttentionModule()
        self.pam_w = _PositionAttentionModule(filters[4])
        self.cam_w = _ChannelAttentionModule()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, x: Tensor, xw: Tensor) -> Tensor:
        x1 = self.step1(x)
        # print('x1:', x1.shape)
        x = self.mp(x1)
        x2 = self.step2(x)
        # print('x2:', x2.shape)
        x = self.mp(x2)
        x3 = self.step3(x)
        # print('x3:',x3.shape)
        x = self.mp(x3)
        
        x4 = self.step4(x)
        # print('x4:', x4.shape)
        x = self.mp(x4)
        x5 = self.step5(x)
        # print('x5:', x5.shape)
        x_s=self.splat(x5)
        x_p=self.pam(x_s)
        x_c=self.cam(x_s)
        x_fusion = x_p + x_c
        x1w = self.step1w(xw)
        xw = self.mp(x1w)
        x2w = self.step2w(xw)
        xw = self.mp(x2w)
        x3w = self.step3w(xw)
        xw = self.mp(x3w)
        x4w = self.step4w(xw)
        xw = self.mp(x4w)
        x5w = self.step5w(xw)
        x_s_w=self.splat_w(x5w)
        x_p_w=self.pam_w(x_s_w)
        x_c_w=self.cam_w(x_s_w)
        x_fusion_w = x_p_w + x_c_w
        final=torch.cat([x_fusion,x_fusion_w],dim=1)
        x6 = self.step6(final, torch.cat([x4,x4w],dim=1))
        x7 = self.step7(x6, torch.cat([x3,x3w],dim=1))
        x8 = self.step8(x7, torch.cat([x2,x2w],dim=1))
        x9 = self.step9(x8, torch.cat([x1,x1w],dim=1))
        x10 = self.step10(x9)
        return x10
        
class Up(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        #self.douconv = nn.Sequential(
        #    nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        #    nn.BatchNorm2d(out_channel),
        #    nn.ReLU(inplace=True),
        #    #DropBlock2D(block_size=7, drop_prob=0.1),
        #    nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
        #    nn.BatchNorm2d(out_channel),
        #    nn.ReLU(inplace=True),
        #    #DropBlock2D(block_size=7, drop_prob=0.1)
        #)
        self.douconv = DoubleConv(in_channel,out_channel)
    def forward(self, x: Tensor, x1: Tensor) -> Tensor:
        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = self.douconv(x)
        return x
        
class NormalConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(NormalConv, self).__init__()
        self.douconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            #DropBlock2D(block_size=7, drop_prob=0.1),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            #DropBlock2D(block_size=7, drop_prob=0.1)
        )
    def forward(self, x: Tensor) -> Tensor:
        x = self.douconv(x)
        return x
        
class SPAConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(SPAConv, self).__init__()
        self.douconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            #DropBlock2D(block_size=7, drop_prob=0.1),
            SplAtConv2d(
                in_channel, out_channel, kernel_size=3,
                stride=1, padding=1, bias=False, dropblock_prob=0.0,radix=2),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            #DropBlock2D(block_size=7, drop_prob=0.1)
        )
    def forward(self, x: Tensor) -> Tensor:
        x = self.douconv(x)
        return x
        
        
class DoubleConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, scale: int = 4) -> None:
        super(DoubleConv, self).__init__()
        self.nums = scale - 1
        self.width = int(out_channel/scale)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        convs, bns = [], []
        for i in range(self.nums):
            #convs.append(SplAtConv2d(
            #    self.width, self.width, kernel_size=3,
            #    stride=1, padding=1, bias=False, dropblock_prob=0.0, radix=2))
            convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(self.width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = nn.ReLU(inplace=True)
        self.drop = DropBlock2D(block_size=7, drop_prob=0.1)
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        # print('re_init:', residual.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        out = self.relu(x)
        # print('out1:', out.shape)
        spx = torch.split(out, self.width, dim=1)
        # print('spx:', len(spx))
        for i in range(self.nums):
            sp = spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            # print('sp{}:'.format(i), sp.shape)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), dim=1)
            # print('out{}:'.format(i), out.shape)
        out = torch.cat((out, spx[self.nums]), dim=1)
        # print('outshape:', out.shape, 'outtype:', type(out))
        # print('residual:', residual.shape, type(residual))
        # out += residual
        out = self.relu(out)
        out=self.drop(out)
        return out