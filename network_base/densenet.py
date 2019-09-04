import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

#from module.attention import ChannelAttention, SpatialAttention


class DenseLayer(nn.Sequential):
    """Dense Layer"""

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DilatedDenseLayer(nn.Sequential):
    """Dense Layer using dilation convolution, default dilation rate is 2"""

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation=2):
        super(DilatedDenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, dilation=dilation, stride=1, padding=dilation, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DilatedDenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    """Dense Block --> stacked (Dilation) Dense Layer"""

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, is_dilated):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            if is_dilated == True:
                layer = DilatedDenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate,
                                          dilation=i % 3 + 1)
            else:
                layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class Transition(nn.Sequential):
    """Transition Layer between different Dense Blocks"""

    def __init__(self, num_input_features, num_output_features, stride=2):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        if stride == 2:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        elif stride == 1:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))


class DenseNet(nn.Module):
    """Densne Net"""

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
                 attention=False, dilation_config=(False, False, False, True), no_channels = 3):
        super(DenseNet, self).__init__()
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(no_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        self.attention = nn.Sequential()
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features,
                               bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate,
                               is_dilated=dilation_config[i])
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                # ---- Attention module ----
            if attention:
                pass
                # self.features.add_module('attention%d_1' % (i + 1), ChannelAttention(num_features))
                # self.features.add_module('attention%d_2' % (i + 1), SpatialAttention(num_features))
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        # ---- Final Attention module ----
        if attention:
            pass
            #self.features.add_module('attention5_1', ChannelAttention(num_features))
            #self.features.add_module('attention5_2', SpatialAttention(num_features))
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from pytorch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet121(pretrained=False, attention=False, dilation_config=(False, False, False, False), drop_rate=0, **kwargs):
    """
    Densenet-121 model from <https://arxiv.org/pdf/1608.06993.pdf>
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), attention=attention,
                     dilation_config=dilation_config, drop_rate=drop_rate, **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model.state_dict()
        state_dict_pretrained = torch.load('./checkpoints/densenet/densenet121.pth')
        for key in list(state_dict_pretrained.keys()):
            if key not in state_dict:
                res = pattern.match(key)
                if res:  # for res block params
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict_pretrained[key]
                else:
                    print('Ignore layer {}'.format(key))
                    continue
            else:  # for base conv params
                state_dict[key] = state_dict_pretrained[key]
        model.load_state_dict(state_dict, strict=False)
        print('success in loading weights!')
    return model

def densenet121_CSN(pretrained=False, attention=False, dilation_config=(False, False, False, False), drop_rate=0, **kwargs):
    """
    Densenet-121 model from <https://arxiv.org/pdf/1608.06993.pdf>
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), attention=attention,
                     dilation_config=dilation_config, drop_rate=drop_rate, **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model.state_dict()
        state_dict_pretrained = torch.load('./checkpoints/densenet/densenet121.pth')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        num_ftrs = 64
        model.classifier = nn.Linear(num_ftrs, 2)


        # print('checkpoint: ', state_dict_pretrained)
        # print('checkpoint keys: ', state_dict_pretrained.keys())
        # # print('checkpoint reshape to 2,1024: ')
        # for key in list(state_dict_pretrained.keys()):
        #     if key not in state_dict:
        #         res = pattern.match(key)
        #         if res:  # for res block params
        #             new_key = res.group(1) + res.group(2)
        #             state_dict[new_key] = state_dict_pretrained[key]
        #         else:
        #             print('Ignore layer {}'.format(key))
        #             continue
        #     else:  # for base conv params
        #         state_dict[key] = state_dict_pretrained[key]
        # model.load_state_dict(state_dict, strict=False)
        # print('success in loading weights!')
    return model
