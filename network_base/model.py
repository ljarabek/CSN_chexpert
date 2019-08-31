import torch
import torch.nn as nn
import torch.nn.functional as F

from network_base.densenet import densenet121, densenet121_CSN

"""
This code is to get the features of densenet using different settings.

# usage #1:
# ### TO TRAIN GLOBAL MODEL:
# learn_feature_global_pool=False, give only ONE input to function forward(), like:
# >>> x = image
# >>> res = model(x)

# ### TO TRAIN SEG_SUBIMAGE MODEL:
# learn_feature_global_pool=False, give TWO inputs (subimage and seg mask) to function forward(), like:
# >>> x,y = [sub_image, seg_mask]
# >>> res = model(x,y)

# ### FOR BOTH CASES:
# line 67~71: use get_xxxx_prob()


# usage #2:
# ### TO TRAIN FUSION NETWORK: (for both global net and sub_img net)
# line 67~71: use get_xxxx_features()
"""

class ResNet50(nn.Module):
    """
    # add parameters here
    """
    def __init__(self, pretrained=False, num_classes=14,
                 groups=1, width_per_group=64, **kwargs):
        super(ResNet50, self).__init__()

        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(planes[3] * block.expansion, num_classes) # need to add block to __init__

        self.resnet50 = resnet.resnet50(pretrained=pretrained, num_classes=num_classes, **kwargs)
        num_ftrs = self.resnet50.avgpool
        self.resnet50.classifier = nn.Sequential(
            # set nn.Linear to linear transformation of 14 features to 14 features
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.resnet50.forward(x) # forward here is wrong
        out = F.relu(features, inplace=True)
        out = self.avgpool(out)
        out = out.view(x.size(0), -1)
        # out = self.fc(out)
        out = self.resnet50.classifier(out)
        return out

class DenseNet121(nn.Module):
    """
    :param pretrained: loading weights from official weights. (True)
    :param attention: using attention mechanism (channel & spatial) for each denseblock. (True)
    :param dilation_config: replacing convolution with dilation convolution. (True)
    :param drop_rate: rate for dropout layer.
    :param num_cls: the number of the pathologies.
    :param use_softmax: outputs are softmax normalized. (True)
    :param learn_feature_global_pool: TODO
    """

    def __init__(self, pretrained=True, attention=False, dilation_config=(False, False, False, False),
                 drop_rate=0, num_cls=14, use_softmax=False, no_sigmoid=False, learn_feature_global_pool=False):
        super(DenseNet121, self).__init__()
        # conduct the official densenet
        self.densenet121 = densenet121(pretrained=pretrained, attention=attention,
                                       dilation_config=dilation_config, drop_rate=drop_rate)
        num_ftrs = self.densenet121.classifier.in_features
        # base classifier
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_cls),
            nn.Sigmoid())
        if no_sigmoid:
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_ftrs, num_cls))
        # softmax normalized (overwrite the default classfier)
        if use_softmax == True:
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_ftrs, num_cls),
                nn.Softmax())
        # TODO
        if learn_feature_global_pool == True:
            self.densenet121.seg_pool = nn.Sequential(
                nn.Conv2d(1, num_ftrs // 8, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(num_ftrs // 8), nn.ReLU(inplace=True),
                nn.Conv2d(num_ftrs // 8, num_ftrs // 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_ftrs // 4), nn.ReLU(inplace=True),
                nn.Conv2d(num_ftrs // 4, num_ftrs, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=2, padding=1, bias=False, groups=num_ftrs),
                nn.BatchNorm2d(num_ftrs), nn.ReLU(inplace=True))

            self.densenet121.seg_pool_for_res4 = nn.Sequential(
                nn.Conv2d(1, num_ftrs // 8, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(num_ftrs // 8), nn.ReLU(inplace=True),
                nn.Conv2d(num_ftrs // 8, num_ftrs // 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_ftrs // 4), nn.ReLU(inplace=True),
                nn.Conv2d(num_ftrs // 4, num_ftrs, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_ftrs), nn.ReLU(inplace=True))

            self.fuse_res4_res5 = nn.Sequential(nn.Linear(num_ftrs, num_ftrs), nn.BatchNorm1d(num_ftrs))
            self.alpha_res4 = nn.Parameter(torch.zeros(1))

    def forward(self, x, y=None):
        if y is not None:
            return self.get_seg_pooled_res4res5_features(x, y)
            # return self.get_seg_pooled_res4res5_prob(x, y)
        else:
            # return self.get_res5_features(x)
            return self.get_prob(x)

    def get_prob(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).view(features.size(0), -1)
        out = self.densenet121.classifier(out)
        return out

    # TODO: functions below are not used for base classification network
    def get_seg_pooled_res4res5_features(self, x, y):
        # base conv
        features = self.densenet121.features.pool0(
            self.densenet121.features.relu0(self.densenet121.features.norm0(self.densenet121.features.conv0(x))))
        # block1
        features = self.densenet121.features.transition1(self.densenet121.features.denseblock1(features))
        features = self.densenet121.features.attention1_2(self.densenet121.features.attention1_1(features))
        # block2
        features = self.densenet121.features.transition2(self.densenet121.features.denseblock2(features))
        features = self.densenet121.features.attention2_2(self.densenet121.features.attention2_1(features))
        # block3
        features_res4 = self.densenet121.features.denseblock3(features)
        features = self.densenet121.features.transition3(features_res4)
        features = self.densenet121.features.attention3_2(self.densenet121.features.attention3_1(features))
        # block4
        features = self.densenet121.features.denseblock4(features)
        features = self.densenet121.features.attention4_2(self.densenet121.features.attention4_1(features))
        features = self.densenet121.features.norm5(features)
        features_res5 = self.densenet121.features.attention5_2(self.densenet121.features.attention5_1(features))

        y = F.interpolate(y, size=128, mode='nearest')
        w_res4 = self.densenet121.seg_pool_for_res4(y)
        w_res5 = self.densenet121.seg_pool(y)

        features_res4 = (features_res4 * w_res4 + features_res4) / 2
        features_res4 = F.adaptive_avg_pool2d(features_res4, 1).view(features_res4.size(0), -1)
        features_res4 = F.relu(features_res4, inplace=True)

        features_res5 = (features_res5 * w_res5 + features_res5) / 2
        features_res5 = F.adaptive_avg_pool2d(features_res5, 1).view(features_res5.size(0), -1)
        features_res5 = F.relu(features_res5, inplace=True)

        features = torch.cat([features_res4, features_res5], dim=1)
        features_res4 = self.fuse_res4_res5(features_res4)
        features = self.alpha_res4 * features_res4 + features_res5
        return features

    def get_seg_pooled_res4res5_features_heatmap(self, x, y):
        # base conv
        features = self.densenet121.features.pool0(
            self.densenet121.features.relu0(self.densenet121.features.norm0(self.densenet121.features.conv0(x))))
        # block1
        features = self.densenet121.features.transition1(self.densenet121.features.denseblock1(features))
        features = self.densenet121.features.attention1_2(self.densenet121.features.attention1_1(features))
        # block2
        features = self.densenet121.features.transition2(self.densenet121.features.denseblock2(features))
        features = self.densenet121.features.attention2_2(self.densenet121.features.attention2_1(features))
        # block3
        features_res4 = self.densenet121.features.denseblock3(features)
        features = self.densenet121.features.transition3(features_res4)
        features = self.densenet121.features.attention3_2(self.densenet121.features.attention3_1(features))
        # block4
        features = self.densenet121.features.denseblock4(features)
        features = self.densenet121.features.attention4_2(self.densenet121.features.attention4_1(features))
        features = self.densenet121.features.norm5(features)
        features_res5 = self.densenet121.features.attention5_2(self.densenet121.features.attention5_1(features))

        y = F.interpolate(y, size=128, mode='nearest')
        w_res4 = self.densenet121.seg_pool_for_res4(y)
        w_res5 = self.densenet121.seg_pool(y)

        features_res4 = (features_res4 * w_res4 + features_res4) / 2
        features_res4 = F.relu(features_res4, inplace=True)

        features_res5 = (features_res5 * w_res5 + features_res5) / 2
        features_res5 = F.relu(features_res5, inplace=True)

        tmp = self.fuse_res4_res5.state_dict()
        features_res4 = F.conv2d(features_res4,
                                 tmp['0.weight'].view(tmp['0.weight'].size(0), tmp['0.weight'].size(1), 1, 1), stride=2)
        features_res4 = F.batch_norm(features_res4, tmp['1.running_mean'], tmp['1.running_var'], weight=tmp['1.weight'],
                                     bias=tmp['1.bias'], training=False)
        # features_res4 = self.fuse_res4_res5(features_res4)
        features = self.alpha_res4 * features_res4 + features_res5
        return features

    def ccc(self, x, y):
        features = self.get_seg_pooled_res4res5_features(x, y)
        features = self.densenet121.classifier(features)
        return features

    def get_seg_pooled_res5_features(self, x, y):
        features = self.get_res5_features(x)
        y = F.interpolate(y, size=128, mode='nearest')
        w = self.densenet121.seg_pool(y)
        features = (features * w + features) / 2
        return features

    def get_seg_pooled_prob(self, x, y):
        features = self.get_res5_features(x)
        y = F.interpolate(y, size=128, mode='nearest')
        w = self.densenet121.seg_pool(y)
        features = (features * w + features) / 2
        features = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        features = F.relu(features, inplace=True)
        features = self.densenet121.classifier(features)
        return features

    def get_res5_features(self, x):
        return self.densenet121.features(x)

    def get_fc_features(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        return F.adaptive_avg_pool2d(out, 1).view(features.size(0), -1)

    def get_cls_heatmap(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        self.densenet121.conv_cls = nn.Conv2d(1024, 14, kernel_size=1, bias=True).cuda()
        s1 = self.densenet121.conv_cls.state_dict()
        s1['weight'] = self.densenet121.classifier[0].state_dict()['weight'].unsqueeze(-1).unsqueeze(-1)
        s1['bias'] = self.densenet121.classifier[0].state_dict()['bias']
        self.densenet121.conv_cls.load_state_dict(s1)
        out = self.densenet121.conv_cls(out)
        out = F.sigmoid(out)
        return out

    def get_internal_features(self, x, attention=False):
        internal = []
        # base conv
        features = self.densenet121.features.pool0(
            self.densenet121.features.relu0(
                self.densenet121.features.norm0(
                    self.densenet121.features.conv0(x))))
        internal.append(features)
        # block1
        features = self.densenet121.features.transition1(
            self.densenet121.features.denseblock1(features))
        if attention:
            features = self.densenet121.attention.attention1(features)
        internal.append(features)
        # block2
        features = self.densenet121.features.transition2(
            self.densenet121.features.denseblock2(features))
        if attention:
            features = self.densenet121.attention.attention2(features)
        internal.append(features)
        # block3
        features = self.densenet121.features.transition3(
            self.densenet121.features.denseblock3(features))
        if attention:
            features = self.densenet121.attention.attention3(features)
        internal.append(features)
        # block4
        features = self.densenet121.features.norm5(
            self.densenet121.features.denseblock4(features))
        if attention:
            features = self.densenet121.attention.attention4(features)
        internal.append(features)
        # classifier
        # features = F.relu(features, inplace=True)
        # features = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        # features = self.densenet121.classifier(features)
        return internal


class CSN(nn.Module):
    """
    :param pretrained: loading weights from official weights. (True)
    :param attention: using attention mechanism (channel & spatial) for each denseblock. (True)
    :param dilation_config: replacing convolution with dilation convolution. (True)
    :param drop_rate: rate for dropout layer.
    :param num_cls: the number of the pathologies.
    :param use_softmax: outputs are softmax normalized. (True)
    :param learn_feature_global_pool: TODO
    """

    def __init__(self, pretrained=False, attention=False, dilation_config=(False, False, False, False),
                 drop_rate=0, num_cls=2, use_softmax=False, no_sigmoid=True, learn_feature_global_pool=False): # nosigmoid -> True, pretrained -> False
        super(CSN, self).__init__()
        # conduct the official densenet
        self.densenet121 = densenet121_CSN(pretrained=pretrained, attention=attention,
                                       dilation_config=dilation_config, drop_rate=drop_rate)
        num_ftrs = self.densenet121.classifier.in_features
        num_ftrs = 1024
        # base classifier
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_cls),
            nn.Sigmoid())
        if no_sigmoid:
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_ftrs, num_cls))
        # softmax normalized (overwrite the default classfier)
        if use_softmax == True:
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_ftrs, num_cls),
                nn.Softmax())
        # TODO
        if learn_feature_global_pool == True:
            self.densenet121.seg_pool = nn.Sequential(
                nn.Conv2d(1, num_ftrs // 8, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(num_ftrs // 8), nn.ReLU(inplace=True),
                nn.Conv2d(num_ftrs // 8, num_ftrs // 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_ftrs // 4), nn.ReLU(inplace=True),
                nn.Conv2d(num_ftrs // 4, num_ftrs, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=2, padding=1, bias=False, groups=num_ftrs),
                nn.BatchNorm2d(num_ftrs), nn.ReLU(inplace=True))

            self.densenet121.seg_pool_for_res4 = nn.Sequential(
                nn.Conv2d(1, num_ftrs // 8, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(num_ftrs // 8), nn.ReLU(inplace=True),
                nn.Conv2d(num_ftrs // 8, num_ftrs // 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_ftrs // 4), nn.ReLU(inplace=True),
                nn.Conv2d(num_ftrs // 4, num_ftrs, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_ftrs), nn.ReLU(inplace=True))

            self.fuse_res4_res5 = nn.Sequential(nn.Linear(num_ftrs, num_ftrs), nn.BatchNorm1d(num_ftrs))
            self.alpha_res4 = nn.Parameter(torch.zeros(1))

    def forward(self, x, y=None):
        if y is not None:
            return self.get_seg_pooled_res4res5_features(x, y)
            # return self.get_seg_pooled_res4res5_prob(x, y)
        else:
            # return self.get_res5_features(x)
            return self.get_prob(x)

    def get_prob(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).view(features.size(0), -1)
        out = self.densenet121.classifier(out)


        beta = out[:, 0]
        gamma = out[:, 1]
        images = torch.transpose(x, 0, 3)
        images = gamma * images
        images = images + beta
        images = torch.transpose(images, 0, 3)
        out_im = torch.tanh(images)

        return out_im

    # TODO: functions below are not used for base classification network
    def get_seg_pooled_res4res5_features(self, x, y):
        # base conv
        features = self.densenet121.features.pool0(
            self.densenet121.features.relu0(self.densenet121.features.norm0(self.densenet121.features.conv0(x))))
        # block1
        features = self.densenet121.features.transition1(self.densenet121.features.denseblock1(features))
        features = self.densenet121.features.attention1_2(self.densenet121.features.attention1_1(features))
        # block2
        features = self.densenet121.features.transition2(self.densenet121.features.denseblock2(features))
        features = self.densenet121.features.attention2_2(self.densenet121.features.attention2_1(features))
        # block3
        features_res4 = self.densenet121.features.denseblock3(features)
        features = self.densenet121.features.transition3(features_res4)
        features = self.densenet121.features.attention3_2(self.densenet121.features.attention3_1(features))
        # block4
        features = self.densenet121.features.denseblock4(features)
        features = self.densenet121.features.attention4_2(self.densenet121.features.attention4_1(features))
        features = self.densenet121.features.norm5(features)
        features_res5 = self.densenet121.features.attention5_2(self.densenet121.features.attention5_1(features))

        y = F.interpolate(y, size=128, mode='nearest')
        w_res4 = self.densenet121.seg_pool_for_res4(y)
        w_res5 = self.densenet121.seg_pool(y)

        features_res4 = (features_res4 * w_res4 + features_res4) / 2
        features_res4 = F.adaptive_avg_pool2d(features_res4, 1).view(features_res4.size(0), -1)
        features_res4 = F.relu(features_res4, inplace=True)

        features_res5 = (features_res5 * w_res5 + features_res5) / 2
        features_res5 = F.adaptive_avg_pool2d(features_res5, 1).view(features_res5.size(0), -1)
        features_res5 = F.relu(features_res5, inplace=True)

        features = torch.cat([features_res4, features_res5], dim=1)
        features_res4 = self.fuse_res4_res5(features_res4)
        features = self.alpha_res4 * features_res4 + features_res5
        return features

    def get_seg_pooled_res4res5_features_heatmap(self, x, y):
        # base conv
        features = self.densenet121.features.pool0(
            self.densenet121.features.relu0(self.densenet121.features.norm0(self.densenet121.features.conv0(x))))
        # block1
        features = self.densenet121.features.transition1(self.densenet121.features.denseblock1(features))
        features = self.densenet121.features.attention1_2(self.densenet121.features.attention1_1(features))
        # block2
        features = self.densenet121.features.transition2(self.densenet121.features.denseblock2(features))
        features = self.densenet121.features.attention2_2(self.densenet121.features.attention2_1(features))
        # block3
        features_res4 = self.densenet121.features.denseblock3(features)
        features = self.densenet121.features.transition3(features_res4)
        features = self.densenet121.features.attention3_2(self.densenet121.features.attention3_1(features))
        # block4
        features = self.densenet121.features.denseblock4(features)
        features = self.densenet121.features.attention4_2(self.densenet121.features.attention4_1(features))
        features = self.densenet121.features.norm5(features)
        features_res5 = self.densenet121.features.attention5_2(self.densenet121.features.attention5_1(features))

        y = F.interpolate(y, size=128, mode='nearest')
        w_res4 = self.densenet121.seg_pool_for_res4(y)
        w_res5 = self.densenet121.seg_pool(y)

        features_res4 = (features_res4 * w_res4 + features_res4) / 2
        features_res4 = F.relu(features_res4, inplace=True)

        features_res5 = (features_res5 * w_res5 + features_res5) / 2
        features_res5 = F.relu(features_res5, inplace=True)

        tmp = self.fuse_res4_res5.state_dict()
        features_res4 = F.conv2d(features_res4,
                                 tmp['0.weight'].view(tmp['0.weight'].size(0), tmp['0.weight'].size(1), 1, 1), stride=2)
        features_res4 = F.batch_norm(features_res4, tmp['1.running_mean'], tmp['1.running_var'], weight=tmp['1.weight'],
                                     bias=tmp['1.bias'], training=False)
        # features_res4 = self.fuse_res4_res5(features_res4)
        features = self.alpha_res4 * features_res4 + features_res5
        return features

    def ccc(self, x, y):
        features = self.get_seg_pooled_res4res5_features(x, y)
        features = self.densenet121.classifier(features)
        return features

    def get_seg_pooled_res5_features(self, x, y):
        features = self.get_res5_features(x)
        y = F.interpolate(y, size=128, mode='nearest')
        w = self.densenet121.seg_pool(y)
        features = (features * w + features) / 2
        return features

    def get_seg_pooled_prob(self, x, y):
        features = self.get_res5_features(x)
        y = F.interpolate(y, size=128, mode='nearest')
        w = self.densenet121.seg_pool(y)
        features = (features * w + features) / 2
        features = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        features = F.relu(features, inplace=True)
        features = self.densenet121.classifier(features)
        return features

    def get_res5_features(self, x):
        return self.densenet121.features(x)

    def get_fc_features(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        return F.adaptive_avg_pool2d(out, 1).view(features.size(0), -1)

    def get_cls_heatmap(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        self.densenet121.conv_cls = nn.Conv2d(1024, 14, kernel_size=1, bias=True).cuda()
        s1 = self.densenet121.conv_cls.state_dict()
        s1['weight'] = self.densenet121.classifier[0].state_dict()['weight'].unsqueeze(-1).unsqueeze(-1)
        s1['bias'] = self.densenet121.classifier[0].state_dict()['bias']
        self.densenet121.conv_cls.load_state_dict(s1)
        out = self.densenet121.conv_cls(out)
        out = F.sigmoid(out)
        return out

    def get_internal_features(self, x, attention=False):
        internal = []
        # base conv
        features = self.densenet121.features.pool0(
            self.densenet121.features.relu0(
                self.densenet121.features.norm0(
                    self.densenet121.features.conv0(x))))
        internal.append(features)
        # block1
        features = self.densenet121.features.transition1(
            self.densenet121.features.denseblock1(features))
        if attention:
            features = self.densenet121.attention.attention1(features)
        internal.append(features)
        # block2
        features = self.densenet121.features.transition2(
            self.densenet121.features.denseblock2(features))
        if attention:
            features = self.densenet121.attention.attention2(features)
        internal.append(features)
        # block3
        features = self.densenet121.features.transition3(
            self.densenet121.features.denseblock3(features))
        if attention:
            features = self.densenet121.attention.attention3(features)
        internal.append(features)
        # block4
        features = self.densenet121.features.norm5(
            self.densenet121.features.denseblock4(features))
        if attention:
            features = self.densenet121.attention.attention4(features)
        internal.append(features)
        # classifier
        # features = F.relu(features, inplace=True)
        # features = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        # features = self.densenet121.classifier(features)
        return internal
