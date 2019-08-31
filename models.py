from torchvision.models.densenet import DenseNet
from network_base.model import DenseNet121
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np


class CSN_backbone(nn.Module):
    def __init__(self, args):
        super(CSN_backbone, self).__init__()
        self.args = args

        self.backbone = DenseNet(growth_rate=32, block_config=(6, 12, 6, 6), num_init_features=3, bn_size=4,
                                 drop_rate=0, num_classes=2)
        self.classifier = self.backbone.classifier
        self.tanh = nn.Tanh()
        self.visualize = False
        self.visualization_filename = str()

    def forward(self, x):
        features = self.backbone.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.args.crop_size // 32, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        # print(x.size())

        beta = out[:, 0]
        gamma = out[:, 1]
        gamma += 0.5  # initial must not be <0!!!
        images = torch.transpose(x, 0, 3)
        images = gamma * images
        images = images + beta
        images = torch.transpose(images, 0, 3)
        out_im = torch.tanh(images)

        if self.visualize and self.args.visualize_:
            plt.clf()
            fig, ((inp_, inp_hist), (out_inter, out_inter_hist), (out_, out_hist)) = plt.subplots(3, 2)

            vis_x = x[0].cpu().detach().numpy()
            im   = vis_x
            for idc, ch in enumerate(im):
                ch += np.abs(np.min(ch))
                ch /= np.max(ch)
                im[idc] = ch
            inp_.imshow(np.moveaxis(im, 0, -1))
            inp_.set_title("input-unnormalized")

            inp_hist.hist(np.ravel(im), bins=30)
            inp_hist.set_title("Un-normalized")
            inp_im = x[0].cpu().detach().numpy()
            for idc, ch in enumerate(inp_im):
                ch += np.abs(np.min(ch))
                ch /= np.max(ch)
                inp_im[idc] = ch

            out_inter.imshow(np.moveaxis(inp_im, 0, -1))
            out_inter.set_title("input image")

            out_inter_hist.hist(np.ravel(x[0].cpu().detach().numpy()), bins=30)
            out_inter_hist.set_title("input image")

            im_out = out_im[0].cpu().detach().numpy()  # )

            for idc, ch in enumerate(im_out):
                ch += np.abs(np.min(ch))
                ch /= np.max(ch)
                im_out[idc] = ch

            out_.imshow(np.moveaxis(im_out, 0, -1))
            out_.set_title("transformed image")

            out_hist.hist(np.ravel(out_im[0].cpu().detach().numpy()), bins=30)
            out_hist.set_title(str(out[0].cpu().detach().numpy()))

            plt.savefig(self.visualization_filename)
        return out_im


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.CSN = CSN_backbone(self.args)
        if self.args.load_model_path is not None:
            pretrained = False
        else:
            pretrained = True
        self.classifier = DenseNet121(attention=False, pretrained=pretrained,
                                      dilation_config=(False, False, False, False))

    def forward(self, x):
        if self.args.CSN:
            image = self.CSN(x)
            preds = self.classifier(image)
        else:
            preds = self.classifier(x)
        return preds



