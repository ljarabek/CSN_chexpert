import torch
import torch.nn as nn

def cal_weight(labels):
    """calculate weights for different categories

    up-weights the underrepresented class :)

    """
    P = 0
    N = 0
    for label in labels:
        for v in label:
            if int(v) == 1:
                P += 1
            elif int(v) == 0:
                N += 1
            # if int(v) == 2
            # the label is unknown and we add a weight of 0.1 to the positive label
            else:
                P += 0.1
    if P != 0 and N != 0:
        BP = (P + N) / P
        BN = (P + N) / N
        weights = torch.FloatTensor([BP, BN]).cuda()
    else:
        weights = None

    return weights

class MultiClassLoss(nn.Module):
    """
    Softmax Cross Entropy for Multi-class Classification
    :param num_cls: number of the pathologies
    :param num_pcls: number of the labels for each pathology
    """

    def __init__(self, num_cls, num_pcls):
        super(MultiClassLoss, self).__init__()
        self.num_cls = num_cls
        self.num_pcls = num_pcls
        self.criterion = nn.CrossEntropyLoss(weight=None)

    def forward(self, preds, targets):
        total_loss = torch.FloatTensor([0]).cuda()
        preds = preds.unsqueeze(-1).unsqueeze(-1)
        batch, channel, height, width = preds.shape
        pred_x = preds.reshape(batch, self.num_cls, self.num_pcls)
        for idx in range(self.num_cls):
            pred = pred_x[:, idx]
            target = targets[:, idx]
            loss = self.criterion(pred, target)
            total_loss += loss
        return total_loss / self.num_cls


class WeightBCELoss(nn.Module):
    """Weighted Binary Cross Entropy for Classification"""

    def __init__(self):
        super(WeightBCELoss, self).__init__()

    def forward(self, inputs, targets):
        output = inputs.clamp(min=1e-5, max=1 - 1e-5)
        target = targets.float()
        weights = cal_weight(targets)

        if weights is not None:
            assert len(weights) == 2
            loss = -weights[0] * (target * torch.log(output)) - weights[1] * ((1 - target) * torch.log(1 - output))
        else:
            loss = -target * torch.log(output) - (1 - target) * torch.log(1 - output)
        loss = torch.sum(loss)

        return loss
