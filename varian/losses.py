import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


eps = 1e-3


def dice_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return dice_loss(preds, trues, is_average=is_average)


def multi_class_dice_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return multi_class_dice(preds, trues, is_average=is_average)


def dice_loss(preds, trues, weight=None, is_average=True):
    preds = preds.contiguous()
    trues = trues.contiguous()
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    if is_average:
        score = scores.sum()/num
        return torch.clamp(score, 0., 1.)
    else:
        return scores


def per_class_dice(preds, trues, weight=None, is_average=True):
    loss = []
    for idx in range(1, preds.shape[1]):
        loss.append(dice_loss(preds[:,idx,...].contiguous(), (trues==idx).float().contiguous(), weight, is_average))
    return loss


def multi_class_dice(preds, trues, weight=None, is_average=True):
    channels = per_class_dice(preds, trues, weight, is_average)
    return sum(channels) / len(channels)


def jaccard_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return jaccard(preds, trues, is_average=is_average)


def jaccard(preds, trues, weight=None, is_average=True):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (intersection + eps) / ((preds + trues).sum(1) - intersection + eps)

    score = scores.sum()
    if is_average:
        score /= num
    return torch.clamp(score, 0., 1.)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return dice_loss(input, target, self.weight, self.size_average)


class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return jaccard(input, target, self.weight, self.size_average)


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight if pos_weight is not None else 0.5
        
    def forward(self, logits, target):
        pos_weights = target
        pos_weights[target == 1.] = self.pos_weight
        pos_weights[target == 0.] = 1 - self.pos_weight
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weights)

    
class LossBinary(nn.Module):
    """
    Loss defined as BCE - log(soft_jaccard)
    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection
    using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0, pos_weight=None):
        super().__init__()
        self.nll_loss = WeightedBCEWithLogitsLoss(pos_weight)
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log(
                (intersection + eps) / (union - intersection + eps))
        return loss
    

class BCEDiceJaccardLoss(nn.Module):
    def __init__(self, weights, pos_weight=None, size_average=True):
        super().__init__()
        self.weights = weights
        self.pos_weight = pos_weight
        self.bce = WeightedBCEWithLogitsLoss(pos_weight)
        self.jacc = JaccardLoss()
        self.dice = DiceLoss()
        self.mapping = {'bce': self.bce,
                        'jacc': self.jacc,
                        'dice': self.dice}
        self.values = {}

    def forward(self, input, target):
        loss = 0
        sigmoid_input = torch.sigmoid(input)
        for k, v in self.weights.items():
            if not v:
                continue

            val = self.mapping[k](
                input if k == 'bce' else sigmoid_input, 
                target
            )
            if k == 'bce':
                loss += self.weights[k] * val
            else:
                loss += self.weights[k] * (1 - val)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
    

class WeightedBCEFocalLoss(nn.Module):
    def __init__(self, weights, alpha=1, gamma=2):
        super(WeightedBCEFocalLoss, self).__init__()
        self.weights = weights
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, input, target):
        input = F.sigmoid(input)
        BCE_loss = weighted_binary_cross_entropy(input, target, self.weights)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return torch.mean(F_loss)
