from unicodedata import category
from scipy.optimize import linear_sum_assignment
from torch import device, nn
from torch.cuda.amp import autocast
from .util import cos_similar

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# focal loss

def IoU_matrix(pred, target):
    row = pred.shape[0]
    col = target.shape[0]

    matrix = torch.zeros((row, col))

    for i in range(row):
        for j in range(col):

            matrix[i][j] = dice_loss(pred[i].unsqueeze(0), target[j].unsqueeze(0))

    return matrix


def mse_loss(image_layer, text, gt_mask:torch.Tensor):

    text_len = gt_mask.shape[0]
    gt_mask = nn.AdaptiveMaxPool2d((image_layer.shape[1], image_layer.shape[2]))(gt_mask)

    gt_mask = gt_mask.view(text_len, -1)
    image_layer = image_layer.view(image_layer.shape[0], -1).permute(1, 0)

    text = text[1: text_len+1, :]

    matrix = 1 - cos_similar(text, image_layer)

    loss = (matrix * gt_mask).sum(dim=-1)

    for n in range(text_len):
        loss[n] = loss[n] / (torch.nonzero(gt_mask[n]).shape[0] + 0.00001)

    return loss

def focal_loss(pt, target, alpha, gamma):

    pt = pt.view(pt.shape[0], -1)
    target = target.view(target.shape[0], -1)
    loss = - alpha * (1 - pt) ** gamma * target * torch.log(pt + 0.000001) - (1 - alpha) * pt ** gamma * (1 - target) * torch.log(1 - pt + 0.000001)

    return loss

# dice loss

def dice_loss(input, target ,mask=None ,eps=0.001):
    N, H, W = input.shape

    input = input.contiguous().view(N, H * W)
    target = target.contiguous().view(N, H * W).float()
    if mask is not None:
        mask = mask.contiguous().view(N, H * W).float()
        input = input * mask
        target = target * mask
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + eps
    c = torch.sum(target * target, 1) + eps
    d = (2 * a) / (b + c)
    # print('1-d max',(1-d).max())
    return 1 - d

# entrophy loss

def entropy_loss(input, target) -> torch.Tensor:
    N, H, W = input.shape

    input = input.view(N, -1)
    target = target.view(N, -1)
    loss = nn.BCELoss()

    return loss(input, target)

# loc loss

def loc_loss(input, target):

    loss = torch.sqrt((input[:, 0] - target[:, 0]) ** 2 + (input[:, 1] - target[:, 1]) ** 2) + torch.sqrt((input[:, 2] - target[:, 2]) ** 2)
    return loss

def multi_label_loss(input, target, input_index, target_index, device):
    len = input.shape[0] * target.shape[0]

    matrix = cos_similar(input, target)
    matrix = torch.softmax(matrix, dim=0)
    category = torch.zeros(matrix.shape).to(device)
    category[input_index, target_index] = 1

    matrix = matrix.view(len)
    category = category.view(len)
    loss_f = torch.nn.BCELoss()

    return loss_f(matrix, category)

# nllloss

def ce_loss(input, target):
    len = input.shape[0]
    
    input = input.view(len, -1).permute(1, 0)
    target = target.view(len-1, -1)

    loss_f = torch.nn.CrossEntropyLoss()

    category = torch.zeros((target.shape[1])).long().cuda()

    for pos in range(len-1):

        category[target[pos] == 1] = pos + 1

    return loss_f(input, category)
        

# constrative loss

def constrative_loss(image_layer, text, gt_mask:torch.Tensor):

    text_len = gt_mask.shape[0]

    
    gt_mask = nn.AdaptiveAvgPool2d((image_layer.shape[1], image_layer.shape[2]))(gt_mask)
    gt_mask[gt_mask>0] = 1
    gt_mask[gt_mask<0] = 0
    

    gt_mask = gt_mask.view(text_len, -1) # l, h*w
    image_layer = image_layer.view(image_layer.shape[0], -1)  # c, h*w

    text = text[1: text_len+1, :] # l, c

    matrix = torch.matmul(text, image_layer) # l, h*w
    matrix = torch.exp(matrix * 0.1)

    positive = matrix * gt_mask

    loss_1 = -torch.log((positive.sum(dim=-1) + 0.000001) / (matrix.sum(dim=-1)+ 0.000001)) # l
    loss_1 = loss_1.mean()

    loss_2 = -torch.log((positive.sum(dim=0) + 0.000001) / (matrix.sum(dim=0)+ 0.000001)) # h*w
    loss_2 = loss_2.mean()
    return loss_1

# word loss

def word_loss(input, target):

    loss = torch.nn.MSELoss()
    # Compute the classification cost. Contrary to the loss, we don't use the NLL,
    # but approximate it in 1 - proba[target class].
    # The 1 is a constant that doesn't change the matching, it can be ommitted.
    cost_word = loss(input, target)
    return cost_word

# matcher

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_dice: float = 1, cost_word=1, num_points: int = 12544):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_dice = cost_dice
        self.cost_word = cost_word
        self.num_points = num_points



    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        num_queries = outputs.shape[0]

        if self.cost_dice != 0:
        # Iterate through batch size

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            out_mask = outputs  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]

            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                    self.cost_dice * cost_dice
            )
        else:

            cost_word = 1 - cos_similar(outputs, targets)
            C = (
                    self.cost_word * cost_word
            )

        C = C.reshape(num_queries, -1).cpu()



        return linear_sum_assignment(C)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_word),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)