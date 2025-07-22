import jittor as jt
from jittor import nn


def my_binary_cross_entropy_with_logits(output, target, weight=None, pos_weight=None, reduction='mean'):
    '''
    Jittor自带的API不支持reduction =none,即输出一个和输入形状相同的张量
    手动实现的 BCEWithLogitsLoss，且 reduction='none'的功能，用于计算每个像素的原始损失
    以图片为单位返回每个元素的损失张量
'''
    max_val = jt.clamp(-output,min_v=0)
    if pos_weight is not None:
        log_weight = (pos_weight-1)*target + 1
        loss = (1-target)*output+(log_weight*(((-max_val).exp()+(-output - max_val).exp()).log()+max_val))
    else:
        loss = (1-target)*output+max_val+((-max_val).exp()+(-output -max_val).exp()).log()
    if weight is not None:
        loss *= weight

    # 根据reduction参数进行规约
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss




class IOU(nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = jt.sigmoid(pred)
        inter = (pred * target).sum(dims=(2,3))
        union = (pred + target).sum(dims=(2,3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def execute(self, pred, target):
        return self._iou(pred, target)



class structure_loss(nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()

    def _structure_loss(slef, pred, mask):
        weit = 1 + 5 * jt.abs(nn.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        # 内部使用了sigmoid
        wbce = my_binary_cross_entropy_with_logits(pred, mask, weight=weit, reduction='none')
        wbce = wbce.sum(dims=(2,3)) / weit.sum(dims=(2,3))
        pred = jt.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dims=(2, 3))
        union = ((pred + mask) * weit).sum(dims=(2, 3))
        wiou = 1 - (inter) / (union - inter)

        return (wbce + wiou).mean()

    def execute(self, pred, mask):
        return self._structure_loss(pred, mask)
