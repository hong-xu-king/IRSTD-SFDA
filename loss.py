import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import cv2
class SC_BCEloss(nn.Module):
    def __init__(self):
        super(SC_BCEloss, self).__init__()
        self.cal_loss1 = nn.BCELoss(size_average=True)
    def forward(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0

            for i in range(len(preds)):
                pred = preds[i]
                # pred = process_tensor(pred)
                gt_mask = gt_masks[i]
                loss = self.cal_loss1(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)

        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                # pred = process_tensor(pred)
                loss = self.cal_loss1(pred, gt_masks)
                a.append(loss)

            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            # preds = process_tensor(preds)
            loss = self.cal_loss1(preds, gt_masks)
            return loss

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]

                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss

class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()
        
    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())
        
        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)
        
        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt)+ self.softiou(preds[1].sigmoid(), edge_gt)
        
        return loss_img + loss_edge
def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou   = 1 - (inter*2 + 1) / (union + 1)
    return iou.mean()


def miou(pred, mask):
    mini  = 1
    maxi  = 1
    nbins = 1
    predict = (pred > 0).float()
    intersection = predict * ((predict == mask).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _   = np.histogram(mask.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    iou = 1 * area_inter / (np.spacing(1) + area_union)

    return iou.mean()


def entropy_minimization_loss(predictions):
    p = F.softmax(predictions, dim=1)
    return -torch.mean(torch.sum(p * torch.log(p + 1e-10), dim=1))


def histogram_matching_loss(tensor1, tensor2, bins=10):
    hist1 = torch.histc(tensor1, bins=bins, min=0.0, max=1.0)
    hist2 = torch.histc(tensor2, bins=bins, min=0.0, max=1.0)

    # 归一化直方图
    hist1 = hist1 / torch.sum(hist1)
    hist2 = hist2 / torch.sum(hist2)

    # 使用MSE来度量两个直方图的差异
    loss = torch.mean((hist1 - hist2) ** 2)

    return loss


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf'):
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val) / len(kernel_val)

    def forward(self, source, target):

        if self.kernel_type == 'rbf':
            kernels = self.gaussian_kernel(source, target)
            XX = kernels[:source.size(0), :source.size(0)]
            YY = kernels[source.size(0):, source.size(0):]
            XY = kernels[:source.size(0), source.size(0):]
            YX = kernels[source.size(0):, :source.size(0)]
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class CORALLoss(nn.Module):
    def forward(self, source, target):
        d = source.size(1)
        source_covar = (1 / (source.size(0) - 1)) * (source.t() @ source)
        target_covar = (1 / (target.size(0) - 1)) * (target.t() @ target)
        loss = torch.sum((source_covar - target_covar) ** 2) / (4 * d * d)
        return loss

def extract_region(image, center, size):
    """
    从图像中提取一个区域。
    :param image: 输入图像
    :param center: 区域中心坐标 (y, x)
    :param size: 区域的宽高 (height, width)
    :return: 截取的区域图像
    """
    y, x = center
    h, w = size
    start_x = max(x - w // 2, 0)
    start_y = max(y - h // 2, 0)
    end_x = min(x + w // 2, image.shape[1])
    end_y = min(y + h // 2, image.shape[0])
    new_w = end_x-start_x
    new_h = end_y-start_y

    if new_w % 2 != 0:
        if new_w<size[0] and start_x==0:
            end_x = end_x - 1
        if new_w < size[0] and end_x == image.shape[1]:
            start_x = start_x+1
    if new_h % 2 != 0:
        if new_h<size[1] and start_y==0:
            end_y = end_y - 1
        if new_h < size[1] and end_y == image.shape[0]:
            start_y = start_y + 1

    return image[start_y:end_y ,start_x:end_x],start_x,start_y,end_x,end_y


class UIUNetLoss(nn.Module):
    def __init__(self):
        super(UIUNetLoss, self).__init__()
        self.bce_loss = nn.BCELoss(size_average=True)

    def forward(self, preds, labels_v):
        loss0 = self.bce_loss(preds[0], labels_v)
        loss1 = self.bce_loss(preds[1], labels_v)
        loss2 = self.bce_loss(preds[2], labels_v)
        loss3 = self.bce_loss(preds[3], labels_v)
        loss4 = self.bce_loss(preds[4], labels_v)
        loss5 = self.bce_loss(preds[5], labels_v)
        loss6 = self.bce_loss(preds[6], labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        # loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
        # loss5.data.item(), loss6.data.item()))

        return loss
def SoftlossFUN(pred,gt_masks):
    smooth = 1
    intersection = pred * gt_masks
    loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() - intersection.sum() + smooth)
    return loss
def custom_cos_function(x):
    """
    使用余弦函数构造的自定义函数，使得：
    - 当 x = 0 时，返回 1
    - 当 x = 1 时，返回 0
    """
    return (1 + math.cos(math.pi * x)) / 2
class RustIoULoss(nn.Module):
    def __init__(self):
        super(RustIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        size=(40,40)
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred_LOSS_SUM = 0
                pred = preds[i]
                for j in range(gt_masks.size(0)):
                    region_LOSS_SUM =0
                    mask_np = np.squeeze(gt_masks[j].cpu().detach().numpy().astype(np.uint8))
                    pred_clone = pred[j, 0, :, :].clone()
                    mask_clone = gt_masks[j, 0, :, :].clone()
                    #pred_np = np.squeeze(pred[i].cpu().detach().numpy().astype(np.uint8))
                    #print(type(mask_np))
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

                    for k in range(1, num_labels):  # 从1开始，跳过背景
                        center = centroids[k]
                        center = (int(center[1]), int(center[0]))  # y x
                        region_pred, start_x, start_y, end_x, end_y = extract_region(pred[j,0,:,:], center, size)

                        #pred_clone[start_y:end_y, start_x:end_x] =torch.zeros_like(region_pred)

                        region_mask, start_x, start_y, end_x, end_y = extract_region(gt_masks[j,0,:,:], center, size)

                        #mask_clone[start_y:end_y, start_x:end_x] =torch.zeros_like(region_mask)
                        region_IOU = SoftlossFUN(region_pred,region_mask)
                        alpha = custom_cos_function(region_IOU)
                        region_LOSS = alpha*region_IOU
                        region_LOSS_SUM = region_LOSS +region_LOSS_SUM
                    if num_labels==1:
                        num_labels=2
                       # +region_LOSS_SUM/(num_labels-1)
                    pred_LOSS = SoftlossFUN(pred_clone, mask_clone)+region_LOSS_SUM/(num_labels-1)
                    pred_LOSS_SUM = pred_LOSS_SUM+pred_LOSS
                pred_LOSS_per = 1-pred_LOSS_SUM/gt_masks.size(0)
                loss_total = loss_total + pred_LOSS_per
            return loss_total / len(preds)
        else:
            pred_LOSS_SUM = 0
            pred = preds
            for i in range(gt_masks.size(0)):
                region_LOSS_SUM = 0
                mask_np = gt_masks[i].cpu().numpy().astype(np.uint8)
                pred_clone = pred[i, 0, :, :].clone()
                mask_clone = gt_masks[i, 0, :, :].clone()
                #pred_np = pred[i].cpu().numpy().astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
                print(num_labels)
                for k in range(1, num_labels):  # 从1开始，跳过背景
                    center = centroids[k]
                    center = (int(center[1]), int(center[0]))  # y x
                    region_pred, start_x, start_y, end_x, end_y = extract_region(pred[i, 0, :, :], center, size)

                    pred_clone[start_y:end_y, start_x:end_x] = torch.zeros_like(region_pred)

                    region_mask, start_x, start_y, end_x, end_y = extract_region(gt_masks[i, 0, :, :], center, size)

                    mask_clone[start_y:end_y, start_x:end_x] = torch.zeros_like(region_mask)
                    region_IOU = SoftlossFUN(region_pred, region_mask)
                    alpha = custom_cos_function(region_IOU)
                    region_LOSS = alpha * region_IOU
                    region_LOSS_SUM = region_LOSS + region_LOSS_SUM
                if num_labels == 1:
                    num_labels = 2
                pred_LOSS = (SoftlossFUN(pred_clone, mask_clone) + region_LOSS_SUM) / (num_labels-1)
                pred_LOSS_SUM = pred_LOSS_SUM + pred_LOSS
            pred_LOSS_per = 1 - pred_LOSS_SUM / gt_masks.size(0)
            return pred_LOSS_per
class RustIoULoss1(nn.Module):
    def __init__(self):
        super(RustIoULoss1, self).__init__()
    def forward(self, preds, gt_masks):
        size=(40,40)
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            loss_total_region = 0
            loss_total_global = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss_global = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss_global = 1 - loss_global.mean()
                loss_total_global = loss_total_global + loss_global

                pred_LOSS_SUM = 0
                for j in range(gt_masks.size(0)):
                    region_LOSS_SUM =0

                    mask_np = np.squeeze(gt_masks[j].cpu().detach().numpy().astype(np.uint8))

                    #pred_np = np.squeeze(pred[i].cpu().detach().numpy().astype(np.uint8))
                    #print(type(mask_np))
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

                    for k in range(1, num_labels):  # 从1开始，跳过背景
                        center = centroids[k]
                        center = (int(center[1]), int(center[0]))  # y x
                        region_pred, start_x, start_y, end_x, end_y = extract_region(pred[j,0,:,:], center, size)

                        #pred_clone[start_y:end_y, start_x:end_x] =torch.zeros_like(region_pred)

                        region_mask, start_x, start_y, end_x, end_y = extract_region(gt_masks[j,0,:,:], center, size)

                        #mask_clone[start_y:end_y, start_x:end_x] =torch.zeros_like(region_mask)
                        region_IOU = SoftlossFUN(region_pred,region_mask)
                        #region_GCE_LOSS = self.Gce(region_pred.squeeze(0).squeeze(0),region_mask.squeeze(0).squeeze(0))
                        #alpha = custom_cos_function(region_IOU)
                        #pred_clone[start_y:end_y, start_x:end_x] =pred_clone[start_y:end_y, start_x:end_x]*alpha
                        #mask_clone[start_y:end_y, start_x:end_x] =mask_clone[start_y:end_y, start_x:end_x]*alpha
                        region_LOSS = region_IOU
                        #print(region_GCE_LOSS)
                        region_LOSS_SUM = region_LOSS + region_LOSS_SUM
                    if num_labels==1:
                        num_labels=2
                       # +region_LOSS_SUM/(num_labels-1)
                    pred_LOSS = region_LOSS_SUM/(num_labels-1)
                    pred_LOSS_SUM = pred_LOSS_SUM+pred_LOSS
                pred_LOSS_per = 1-pred_LOSS_SUM/gt_masks.size(0)
                loss_total_region = loss_total_region + pred_LOSS_per
            return loss_total_global / len(preds)+loss_total_region/len(preds)
        else:
            loss_total = 0
            loss_total_region = 0
            loss_total_global = 0

            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss_global = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss_global = 1 - loss_global.mean()
            loss_total_global = loss_total_global + loss_global

            pred_LOSS_SUM = 0
            for j in range(gt_masks.size(0)):
                region_LOSS_SUM =0

                mask_np = np.squeeze(gt_masks[j].cpu().detach().numpy().astype(np.uint8))

                #pred_np = np.squeeze(pred[i].cpu().detach().numpy().astype(np.uint8))
                #print(type(mask_np))
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

                for k in range(1, num_labels):  # 从1开始，跳过背景
                    center = centroids[k]
                    center = (int(center[1]), int(center[0]))  # y x
                    region_pred, start_x, start_y, end_x, end_y = extract_region(pred[j,0,:,:], center, size)

                    #pred_clone[start_y:end_y, start_x:end_x] =torch.zeros_like(region_pred)

                    region_mask, start_x, start_y, end_x, end_y = extract_region(gt_masks[j,0,:,:], center, size)

                    #mask_clone[start_y:end_y, start_x:end_x] =torch.zeros_like(region_mask)
                    region_IOU = SoftlossFUN(region_pred,region_mask)
                    #region_GCE_LOSS = self.Gce(region_pred.squeeze(0).squeeze(0),region_mask.squeeze(0).squeeze(0))
                    #alpha = custom_cos_function(region_IOU)
                    #pred_clone[start_y:end_y, start_x:end_x] =pred_clone[start_y:end_y, start_x:end_x]*alpha
                    #mask_clone[start_y:end_y, start_x:end_x] =mask_clone[start_y:end_y, start_x:end_x]*alpha
                    region_LOSS = region_IOU
                    #print(region_GCE_LOSS)
                    region_LOSS_SUM = region_LOSS + region_LOSS_SUM
                if num_labels==1:
                    num_labels=2
                   # +region_LOSS_SUM/(num_labels-1)
                pred_LOSS = region_LOSS_SUM/(num_labels-1)
                pred_LOSS_SUM = pred_LOSS_SUM+pred_LOSS
            pred_LOSS_per = 1-pred_LOSS_SUM/gt_masks.size(0)
            loss_total_region = loss_total_region + pred_LOSS_per
            return loss_total_global +loss_total_region

class RustIoULoss2(nn.Module):
    def __init__(self):
        super(RustIoULoss2, self).__init__()
    def forward(self, preds, gt_masks):
        size=(40,40)
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            loss_total_region = 0
            loss_total_global = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss_global = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss_global = 1 - loss_global.mean()
                loss_total_global = loss_total_global + loss_global

                pred_LOSS_SUM = 0
                for j in range(gt_masks.size(0)):
                    region_LOSS_SUM =0

                    mask_np = np.squeeze(gt_masks[j].cpu().detach().numpy().astype(np.uint8))

                    #pred_np = np.squeeze(pred[i].cpu().detach().numpy().astype(np.uint8))
                    #print(type(mask_np))
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

                    for k in range(1, num_labels):  # 从1开始，跳过背景
                        center = centroids[k]
                        center = (int(center[1]), int(center[0]))  # y x
                        region_pred, start_x, start_y, end_x, end_y = extract_region(pred[j,0,:,:], center, size)

                        #pred_clone[start_y:end_y, start_x:end_x] =torch.zeros_like(region_pred)

                        region_mask, start_x, start_y, end_x, end_y = extract_region(gt_masks[j,0,:,:], center, size)

                        #mask_clone[start_y:end_y, start_x:end_x] =torch.zeros_like(region_mask)
                        region_IOU = SoftlossFUN(region_pred,region_mask)
                        #region_GCE_LOSS = self.Gce(region_pred.squeeze(0).squeeze(0),region_mask.squeeze(0).squeeze(0))
                        alpha = custom_cos_function(region_IOU)
                        #pred_clone[start_y:end_y, start_x:end_x] =pred_clone[start_y:end_y, start_x:end_x]*alpha
                        #mask_clone[start_y:end_y, start_x:end_x] =mask_clone[start_y:end_y, start_x:end_x]*alpha
                        region_LOSS = alpha*region_IOU
                        #print(region_GCE_LOSS)
                        region_LOSS_SUM = region_LOSS + region_LOSS_SUM
                    if num_labels==1:
                        num_labels=2
                       # +region_LOSS_SUM/(num_labels-1)
                    pred_LOSS = region_LOSS_SUM/(num_labels-1)
                    pred_LOSS_SUM = pred_LOSS_SUM+pred_LOSS
                pred_LOSS_per = 1-pred_LOSS_SUM/gt_masks.size(0)
                loss_total_region = loss_total_region + pred_LOSS_per
            return loss_total_global / len(preds)+loss_total_region/len(preds)
        else:
            loss_total = 0
            loss_total_region = 0
            loss_total_global = 0

            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss_global = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss_global = 1 - loss_global.mean()
            loss_total_global = loss_total_global + loss_global

            pred_LOSS_SUM = 0
            for j in range(gt_masks.size(0)):
                region_LOSS_SUM =0

                mask_np = np.squeeze(gt_masks[j].cpu().detach().numpy().astype(np.uint8))

                #pred_np = np.squeeze(pred[i].cpu().detach().numpy().astype(np.uint8))
                #print(type(mask_np))
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

                for k in range(1, num_labels):  # 从1开始，跳过背景
                    center = centroids[k]
                    center = (int(center[1]), int(center[0]))  # y x
                    region_pred, start_x, start_y, end_x, end_y = extract_region(pred[j,0,:,:], center, size)

                    #pred_clone[start_y:end_y, start_x:end_x] =torch.zeros_like(region_pred)

                    region_mask, start_x, start_y, end_x, end_y = extract_region(gt_masks[j,0,:,:], center, size)

                    #mask_clone[start_y:end_y, start_x:end_x] =torch.zeros_like(region_mask)
                    region_IOU = SoftlossFUN(region_pred,region_mask)
                    #region_GCE_LOSS = self.Gce(region_pred.squeeze(0).squeeze(0),region_mask.squeeze(0).squeeze(0))
                    alpha = custom_cos_function(region_IOU)
                    #pred_clone[start_y:end_y, start_x:end_x] =pred_clone[start_y:end_y, start_x:end_x]*alpha
                    #mask_clone[start_y:end_y, start_x:end_x] =mask_clone[start_y:end_y, start_x:end_x]*alpha
                    region_LOSS = alpha*region_IOU
                    #print(region_GCE_LOSS)
                    region_LOSS_SUM = region_LOSS + region_LOSS_SUM
                if num_labels==1:
                    num_labels=2
                   # +region_LOSS_SUM/(num_labels-1)
                pred_LOSS = region_LOSS_SUM/(num_labels-1)
                pred_LOSS_SUM = pred_LOSS_SUM+pred_LOSS
            pred_LOSS_per = 1-pred_LOSS_SUM/gt_masks.size(0)
            loss_total_region = loss_total_region + pred_LOSS_per
            return loss_total_global +loss_total_region