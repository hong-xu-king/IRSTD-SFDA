import argparse
import time
from utils import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
from proprecess import MDA
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_name", default='DNANet', type=str,
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet','SCTransNet")
parser.add_argument("--source_dataset_name", default='IRSTD-1K', type=str,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea', 'IRDST-real'")
parser.add_argument("--target_dataset_name", default='NUAA-SIRST', type=str,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea', 'IRDST-real'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--warmup", default=True, nargs='+', help="Whether to warm up on the source data")
parser.add_argument("--warmup_epoch", type=int, default=5, help="Training batch sizse")
parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
parser.add_argument("--warmup_save", default='./warmup_log', type=str, help="Save path of warmup checkpoints")
parser.add_argument("--resume", default=None, nargs='+', help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--pretrained", default=None, nargs='+', help="Load pretrained checkpoints (default: None)")
parser.add_argument("--nEpochs", type=int, default=1500, help="Number of epochs")
parser.add_argument("--optimizer_name", default='Adagrad', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.5}, type=dict, help="scheduler settings")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--intervals", type=int, default=10, help="Intervals for print loss")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument("--weight_path", default='./preweight/IRSTD-1K/DNANet/DNANet_1500.pth.tar', type=str, help="train_weight_path")

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
  opt.img_norm_cfg = dict()
  opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
  opt.img_norm_cfg['std'] = opt.img_norm_cfg_std

seed_pytorch(opt.seed)
# 定义熵最小化损失



def generate_pseudo_labels(model, dataname, threshold=0.5,save=False):
    print('---------------pseudo labels generating-------------')

    test_set = plabelSetLoader(opt.dataset_dir, dataname, dataname, img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    model.eval()
    pseudo_labels = []
    original_imgs = []
    with torch.no_grad():
        for images,gt_mask,original_img,size,path in test_loader:
            images = images.cuda()
            pred = model(images)
            pred = pred[:,:,:size[0],:size[1]]
            original_img = original_img[:,:,:size[0],:size[1]]

            # 置信度阈值法
            mask = pred > threshold
            pseudo_label = mask.int().cpu().numpy().astype(np.uint8) * 255  # 0 255
            original_img = original_img.cpu().numpy().astype(np.uint8)
            pseudo_label,original_img = np.squeeze(pseudo_label),np.squeeze(original_img)
            pseudo_label = MDA(original_img,pseudo_label,size=(40,40))
            original_imgs.append(original_img)
            pseudo_labels.append(pseudo_label)

    print('---------------pseudo labels finish-------------')
    return original_imgs,pseudo_labels


def update_TrainSetLoader(net,threshold=0.5):
    original_imgs,pseudo_labels=generate_pseudo_labels(net, opt.target_dataset_name, threshold=0.5)
    target_set = PlabelTrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.target_dataset_name, patch_size=opt.patchSize,
                               img_norm_cfg=opt.img_norm_cfg,original_imgs=original_imgs,pseudo_labels=pseudo_labels)
    target_loader = DataLoader(dataset=target_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    return target_loader

def train():

    
    src_net = Net(model_name=opt.model_name, mode='train').cuda()
    src_net.train()


    epoch_state_tar = 0
    total_loss_list_tar = []
    total_loss_epoch_tar = []


    ### Default settings of DNANet
    if opt.model_name == 'DNANet':
        opt.optimizer_name == 'Adagrad'
        opt.optimizer_settings = {'lr': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs':500, 'min_lr':1e-5}

    if opt.model_name == 'RDIAN':
        opt.optimizer_name == 'Adagrad'
        opt.optimizer_settings = {'lr': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': 500, 'min_lr': 1e-5}

    if opt.model_name == 'SCTransNet':
        opt.optimizer_name = 'Adam'
        opt.optimizer_settings = {'lr': 0.001}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': 500, 'min_lr': 1e-5, 'last_epoch': -1}
    if opt.model_name == 'UIUNet':
        opt.optimizer_name = 'Adam'
        opt.optimizer_settings = {'lr': 0.001}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': 500, 'min_lr': 1e-5, 'last_epoch': -1}
    opt.nEpochs = opt.scheduler_settings['epochs']



    save_pth = opt.weight_path
    print(save_pth)
    test(save_pth)
    #---------创建eval模型   更新伪标签
    eval_net = Net(model_name=opt.model_name, mode='test').cuda()
    eval_net.eval()
    eval_net.load_state_dict(torch.load(save_pth)['state_dict'])
    print(f'load {save_pth}')
    target_loader = update_TrainSetLoader(eval_net,threshold=0.5)

    # 将模型重新加载进训练模型
    tar_net = Net(model_name=opt.model_name, mode='train').cuda()
    tar_net.load_state_dict(torch.load(save_pth)['state_dict'])
    tar_net.train()
    tar_net = torch.nn.DataParallel(tar_net)
    # print('训练模型加载完成')


    optimizer, scheduler = get_optimizer(tar_net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                         opt.scheduler_settings)
    print('-----------------------target data train start -------------------------------')
    for idx_epoch in range(epoch_state_tar, opt.nEpochs):
        for idx_iter, (img, gt_mask) in enumerate(target_loader):
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
            if img.shape[0] == 1:
                continue
            pred = tar_net(img)

            loss = tar_net.module.loss(pred, gt_mask)
            total_loss_epoch_tar.append(loss.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        if (idx_epoch + 1) % 1 == 0:
            total_loss_list_tar.append(float(np.array(total_loss_epoch_tar).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss_tar---%f,current_lr---%f'
                  % (idx_epoch + 1, total_loss_list_tar[-1],current_lr))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss_tar---%f,current_lr---%f,\n'
                        % (idx_epoch + 1, total_loss_list_tar[-1],current_lr))
            total_loss_epoch_tar = []

        if (idx_epoch + 1) % 10 == 0 :
            save_pth = opt.save + '/' + opt.target_dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': tar_net.module.state_dict(),
                'total_loss': total_loss_list_tar,
            }, save_pth)
            test(save_pth)

        if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % 50 != 0:
            save_pth = opt.save + '/' + opt.target_dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': tar_net.module.state_dict(),
                'total_loss': total_loss_list_tar,
            }, save_pth)
            test(save_pth)

def test(save_pth):
    test_set = TestSetLoader(opt.dataset_dir, opt.target_dataset_name, opt.target_dataset_name, img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0)
    for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
        img = Variable(img).cuda()
        pred = net.forward(img)
        pred = pred[:,:,:size[0],:size[1]]
        gt_mask = gt_mask[:,:,:size[0],:size[1]]
        eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)
        nIoU_metric.update(pred, gt_mask)  # 像素
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    nIoU = nIoU_metric.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("nIOU:\t" + str(nIoU))
    print("PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("nIOU:\t" + str(nIoU) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')

def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path

if __name__ == '__main__':
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)
    opt.f = open(opt.save + '/' + opt.source_dataset_name  +'to'+opt.target_dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ',
                                                                                                            '_').replace(
        ':', '_') + '.txt', 'w')
    print(opt.source_dataset_name  +'_to_'+opt.target_dataset_name + '\t' + opt.model_name)
    train()
    print('\n')
    opt.f.close()
