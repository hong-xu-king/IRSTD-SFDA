from utils import *
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, model_name, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        self.model_name = model_name
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
        self.tranform_nyh = augumentation_nyh()

    def __getitem__(self, idx):
        # try:
        img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
            'I')
        mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        # if self.model_name == "NyhNet":
        #     mask = cv2.imread((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png'), 0)
        #     body = cv2.blur(mask, ksize=(5, 5))
        #     body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        #     body = body ** 0.5
        #     detail = mask - body
        # except:
        #     img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//','/')).convert('I')
        #     mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//','/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        if self.model_name == "NyhNet":
            img_patch, mask_patch, body_patch, detail_patch = random_crop_nyh(img, mask, body, detail, self.patch_size,
                                                                              pos_prob=0.5)
            img_patch, mask_patch, body_patch, detail_patch = self.tranform_nyh(img_patch, mask_patch, body_patch,
                                                                                detail_patch)
            img_patch, mask_patch, body_patch, detail_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis,
                                                                                        :], body_patch[np.newaxis,
                                                                                            :], detail_patch[np.newaxis,
                                                                                                :]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
            body_patch = torch.from_numpy(np.ascontiguousarray(body_patch))
            detail_patch = torch.from_numpy(np.ascontiguousarray(detail_patch))
            return img_patch, mask_patch, body_patch, detail_patch
        else:
            img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)
            img_patch, mask_patch = self.tranform(img_patch, mask_patch)
            img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
            return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)

class PlabelTrainSetLoader(Dataset):
    def __init__(self, original_imgs,pseudo_labels,dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.pseudo_labels = pseudo_labels
        self.original_imgs = original_imgs
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

    def __getitem__(self, idx):
        pseudo_label = self.pseudo_labels[idx]
        original_img = self.original_imgs[idx]
        img = Normalized(np.array(original_img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(pseudo_label, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch

    def __len__(self):
        return len(self.original_imgs)


class plabelSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/train_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//', '/')).convert(
                'I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//', '/')).convert(
                'I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//', '/'))
        original_img = np.array(img, dtype=np.float32)

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape

        img = PadImg(img)
        original_img = PadImg(original_img)
        mask = PadImg(mask)

        img,original_img, mask = img[np.newaxis, :],original_img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        original_img = torch.from_numpy(np.ascontiguousarray(original_img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask,original_img, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()

        self.dataset_dir = dataset_dir + '/' + test_dataset_name

        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        # try:
        img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//', '/')).convert(
            'I')
        mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] +'.png').replace('//', '/'))

        # except:
        #     img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//', '/')).convert(
        #         'I')
        #     mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//', '/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        mask_pred = Image.open((self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' +
                                self.test_list[idx] + '.png').replace('//', '/'))
        mask_gt = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] + '.png')

        mask_pred = np.array(mask_pred, dtype=np.float32) / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32) / 255.0

        if len(mask_pred.shape) == 3:
            mask_pred = mask_pred[:, :, 0]

        h, w = mask_pred.shape

        mask_pred, mask_gt = mask_pred[np.newaxis, :], mask_gt[np.newaxis, :]

        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h, w]

    def __len__(self):
        return len(self.test_list)


class augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target


class augumentation_nyh(object):
    def __call__(self, input, target, body, detail):
        if random.random() < 0.5:
            input = input[::-1, :]
            target = target[::-1, :]
            body = target[::-1, :]
            detail = target[::-1, :]
        if random.random() < 0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
            body = target[:, ::-1]
            detail = target[:, ::-1]
        if random.random() < 0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
            body = target.transpose(1, 0)
            detail = target.transpose(1, 0)
        return input, target, body, detail
