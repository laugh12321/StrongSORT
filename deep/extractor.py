import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from .baseline import Baseline


class Extractor(object):

    def __init__(
        self,
        num_classes,
        last_stride,
        neck,
        neck_feat,
        model_name,
        pretrain_choice,
        pretrain_path,
        use_cuda=True
    ):
        self.net = Baseline(
            num_classes, last_stride, None, neck, neck_feat, model_name, pretrain_choice
        )
        self.net.load_param(pretrain_path)
        self.device = "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        self.net.eval()
        self.net.to(self.device)
        self.size = (128, 256)
        self.norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _preprocess(self, im_crops):
        """
        预处理
        TODO:
            1. to float with scale from 0 to 1 归一化
            2. resize to (128, 256) as Market1501 dataset did  转 128,256
            3. concatenate to a numpy array => numpy
            3. to torch Tensor  => Tensor
            4. normalize  标准化
        """

        def _resize(im, size):
            return cv2.resize(
                im.astype(np.float32) / 255., size
            )  # -> numpy(float) -> 0~1 -> 128,256

        # im_crops: List[cv.Mat], 多出一个维度B, 再cat组成批次 [B,C,H,W] : torch.Tensor-float
        im_batch = torch.cat(
            [self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0
        ).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)  # 输入送入GPU
            features = self.net(im_batch)  # 获取特征集合
        return features.cpu().numpy()  # 从GPU->CPU->numpy数组
