'''
@File    :   tracking.py
@Version :   1.0
@Author  :   laugh12321
@Contact :   laugh12321@vip.qq.com
@Date    :   2022/09/15 16:09:53
@Desc    :   None
'''
import numpy as np
import torch
from haversine import haversine

from .configs.parser import get_config
from .deep.extractor import Extractor
from .sort.detection import Detection
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.tracker import Tracker


def build_tracker(cfg_pth='./configs/config.yaml', use_cuda=True):

    cfg = get_config(cfg_pth)
    return StrongSORT(
        pretrain_path=cfg.REID.PRETRAIN_PATH,
        num_classes=cfg.REID.NUM_CLASS,
        last_stride=cfg.REID.LAST_STRIDE,
        neck=cfg.REID.NECK,
        neck_feat=cfg.REID.NECK_FEAT,
        model_name=cfg.REID.NAME,
        pretrain_choice=cfg.REID.PRETRAIN,
        max_dist=cfg.STRONGSORT.MAX_DIST,
        max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
        max_age=cfg.STRONGSORT.MAX_AGE,
        n_init=cfg.STRONGSORT.N_INIT,
        nn_budget=cfg.STRONGSORT.NN_BUDGET,
        mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
        ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
        use_cuda=use_cuda,
        update_rate=cfg.VELO.UPDATE_RATE,
        p_matrix=cfg.VELO.P_MATRIX
    )


class StrongSORT(object):

    def __init__(
        self,
        pretrain_path,
        p_matrix,  # 投影矩阵
        use_cuda=True,
        num_classes=751,
        last_stride=1,
        neck='bnneck',
        neck_feat='after',
        model_name='resnet34_ibn_a',
        pretrain_choice=False,
        max_dist=0.2,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
        update_rate=4,  # 更新频率
    ):

        self.extractor = Extractor(
            num_classes=num_classes,
            last_stride=last_stride,
            neck=neck,
            neck_feat=neck_feat,
            model_name=model_name,
            pretrain_choice=pretrain_choice,
            pretrain_path=pretrain_path,
            use_cuda=use_cuda
        )  # 提取特征网络
        metric = NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)  # 度量
        self.tracker = Tracker(
            metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            ema_alpha=ema_alpha,
            mc_lambda=mc_lambda
        )  # 初始化轨迹
        self.update_rate = update_rate  # 更新频率
        self.p_matrix = p_matrix  # 单应性矩阵
        self.ema_alpha = ema_alpha
        self.frame_count = 0  # 帧数
        self.tracks_active = {}  # 轨迹日志

    def update(self, bbox_xyxy, confidences, classes, ori_img, frame_rate):
        self.frame_count += 1
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xyxy, ori_img)
        bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)
        detections = [
            Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences)
        ]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            bbox_xyxy = self._tlwh_to_xyxy(box)

            track_id = track.track_id
            class_id = track.class_id
            newestRealLoc = self.homography2Dto3D(self.getLocation(bbox_xyxy), self.p_matrix)

            if track_id in self.tracks_active:
                self.tracks_active[track_id]["frame"] = self.frame_count
                self.tracks_active[track_id]["track_id"] = track_id
                self.tracks_active[track_id]["class_id"] = class_id
                self.tracks_active[track_id]["bbox"] = bbox_xyxy
                self.tracks_active[track_id]["newestRealLoc"] = newestRealLoc
            else:
                self.tracks_active[track_id] = {
                    "track_id": track_id,
                    "class_id": class_id,
                    "bbox": bbox_xyxy,
                    "newestRealLoc": newestRealLoc,
                    "prevRealLoc": newestRealLoc,
                    "frame": self.frame_count,
                    "prevFrame": self.frame_count,
                }
            if self.frame_count % self.update_rate == 0 and self.tracks_active[track_id][
                "frame"] != self.tracks_active[track_id]["prevFrame"]:
                frame_count = self.tracks_active[track_id]["frame"] - self.tracks_active[track_id][
                    "prevFrame"]
                componentDist = haversine(
                    self.tracks_active[track_id]['newestRealLoc'],
                    self.tracks_active[track_id]['prevRealLoc']
                )
                speed = (componentDist / (frame_count / frame_rate)) * 3600
                self.tracks_active[track_id]["speed"] = 0 if speed < 7 else speed
                self.tracks_active[track_id]['prevRealLoc'] = self.tracks_active[track_id][
                    'newestRealLoc']
                self.tracks_active[track_id]['EMAspeed'] = self.emaVelocity(
                    self.tracks_active[track_id]
                )
                self.tracks_active[track_id]['prevFrame'] = self.tracks_active[track_id]['frame']
            outputs.append(self.tracks_active[track_id])
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    @staticmethod
    def _xyxy_to_tlwh(bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_tlwh = bbox_xyxy.clone()

        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]  # width
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]  # height
        return bbox_tlwh

    @staticmethod
    def getLocation(bbox: list):
        """_summary_

        Args:
            bbox (list): _description_

        Returns:
            _type_: _description_
        """
        return (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2

    @staticmethod
    def homography2Dto3D(o2dpt, homoMat):
        """_summary_

        Args:
            o2dpt (_type_): _description_
            homoMat (_type_): _description_

        Returns:
            _type_: _description_
        """
        homogen2dPt = [*o2dpt, 1]
        matInv = np.linalg.inv(homoMat)
        hh = np.dot(matInv, homogen2dPt)
        scalar = hh[2]
        return [hh[0] / scalar, hh[1] / scalar]

    def emaVelocity(self, track):
        """_summary_

        Args:
            track (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.ema_alpha * track['speed'] + (1 - self.ema_alpha) * track[
            'EMAspeed'] if "EMAspeed" in track else track['speed']

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return [x1, y1, x2, y2]

    def increment_ages(self):
        self.tracker.increment_ages()

    def _get_features(self, bbox_xyxy, ori_img):
        im_crops = []
        for box in bbox_xyxy:
            x1, y1, x2, y2 = list(map(int, box))
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        return self.extractor(im_crops) if im_crops else np.array([])
