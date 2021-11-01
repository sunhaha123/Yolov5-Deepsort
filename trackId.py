# from deep_sort.utils.parser import get_config
# from deep_sort.deep_sort import DeepSort
import torch
import cv2
import uuid
import cv2
import time
from flask import Flask, render_template, Response
import pandas as pd
import numpy  as np
import threading


from diff_lane import beyond
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
#*******************#
# deepsort不再此处初始
#*******************#
# cfg = get_config()
# cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
# deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
#                     max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
#                     nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
#                     max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
#                     use_cuda=True)


def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    # tl = line_thickness or round(
    #     0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    tf,tl = 1,1
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if cls_id in ['person']:
            color = (0, 255,0 )
        else:
            color = (0, 255, 0)
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        # tf = max(tl - 1, 1)  # font thickness

        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{}'.format(pos_id), (c1[0], c1[1] - 2), 0, 0.5,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image


def update_tracker(deepsort,bboxes, image):
    t1 = time.time()
    bbox_xywh = []
    confs = []
    clss = []

    for x1, y1, x2, y2, cls_id, conf in bboxes:

        obj = [
            int((x1+x2)/2), int((y1+y2)/2),
            x2-x1, y2-y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)
    #追踪
    outputs = deepsort.update(xywhs, confss, clss, image)
    t3 = time.time()
    # print('track wasted time %5f' % (t3 - t1))

    bboxes2draw = []
    face_bboxes = []
    current_ids = []
    for value in list(outputs):
        x1, y1, x2, y2, cls_, track_id = value
        central_x = (x1+x2)/2
        central_y = (y1+y2)/2
        # 判断是否出界
        if beyond(central_x, central_y) == 0:
            continue
        bboxes2draw.append(
            (x1, y1, x2, y2, cls_, track_id)
        )
    # 画框+编号
    image = plot_bboxes(image, bboxes2draw)

    #获取画面长和宽
    width = image.shape[1]
    height = image.shape[0]

    #生成magic_box
    magic_box = []
    for track in bboxes2draw:
        id = track[5]
        width = width
        height = height
        tl = (track[0],track[1])
        br = (track[2],track[3])
        central_x = (tl[0] + br[0]) / 2
        central_y = (tl[1] + br[1]) / 2

        w = track[2] - track[0]
        h = track[3] - track[1]
        # 求 lane
        lane = 1
        section = 1
        # uuid
        uuid_t = ''.join(str(uuid.uuid4()).split('-'))
        status = 1  # status 默认为1 为正常状态
        timedelta = int(time.time() * 1000)


        # 打印出第几帧的所有id和框
        # print(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
        #         f'{w:.6f},{h:.6f},-1,-1,-1\n')
        # log.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
        #        f'{w:.6f},{h:.6f},-1,-1,-1\n')
        magic_box.append([uuid_t, int(id), timedelta, status, lane, section,
                          central_x / width, central_y / height,
                          w / width, h / height])
    return image, magic_box
