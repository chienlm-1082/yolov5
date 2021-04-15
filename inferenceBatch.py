import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
device = '0'
device =select_device(device)
import os

def load_model(weights_path, device):
    half = device.type != 'cpu'
    model = attempt_load(weights=weights_path, map_location=device)
    if half:
        model.half()
    stride = int(model.stride.max())
    return model, stride




def gen_batch(images: list, img_size, stride, device):
    def process(img):
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        half = device.type != 'cpu'
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0 
        return img
    
    imgs = [letterbox(x, img_size, stride=stride)[0] for x in images]
    imgs = list(map(process, imgs))
    imgs = torch.stack(imgs)
    return imgs


def predict(model, imgs, images, conf_thres=0.5, iou_thres=0.45):
    t1 = time_synchronized()
    pred = model(imgs, augment=True)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=0, agnostic=1)
    # print(pred)
    t2 = time_synchronized()
    print("Inference time: ", t2-t1)
    all_boxes = []
    for idx, det in enumerate(pred):
        print("===========================================================================")
        if len(det):
            det[:, :4] = scale_coords(imgs[idx].shape[1:], det[:, :4], images[idx].shape).round()
            print(det.cpu().numpy())
            all_boxes.append(det.cpu().numpy())
    return all_boxes

if __name__ == '__main__':
    weights_path = "pretrained_table_yolov5l.pt"
    model, stride = load_model(weights_path, device)
    path_images = 'Data_test/'
    list_images = os.listdir(path_images)
    list_images = sorted(list_images)
    print(list_images)
    images = []
    print(len(list_images))
    for path in list_images:
        img = cv2.imread(path_images+path)
        if img is not None:
            images.append(cv2.resize(img, (600, 900)))
    imgs = gen_batch(images, 640, stride, device)

    with torch.no_grad():
        all_boxes = predict(model, imgs, images)
    for idx, img_boxes in enumerate(all_boxes):
        img = cv2.imread(path_images+list_images[idx])
        img = cv2.resize(img, (600, 900))
        for box in img_boxes:
            if len(box):
                print(box)
                img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
        cv2.imwrite("Results_inference/"+list_images[idx], img)
        

