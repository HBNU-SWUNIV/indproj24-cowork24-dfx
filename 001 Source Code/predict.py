#!/usr/bin/env python

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import os
import pyrealsense2 as rs
import _init_paths
import models
import rospy
from std_msgs.msg import String

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
color_map = [(0, 0, 0), (188, 113, 0), (24, 82, 216)]


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    
    parser.add_argument('--model-type', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-s', type=str)
    parser.add_argument('--n-class', help='Number of categories', type=int, default=3)
    parser.add_argument('--model', help='dir for pretrained model', default='모델경로', type=str)
    parser.add_argument('--input', help='root or dir for input images', default='입력경로', type=str)

    args = parser.parse_args()
    return args


def input_transform(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image -= mean
    image /= std
    return image


def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cuda:0')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    return model


def draw_convexity_defects(image, roi, class_id, publisher):
    mask = np.zeros_like(roi, dtype=np.uint8)
    mask[roi == class_id] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c, returnPoints=False)
    defects = cv2.convexityDefects(c, hull)

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(c[s][0])
            end = tuple(c[e][0])
            far = tuple(c[f][0])
            cv2.line(image, start, end, [0, 255, 0], 2)
            cv2.circle(image, far, 5, [0, 0, 255], -1)

    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    publisher.publish(f"{cX},{cY}")

    cv2.circle(image, (cX, cY), 7, [255, 0, 255], -1)

    return image


def callback_request_next_target(msg):
    rospy.loginfo(f"Received request for next target: {msg.data}")

class RealSenseCapture:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()        
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        return aligned_frames

    def get_color_frame(self):
        aligned_frames = self.get_frames()
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def get_depth_frame(self):
        aligned_frames = self.get_frames()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame:
            return None
        return np.asanyarray(depth_frame.get_data())

    def release(self):
        self.pipeline.stop()


class InferenceEngine:
    def __init__(self, model_path, model_type, n_class):
        self.model = self.load_pretrained(model_type, model_path, n_class).cuda()
        self.model.eval()

    @staticmethod
    def load_pretrained(model_type, pretrained, n_class):
        model = models.pidnet.get_pred_model(model_type, n_class)
        pretrained_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        return model

    def infer(self, imgOrigin):
        img = input_transform(imgOrigin)
        img = img.transpose((2, 0, 1)).copy()
        img = torch.from_numpy(img).unsqueeze(0).cuda()
        pred = self.model(img)
        pred = F.interpolate(pred, size=imgOrigin.shape[:-1], mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        return pred


def main(args):
    rospy.init_node('segpredict_node')
    center_point_pub = rospy.Publisher('center_point', String, queue_size=10)
    rospy.Subscriber('request_next_target', String, callback_request_next_target)

    capture = RealSenseCapture()
    engine = InferenceEngine(args.model, args.model_type, args.n_class)

    try:
        while not rospy.is_shutdown():
            frame_color = capture.get_color_frame()
            frame_depth = capture.get_depth_frame()

            if frame_color is None or frame_depth is None:
                break

            height = frame_color.shape[0]
            roi = frame_color[2*height//3:]

            pred = engine.infer(frame_color)

            save_img = np.zeros_like(frame_color).astype(np.uint8)
            for i in np.unique(pred):
                for j in range(3):
                    save_img[:, :, j][pred == i] = color_map[i][j]

            target_class_id = color_map.index((188, 113, 0))

            draw_convexity_defects(save_img, pred[2*height//3:], target_class_id, center_point_pub)

            cv2.imshow("Result", save_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    main(args)
