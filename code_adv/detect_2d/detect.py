import os, sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(parent_dir))
import torch.nn.functional as F
import torch 
import numpy as np
from detect_utils import letterbox, non_max_suppression, scale_boxes
import cv2
import copy

class Detect():
    def __init__(self, checkpoint_file, device_idx):
        self.img_shape = None
        
        self.device_idx = device_idx
        self.input_shape = (640, 640)
        model = torch.jit.load(checkpoint_file)
        self.model = model.to(device_idx)

    def preprocess(self, input_data):
        """
        对输入图像进行预处理
        :param input_data: 输入图像
        :return: 处理后的图像张量
        """
        # 调整图像大小，使用letterbox方法填充
        self.origin_img = copy.deepcopy(input_data)
        self.img_shape = input_data.shape
        resized_img = letterbox(input_data, new_shape=self.input_shape, auto=False, stride=64)[0]

        # 调整图像通道顺序和维度顺序
        transposed_img = resized_img[:, :, ::-1].transpose(2, 0, 1)

        # 转换为张量并归一化
        tensor_img = torch.from_numpy(np.ascontiguousarray(transposed_img)).float() / 255.0

        # 扩展维度以匹配模型输入要求
        tensor_img = tensor_img.unsqueeze(0)

        return tensor_img.to(self.device_idx)
    
    def postprocess(self, pred):
        # 进行极大值抑制
        pred = non_max_suppression(pred)[0]
        if len(pred):
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_boxes(self.input_shape, pred[:, :4], self.img_shape).round()
        # result = self.convert_detection_result(pred, self.names, self.img_shape[1]/2)  
        return pred
    
    def draw_rec(self, results):
        # 不同颜色的列表，每个框将使用不同的颜色
        colors = [
            (255, 0, 0),   # 蓝色 
            (0, 255, 0),   # 绿色 
            (0, 0, 255),   # 红色 
            (128, 0, 128), # 紫色 
        ]

        # 颜色对应的中文名称
        color_names = [
            "蓝色",  
            "绿色",     
            "红色",   
            "紫色",   
        ]

        rec_info = []
        # 遍历结果并绘制框
        for idx, result in enumerate(results):
            # if idx != 0:
            #     continue
            x1, y1, x2, y2, confidence, class_id = result
            
            # 将坐标转换为整数
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 选择颜色
            color = colors[idx % len(colors)]
            
            # 画框
            cv2.rectangle(self.origin_img, (x1, y1), (x2, y2), color, 10)
            
            # 在框上添加标签和置信度
            label = f"{idx}"
            cv2.putText(self.origin_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 6)
            rec_info.append((idx, color_names[idx % 4]))
        cv2.imwrite('detect.png', self.origin_img)
        return rec_info
    
    def __call__(self, input_data):
        """
        对输入图像进行检测
        :param input_data: 输入图像
        :return: 检测结果
        """
        tensor_img = self.preprocess(input_data)
        pred = self.model(tensor_img)
        result = self.postprocess(pred)
        rec_info = self.draw_rec(result)
        return rec_info