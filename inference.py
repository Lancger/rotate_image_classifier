#!/usr/bin/env python3
"""
验证训练模型。

Author: pankeyu
Date: 2022/05/19
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.serialization import add_safe_globals
from utils import RotNetDataset
from model import RotateNet

torch.nn.Module.dump_patches = True
input_shape = (3, 244, 244)
add_safe_globals([RotateNet])

def calculate_angle_error(pred_angle, true_angle):
    """
    计算角度误差，考虑角度的循环性
    """
    diff = abs(pred_angle - true_angle)
    return min(diff, 360 - diff)

if __name__ == '__main__':
    with torch.no_grad():
        model = RotateNet()
        try:
            checkpoint = torch.load(
                'models/model_13.pth',
                map_location=torch.device('cpu'),
                weights_only=False
            )
            
            if isinstance(checkpoint, RotateNet):
                model = checkpoint
            elif isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            
            model.eval()
            
            # 准备数据
            data_path = './img_examples'
            files = os.listdir(data_path)
            labels = [int(file.split('.')[0].split('_')[1]) for file in files]  # 确保标签是整数
            files = [os.path.join(data_path, f) for f in files]
            
            # 创建数据加载器
            val_dataset = RotNetDataset(files, input_shape=input_shape, rotate=False, normalize=True)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)  # 使用batch_size=1便于调试
            
            # 进行预测
            all_preds = []
            all_labels = []
            
            for (img, _), label in zip(val_dataloader, labels):
                img = img.float()
                logits = model(img)
                probs = F.softmax(logits, dim=-1)
                
                # 获取最可能的角度和其概率
                pred_prob, pred_angle = torch.max(probs, dim=1)
                pred_angle = pred_angle.item()
                
                all_preds.append(pred_angle)
                all_labels.append(label)
                
                # 打印详细信息
                angle_error = calculate_angle_error(pred_angle, label)
                print(f'预测角度: {pred_angle:3d}° | 实际角度: {label:3d}° | 误差: {angle_error:3d}° | 置信度: {pred_prob.item():.2f}')
            
            # 计算平均误差
            total_error = 0
            for pred, label in zip(all_preds, all_labels):
                error = calculate_angle_error(pred, label)
                total_error += error
            
            avg_error = total_error / len(all_preds)
            print('\n统计信息:')
            print(f'平均角度误差: {avg_error:.2f}°')
            print(f'最大角度误差: {max([calculate_angle_error(p, l) for p, l in zip(all_preds, all_labels)])}°')
            print(f'最小角度误差: {min([calculate_angle_error(p, l) for p, l in zip(all_preds, all_labels)])}°')
            
        except Exception as e:
            print(f"错误: {e}")
            print("请确保模型文件存在且格式正确")