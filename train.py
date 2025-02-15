#!/usr/bin/env python3
"""
旋转图片角度计算器。

Author: pankeyu
Date: 2022/05/17
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_filenames
from ImageDataset import RotateImageDataset
from iTrainingLogger import iSummaryWriter
from model import RotateNet

# 配置参数
batch_size = 32
n_epoch = 200  # 增加训练轮数
learning_rate = 0.0001  # 调整学习率
log_interval = 100  # 定义日志记录间隔
eval_interval = 500  # 定义评估间隔

# 创建保存模型的目录
os.makedirs('models', exist_ok=True)

# 初始化日志记录器
writer = iSummaryWriter(log_path='.', log_name='Rotate Net Training Log')

# 准备数据
data_path = os.path.join('./datasets', 'street_view')
train_filenames, test_filenames = get_filenames(data_path)

train_dataset = RotateImageDataset(input=train_filenames, input_shape=(3, 244, 244), normalize=True)
test_dataset = RotateImageDataset(input=test_filenames, input_shape=(3, 244, 244), normalize=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型和优化器
model = RotateNet()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 添加学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.1, 
    patience=5
)

def train():
    """训练分类器。"""
    for i in range(n_epoch):
        model.train()
        for batch_idx, (imgs, targets) in enumerate(train_dataloader):
            imgs = torch.as_tensor(imgs, dtype=torch.float32)
            targets = torch.as_tensor(targets, dtype=torch.long)
            
            logits = model(imgs)
            optimizer.zero_grad()
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            current_steps = i * len(train_dataloader) + batch_idx * batch_size
            if current_steps % log_interval == 0:
                writer.add_scalar('train_loss', loss.item(), current_steps)
                writer.record()
            
            if current_steps % eval_interval == 0:
                evaluate(current_steps)

def evaluate(current_steps: int):
    """测试训练器的效果。"""
    model.eval()
    with torch.no_grad():
        test_loss, correct = 0, 0
        for imgs, targets in test_dataloader:
            imgs = torch.as_tensor(imgs, dtype=torch.float32)
            targets = torch.as_tensor(targets, dtype=torch.long)
            
            logits = model(imgs)
            test_loss += criterion(logits, targets).item()
            pred = logits.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_dataloader.dataset)
        writer.add_scalar('eval_loss', test_loss, current_steps)
        writer.add_scalar('eval_acc', 100. * correct / len(test_dataloader.dataset), current_steps)
        writer.record()
        print('Eval Acc: {:.2f}%'.format(100. * correct / len(test_dataloader.dataset)))
        
        # 保存模型的状态字典而不是整个模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': correct / len(test_dataloader.dataset)
        }, f'models/model_{correct / len(test_dataloader.dataset):.2f}.pth')

if __name__ == '__main__':
    train()