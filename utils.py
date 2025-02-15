import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def get_filenames(path: str):
    """
    读取一个文件夹下的所有图片。

    Args:
        path (str): 图片文件夹路径

    Returns:
        tuple: (训练文件列表, 测试文件列表)
    """
    image_paths = []
    for filename in os.listdir(path):
        if not filename.endswith('.jpg'):
            continue
        view_id = filename.split('_')[1].split('.')[0]
        # ignore images with markers (0) and upward views (5)
        if not (view_id == '0' or view_id == '5'):
            image_paths.append(os.path.join(path, filename))

    # 90% train images and 10% test images
    n_train_samples = int(len(image_paths) * 0.9)
    train_filenames = image_paths[:n_train_samples]
    test_filenames = image_paths[n_train_samples:]

    return train_filenames, test_filenames


def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - abs(abs(x - y) - 180)


class RotNetDataset(Dataset):
    """
    Dataset class for loading and preprocessing images for rotation prediction.
    """
    def __init__(self, files, input_shape=(3, 224, 224), rotate=True, normalize=True):
        """
        初始化数据集。

        Args:
            files: 图片文件路径列表
            input_shape: 输入图片的形状 (channels, height, width)
            rotate: 是否随机旋转图片
            normalize: 是否对图片进行标准化
        """
        self.files = files
        self.input_shape = input_shape
        self.rotate = rotate
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        获取一个数据样本。

        Args:
            idx: 索引

        Returns:
            tuple: (处理后的图片, 旋转角度)
        """
        # 读取图片
        img_path = self.files[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")
        
        # 调整图片大小
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 转换为 CHW 格式
        img = np.transpose(img, (2, 0, 1))

        # 如果需要旋转
        if self.rotate:
            rotate_angle = np.random.randint(360)
            # 转回 HWC 格式进行旋转
            img = np.transpose(img, (1, 2, 0))
            matrix = cv2.getRotationMatrix2D(
                (self.input_shape[1]//2, self.input_shape[2]//2),
                rotate_angle,
                1.0
            )
            img = cv2.warpAffine(
                img,
                matrix,
                (self.input_shape[1], self.input_shape[2])
            )
            # 转回 CHW 格式
            img = np.transpose(img, (2, 0, 1))
        else:
            rotate_angle = 0

        # 标准化
        if self.normalize:
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            for i in range(3):
                img[i] = (img[i] - mean[i]) / std[i]

        return img, rotate_angle


if __name__ == "__main__":
    # 测试代码
    import os
    import random
    
    # 测试数据集
    input_shape = (3, 244, 244)
    data_path = os.path.join('./datasets', 'street_view')
    
    # 获取文件列表
    train_filenames, test_filenames = get_filenames(data_path)
    
    # 检查文件列表
    if not train_filenames:
        print("警告：没有找到训练文件！")
        exit(1)
    
    # 创建数据集实例
    train_dataset = RotNetDataset(
        input=train_filenames,
        input_shape=input_shape,
        normalize=True
    )
    
    # 打印数据集信息
    total_samples = len(train_dataset)
    print(f"数据集中共有 {total_samples} 个样本")
    
    # 创建示例输出目录
    os.makedirs('./img_examples', exist_ok=True)
    
    # 生成一些示例
    for i in range(min(5, total_samples)):
        img, angle = train_dataset[i]
        # 转换回 HWC 格式用于保存
        img = np.transpose(img, (1, 2, 0))
        # 反标准化
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 保存图片
        cv2.imwrite(f'./img_examples/angle_{angle}.png', img)
        print(f"已保存图片 angle_{angle}.png")