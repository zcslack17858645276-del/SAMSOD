import cv2
import numpy as np
import random
import torch
from torchvision.transforms import functional as F


class Compose:
    """组合多个变换"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
    
class Resize:
    """
    同时调整图像和掩码的大小。
    Image 使用 Linear 插值，Mask 使用 Nearest 插值。
    """
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image, mask):
        # image: [H, W, 3]
        # mask: [H, W]
        image = cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        return image, mask
    
    
class RandomHVFlip:
    """Random horizontal or vertical flip"""
    def __init__(self, prob=0.5):
        self.prob=prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            if random.random() < 0.5: # H-wise
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            else: # V-wise
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)

        return image, mask
    
class RandomRotate:
    """
    random rotate (-degree, +degree)
    Image: fill edge mirror
    Mask:fill black(0)
    """
    def __init__(self, degree=15, prob=0.5):
        self.degree = degree
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            angle = np.random.uniform(-self.degree, self.degree)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # fill the image
            image = cv2.warpAffine(
                image, M, (w, h), 
                flags=cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_REFLECT
            )
            # fill the mask
            mask = cv2.warpAffine(
                mask, M, (w, h), 
                flags=cv2.INTER_NEAREST, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
        return image, mask
    
class ToTensorAndNormalize:
    """
    transform numpy to tensor and normalize
    return tensor
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        # Image [H, W, 3] -> Tensor [3, H, W] -> Normalize
        img_tensor = F.to_tensor(image)
        img_tensor = F.normalize(img_tensor, mean=self.mean, std=self.std)
        
        # Mask [H, W] -> Tensor [1, H, W]
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        mask_tensor = (mask_tensor > 0).float() # binary mask
        
        return img_tensor, mask_tensor
