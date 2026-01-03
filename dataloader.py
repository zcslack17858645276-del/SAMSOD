import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from torchvision.transforms import functional as F

from options import get_argparser
from transforms import Compose, Resize, RandomRotate, RandomHVFlip, ToTensorAndNormalize


class OurDataset(Dataset):
    def __init__(self, data_root, transform=None):

        self.data_root = data_root
        self.images_path = os.path.join(data_root, "im")
        self.masks_path = os.path.join(data_root, "gt")

        # the target input size for Image Encoder
        input_size = get_argparser().input_size
        self.target_size = input_size

        # 如果没有传入transform，定义默认的transforms流程
        if transform is None:
            # 默认只做Resize 和 ToTensor
            self.tranforms = Compose([
                Resize(input_size),
                ToTensorAndNormalize()
            ])
        else:
            self.tranforms = transform
        
        # get all image filenames
        # 这里在小规模测试，记得删[:200]
        self.filenames = [f for f in os.listdir(self.images_path) if f.endswith('.jpg') or f.endswith('.png')][:200]

        # the mask prompt size for prompt encoder
        self.prompt_mask_size = 256

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        # construct full paths
        img_path = os.path.join(self.images_path, fname)
        
        fname_no_ext = os.path.splitext(fname)[0]
        mask_name = fname_no_ext + ".png" 
        mask_path = os.path.join(self.masks_path, mask_name)
        
        # read image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转 RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # check if image and mask are loaded properly
        if image is None or mask is None:
            raise FileNotFoundError(f"image load failed {img_path} or {mask_path}")

        '''
        the origion logits(befor data_preprocess)    
        # get original size
        orig_h, orig_w = image.shape[:2]

        # ==================================================
        # generate prompts from mask(now: points + box, future: mask)
        # ==================================================
        points, labels = self._simulate_click_from_mask(mask)
        box = self._simulate_box_from_mask(mask) # (1, 4)

        # ==================================================
        # Generate Mask Prompt
        # ==================================================
        # 生成一个低分辨率、带噪声的 Mask 用于提示
        mask_prompt_np = self._preprocess_mask_prompt(mask)
        
        # ==================================================
        # resize & (future transform)
        # ==================================================
        image_1024 = cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        # resize mask
        mask_1024 = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        
        # scale points and box
        scale_x = self.target_size / orig_w
        scale_y = self.target_size / orig_h

        if box.sum() > 0: # non-zero box
            box[:, 0] *= scale_x # x_min
            box[:, 2] *= scale_x # x_max
            box[:, 1] *= scale_y # y_min
            box[:, 3] *= scale_y # y_max

        tensor_box = torch.from_numpy(box).float()
        
        points[:, 0] *= scale_x # x
        points[:, 1] *= scale_y # y

        # ==================================================
        # to Tensor
        # ==================================================
        # [3, 1024, 1024] to Normalize
        tensor_img = F.to_tensor(image_1024)
        tensor_img = F.normalize(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Mask: [1, 1024, 1024] to binary
        tensor_mask = torch.from_numpy(mask_1024).float().unsqueeze(0)
        tensor_mask = (tensor_mask > 0).float() # Binarize
        '''
        
        # get the tensor img and mask
        tensor_img, tensor_mask = self.tranforms(image, mask)

        # generate the prompt
        mask_np_1024 = tensor_mask.squeeze(0).cpu().numpy().astype(np.uint8)
        points, labels = self._simulate_click_from_mask(mask_np_1024)
        box = self._simulate_box_from_mask(mask_np_1024)
        mask_prompt_np = self._preprocess_mask_prompt(mask_np_1024)
        
        # Points: (N, 2) -> FloatTensor
        tensor_points = torch.from_numpy(points).float()

        # Labels: (N,) -> IntTensor (1=foreground, 0=background)
        tensor_labels = torch.from_numpy(labels).long()

        #
        tensor_box = torch.from_numpy(box).float()

        # === Mask Prompt Tensor [1, 256, 256] ===
        tensor_mask_prompt = torch.from_numpy(mask_prompt_np).float().unsqueeze(0)
        # 确保也是 0/1 (SAM 内部会处理，但输入干净点比较好)
        tensor_mask_prompt = (tensor_mask_prompt > 0).float()
    

        return {
            "image": tensor_img,   # [3, 1024, 1024]
            "mask": tensor_mask,   # [1, 1024, 1024]
            "mask_prompt": tensor_mask_prompt, # [1, 256, 256]
            "points": tensor_points, # [N, 2]
            "labels": tensor_labels, # [N,]
            "box": tensor_box,     # [1, 4]
            "orig_size": (self.target_size, self.target_size)  # [H, W]
        }

    def _simulate_click_from_mask(self, mask):
        """
        get a random foreground point from the mask as positive point
        """
        # find all foreground pixel coordinates
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) > 0:
            # select a random foreground pixel
            random_idx = np.random.randint(0, len(y_indices))
            x = x_indices[random_idx]
            y = y_indices[random_idx]
            
            # [[x, y]]
            points = np.array([[x, y]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32) # 1 positive sample
        else:
            # if no foreground, return a dummy negative point
            points = np.array([[0, 0]], dtype=np.float32)
            labels = np.array([-1], dtype=np.int32) # -1 indicates no valid point
            
        return points, labels
    
    def _simulate_box_from_mask(self, mask):
        """
        Bounding Box
        return: np.array([[x_min, y_min, x_max, y_max]])
        """
        # find all foreground pixel coordinates
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) > 0:
            # compute bounding box
            x_min = np.min(x_indices)
            x_max = np.max(x_indices)
            y_min = np.min(y_indices)
            y_max = np.max(y_indices)
            
            # add some random jitter
            x_min = max(0, x_min - np.random.randint(0, 5))
            x_max = min(mask.shape[1], x_max + np.random.randint(0, 5))
            y_min = max(0, y_min - np.random.randint(0, 5))
            y_max = min(mask.shape[1], y_max + np.random.randint(0, 5))
            
            # box: [[x_min, y_min, x_max, y_max]]，shape: [1, 4]
            box = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
        else:
            # no foreground, return a zero box
            box = np.array([[0, 0, 0, 0]], dtype=np.float32)
            
        return box
    
    def _preprocess_mask_prompt(self, mask):
        """
        处理步骤：
        1. Resize 到 256x256
        2. 数据增强（随机腐蚀/膨胀）模拟不准确的提示
        """
        # 1. Resize 到 256x256
        mask_low_res = cv2.resize(mask, (self.prompt_mask_size, self.prompt_mask_size), interpolation=cv2.INTER_NEAREST)

        # 2. 随机加噪声 (Simulate Noise)
        # 如果是全黑mask就不处理了，否则进行扰动
        if mask_low_res.max() > 0:
            prob = np.random.random()
            
            kernel_size = np.random.randint(3, 8) # 随机核大小
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            if prob < 0.4:
                # 40% 概率：腐蚀 (提示比真实物体小)
                mask_low_res = cv2.erode(mask_low_res, kernel, iterations=1)
            elif prob < 0.8:
                # 40% 概率：膨胀 (提示比真实物体大)
                mask_low_res = cv2.dilate(mask_low_res, kernel, iterations=1)
            # 20% 概率：保持原样 (精准提示)
        
        return mask_low_res