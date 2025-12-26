from sam2.build_sam import build_sam2
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

from sam2.utils.transforms import SAM2Transforms
from options import get_argparser
from adapter import Adapter, inject_adapters


args = get_argparser()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def manual_postprocess_masks(low_res_masks, input_size, original_size):
    """
    low_res_masks: [B, C, 256, 256]
    input_size: [1024, 1024] - the input size for SAM2
    original_size: [H, W] - the original image size
    """
    
    # resize to input_size first
    masks = F.interpolate(
        low_res_masks,
        size=input_size,
        mode="bilinear",
        align_corners=False,
    )
    
    # resize to original_size
    masks = F.interpolate(
        masks,
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )
    
    return masks


# build SAM2 model
model_cfg = args.model_cfg

sam2_model = build_sam2(model_cfg, ckpt_path=None, device=args.device)

# 这里到时候重写一个集成好的网络然后调用，不然会比较麻烦
sam2_model = inject_adapters(sam2_model, split_ratio=0) 
sam2_model = sam2_model.to(args.device)

# upload finetuned weights
finetuned_ckpt_path = args.predict_checkpoint
weights = torch.load(finetuned_ckpt_path, map_location=args.device)

# (future: strict to False)
msg = sam2_model.load_state_dict(weights, strict=True)
print(f"Loaded finetuned weights: {msg}")

# build predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor(sam2_model)

# test image
image_path = "image/sun.jpg" 

if not os.path.exists(image_path):
    print(f"Warning: {image_path} not found. Generating a random image for testing.")
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
else:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


with torch.no_grad(), torch.autocast(device_type=args.device):
    # preprocess image
    tr = SAM2Transforms(resolution=args.input_size, mask_threshold=0.0)
    input_image = tr(image) # 这里的 image 是 numpy (H, W, 3)
    input_image = input_image.unsqueeze(0).to(args.device) # (1, 3, 1024, 1024)

    # get image features
    features = sam2_model.forward_image(input_image)
    image_embed = features["vision_features"]  # [B, 256, 64, 64]
    high_res_features  = [
        features["backbone_fpn"][0],
        features["backbone_fpn"][1],
    ]

    # generate prompts
    scale_x = args.input_size / image.shape[1]
    scale_y = args.input_size / image.shape[0]
    # future: box(now: have some problems)
    points = torch.tensor([[[image.shape[1]//2 * scale_x, image.shape[0]//2 * scale_y]]], dtype=torch.float, device=args.device) # 中心点
    labels = torch.tensor([[1]], dtype=torch.int, device=args.device) # 正样本

    sparse_embeddings, dense_embeddings = sam2_model.sam_prompt_encoder(
        points=(points, labels),
        boxes=None,
        masks=None,
    )

    # mask decoder
    image_pe = sam2_model.sam_prompt_encoder.get_dense_pe()
    
    low_res_masks, iou_predictions, _, _ = sam2_model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=False,
        high_res_features=high_res_features,
    )
    
    # post-process masks to original size
    prd_masks = manual_postprocess_masks(
        low_res_masks, input_size=(args.input_size, args.input_size), original_size=image.shape[:2]
    )
    
    # select best mask based on iou_predictions
    best_idx = torch.argmax(iou_predictions[0])
    final_mask = prd_masks[0, best_idx].sigmoid().cpu().numpy() > 0.5

    # visualize
    plt.imshow(image)
    plt.imshow(final_mask, alpha=0.5)
    plt.show()