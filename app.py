import gradio as gr
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from options import get_argparser
from sam2.build_sam import build_sam2

from adapter import inject_adapters
import torch

args = get_argparser()

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
msg = sam2_model.load_state_dict(weights, strict=False)
print(f"Loaded finetuned weights: {msg}")

# build predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor(sam2_model)

def sod_predict(input_image):
    if input_image is None:
        return None
        
    # --- 全图 Box ---
    predictor.set_image(input_image)
    H, W = input_image.shape[:2]
    masks, _, _ = predictor.predict(box=np.array([[0, 0, W, H]]), multimask_output=False)
    result_mask = masks[0]
    
    # 可视化：将 Mask 叠加到原图
    # 简单的绿色半透明覆盖
    colored_mask = np.zeros_like(input_image)
    colored_mask[result_mask > 0] = [0, 255, 0] # 绿色
    
    # 融合
    vis_image = input_image.copy()
    alpha = 0.5
    mask_indices = result_mask > 0
    vis_image[mask_indices] = (input_image[mask_indices] * (1 - alpha) + 
                               colored_mask[mask_indices] * alpha).astype(np.uint8)
    
    return vis_image

# --- 界面 ---
custom_css = ".output-image {height: 400px;}"

with gr.Blocks(theme=gr.themes.Soft(), title="SOD Auto Demo") as demo:
    gr.Markdown("# 🚀 显著性目标自动检测 (SOD)")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="上传图片", type="numpy")
            # 使用 change 事件，图片上传完毕自动触发，不需要点按钮
            # 也可以加个按钮手动触发
            btn = gr.Button("开始检测", variant="primary")
            
        with gr.Column():
            img_output = gr.Image(label="检测结果")
            
    # 绑定事件
    btn.click(fn=sod_predict, inputs=img_input, outputs=img_output)
    
    # 或者上传即预测
    # img_input.change(fn=sod_predict, inputs=img_input, outputs=img_output)

demo.launch()

def predict_automatic(image):
    """
    image: numpy array (H, W, 3)
    """
    predictor.set_image(image)
    
    H, W = image.shape[:2]
    
    # 构建一个覆盖全图的 Box [x1, y1, x2, y2]
    # 提示模型：在这个范围内找最显著的东西
    box_prompt = np.array([[0, 0, W, H]]) 
    
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_prompt,  # 传入全图 Box
        multimask_output=False # SOD 通常只需要一个输出
    )
    
    # masks[0] 就是结果
    return masks[0]