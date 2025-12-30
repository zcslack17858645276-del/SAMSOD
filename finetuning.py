from torchvision import transforms
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np

from sam2.build_sam import build_sam2
from dataloader import OurDataset
from options import get_argparser
from adapter import inject_adapters
from evaluator import SODEvaluator

args = get_argparser()

best_loss = float('inf')
global_step = 0 

# load model
sam2_model = build_sam2(
    args.model_cfg,
    ckpt_path=None,
    device=args.device
)

if args.sam2_checkpoint:
    print(f"Loading checkpoint from {args.sam2_checkpoint}...")
    state_dict = torch.load(args.sam2_checkpoint, map_location=args.device)
    
    # strict=False 允许部分权重缺失（因为我在后面添加了注意力机制）
    missing_keys, unexpected_keys = sam2_model.load_state_dict(state_dict, strict=False)
    
    # 打印一下确认，missing_keys 里应该包含你的 my_attention
    print(f"Missing keys (Expected for new modules): {missing_keys}")


# prepare dataloader
train_dataset = OurDataset(data_root=args.train_data_root)
val_dataset = OurDataset(data_root=args.val_data_root)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size_train,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size_val,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True
)

# freeze all layers first
for param in sam2_model.parameters():
    param.requires_grad = False

sam2_model = inject_adapters(sam2_model, split_ratio=0)
sam2_model = sam2_model.to(args.device) 

# unfreeze Mask Decoder
for param in sam2_model.sam_mask_decoder.parameters():
    param.requires_grad = True

# unfreeze Prompt Encoder
for param in sam2_model.sam_prompt_encoder.parameters():
    param.requires_grad = True

# 这里我是加了注意力机制，可能换其他的方法会更好？（没必要执着于注意力）
if hasattr(sam2_model, 'my_attention'):
    print("Unfreezing custom attention module...")
    for param in sam2_model.my_attention.parameters():
        param.requires_grad = True
else:
    print("Warning: 'my_attention' not found in model. Did you modify sam2_base.py?")

# collect trainable parameters
trainable_params = [p for p in sam2_model.parameters() if p.requires_grad]

# check trainable params
print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params)}")

# 打印第一个 Block，看它是否变成了 HieraBlockAdapter
print(f"Block 0 type: {type(sam2_model.image_encoder.trunk.blocks[0])}")

# define optimizer, scheduler, scaler
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=1000, # max iterations
    eta_min=1e-6
)
scaler = torch.amp.GradScaler(args.device)

def train_epoch(dataloader, model, optimizer, scheduler, scaler, global_step, epoch, args):
    print("============ train... ============")
    model.train()

    epoch_loss = 0.0
    for itr, batch_data in enumerate(dataloader):
        # init data
        images = batch_data['image'].cuda() # [B, 3, 1024, 1024]
        points = batch_data['points'].cuda() # [N, 2]
        labels = batch_data['labels'].cuda() # [N,]
        box = batch_data['box'].cuda() # [1, 4]
        gt_masks = batch_data['mask'].cuda() # [1, 1024, 1024]
        mask_inputs = batch_data['mask_prompt'].cuda() # [B, 1, 256, 256]

        with autocast(device_type=args.device):
            # =================================================
            # Image Encoder
            # =================================================
            # ago: {'image_embed': ..., 'high_res_feats': ...}
            # now: backbone_out
            features = model.forward_image(images)
            
            # 因为要调整，之后可能会关闭这个
            # SAM的decoder自己就做了多尺度的优化（输入high_res_features）
            high_res_features  = [
                features["backbone_fpn"][0],
                features["backbone_fpn"][1],
            ]

            # image_embed = features["image_embed"] # (B, 256, 64, 64)
            # high_res_features = features["high_res_feats"] # List of tensors

            image_embed = features["vision_features"]  # [B, 256, 64, 64]

            # =================================================
            # Prompt Encoder
            # =================================================
            
            sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                points=(points, labels),
                boxes=box,
                masks=mask_inputs,
            )

            # =================================================
            # Mask Decoder
            # =================================================
            # position embeddings
            image_pe = model.sam_prompt_encoder.get_dense_pe()


            low_res_masks, prd_scores, _, _ = model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            # =================================================
            # Compute Loss
            # =================================================
            # low_res_masks: [B, 3, 256, 256]
            pred_logits = low_res_masks[:, 0].unsqueeze(1) # [B, 1, 256, 256]
            
            # resize gt_masks to 256x256
            with torch.no_grad():
                gt_mask_256 = F.interpolate(gt_masks.float(), size=(256, 256), mode='nearest')
            
            # compute Loss (now: BCE, future: Dice + BCE)
            loss = F.binary_cross_entropy_with_logits(pred_logits, gt_mask_256)
            
        # =================================================
        # Backward & Optimize
        # =================================================
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # update global step
        global_step += 1
        epoch_loss += loss.item()

        # ============================================================
        # print loss
        # ============================================================
        if itr % args.print_interval == 0:
            print(f"Step [{itr}/{len(dataloader)}], "
                f"Total Loss: {loss.item():.4f}, "
                #f"Seg Loss: {seg_loss.item():.4f}, "
                #f"Score Loss: {score_loss.item():.4f}"
            )

    # ============================================================
    # save checkpoint
    # ============================================================    
    save_path = os.path.join(args.save_dir, f"sam2_finetune_epoch_{global_step}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Checkpint saved to: {save_path}")

    print(f"Epoch Training Completed. - Epoch {epoch}:  Avg Loss: {epoch_loss / len(dataloader):.4f}")

    return global_step, epoch_loss / len(dataloader)
    
@torch.no_grad() # when validating, no grad needed
def validate(model, dataloader, device, epoch):
    print("============ Validating... ============")
    model.eval()

    # 实例化评估器
    evaluator = SODEvaluator()

    total_val_loss = 0
    num_batches = 0

    for batch_data in dataloader:
        images = batch_data['image'].to(device)
        points = batch_data['points'].to(device)
        labels = batch_data['labels'].to(device)
        box = batch_data['box'].to(device) if 'box' in batch_data else None
        gt_masks = batch_data['mask'].to(device)

        with torch.amp.autocast(device):
            # --- Forward Pass ---
            features = model.forward_image(images)
            image_embed = features["vision_features"]
            high_res_features = [features["backbone_fpn"][0], features["backbone_fpn"][1]]

            sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                points=(points, labels),
                boxes=box,
                masks=None,
            )

            image_pe = model.sam_prompt_encoder.get_dense_pe()

            low_res_masks, _, _, _ = model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            # --- Compute Metrics ---
            pred_logits = low_res_masks[:, 0].unsqueeze(1)
            
            gt_mask_256 = F.interpolate(gt_masks.float(), size=(256, 256), mode='nearest')
            
            # Val Loss
            loss = F.binary_cross_entropy_with_logits(pred_logits, gt_mask_256)
            total_val_loss += loss.item()

            evaluator.update(pred_logits, gt_mask_256)
            
            
            num_batches += 1

    avg_loss = total_val_loss / num_batches
    metrics = evaluator.get_results()

    return avg_loss, metrics

for epoch in range(args.num_epochs):
    print(f"Starting Epoch {epoch+1}/{args.num_epochs}...")
    
    # train
    global_step, train_avg_loss = train_epoch(
        dataloader=train_dataloader,
        model=sam2_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        global_step=global_step,
        epoch=epoch,
        args=args
    )

    print(f"Avg Train Loss: {train_avg_loss:.4f}")

    val_loss, res = validate(sam2_model, val_dataloader, args.device, epoch)

    print(f"Epoch {epoch+1} Finished. Validation Results:")
    print(f"  Loss:   {val_loss:.4f}")
    print(f"  MAE:    {res['Mae']:.4f}")
    print(f"  Sm:     {res['Sm']:.4f}")
    print(f"  WFm:     {res['WFm']:.4f}")
    print(f"  MaxF:   {res['MaxF']:.4f}")
    print(f"  MaxEm:   {res['MaxEm']:.4f}")
    print(f"  MeanEm: {res['MeanEm']:.4f}")

    # Saving the last modal
    # 这里也要改一下
    final_path = os.path.join(args.save_dir, "sam2_final.pt")
    torch.save(sam2_model.state_dict(), final_path)
    print("Epoch Training Finished. Model saved.")


