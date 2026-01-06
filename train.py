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
from loss import SODLoss
from transforms import Compose, Resize, RandomHVFlip, RandomRotate, ToTensorAndNormalize


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.augment = args.augment
        self.sam2_size = args.input_size

        self.global_step = 0
        self.best_loss = float("inf")

        self.model = self.build_model()
        self.train_loader, self.val_loader = self.build_dataloader()
        self.freeze_and_unfreeze()

        self.optimizer, self.scheduler, self.scaler = self.configure_optimizer()
        self.criterion = self.configure_criterion()

    def build_model(self):
        model = build_sam2(
            self.args.model_cfg,
            ckpt_path=None,
            device=self.device
        )

        if self.args.sam2_checkpoint:
            print(f"Loading checkpoint from {self.args.sam2_checkpoint}...")
            state_dict = torch.load(self.args.sam2_checkpoint, map_location=self.device)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Missing keys (Expected): {missing_keys}")

        model = inject_adapters(model, split_ratio=0)
        model = model.to(self.device)

        return model
    
    def build_dataloader(self):
        # definite the augment stratergy
        train_transforms, val_transforms = None, None
        if self.augment:
            train_transforms = Compose([
                RandomHVFlip(prob=0.5),
                RandomRotate(degree=15, prob=0.5),
                Resize(self.sam2_size),
                ToTensorAndNormalize()
            ])
            val_transforms = Compose([
                Resize(self.sam2_size),
                ToTensorAndNormalize()
            ])

        train_dataset = OurDataset(data_root=self.args.train_data_root, transform=train_transforms)
        #val_dataset = OurDataset(data_root=self.args.val_data_root, transform=val_transforms)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size_train,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        val_loaders = []
        for val_root in self.args.val_data_root:
            print(f"Loading Val Dataset: {val_root}")
            val_dataset = OurDataset(
                data_root=val_root, 
                transform=val_transforms, 
            )
            
            # 每个数据集一个独立的 Loader
            loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size_val,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=False
            )
            val_loaders.append(loader)
        '''
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size_val,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        '''

        return train_loader, val_loaders

    def freeze_and_unfreeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

        # Mask Decoder
        for p in self.model.sam_mask_decoder.parameters():
            p.requires_grad = True

        # Prompt Encoder
        for p in self.model.sam_prompt_encoder.parameters():
            p.requires_grad = True

        # Custom Attention
        if hasattr(self.model, "my_attention"):
            print("Unfreezing custom attention module...")
            for p in self.model.my_attention.parameters():
                p.requires_grad = True
        else:
            print("Warning: my_attention not found")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Total trainable params: {sum(p.numel() for p in trainable_params)}")
        print(f"Block 0 type: {type(self.model.image_encoder.trunk.blocks[0])}")

    def configure_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1000,
            eta_min=1e-6
        )

        scaler = torch.amp.GradScaler(self.device)

        return optimizer, scheduler, scaler

    def configure_criterion(self):
        bce_weight = self.args.bce_weight
        dice_weight = self.args.dice_weight
        ssim_weight = self.args.ssim_weight
        criterion = SODLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            ssim_weight=ssim_weight
        ).cuda()
        return criterion

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0

        for itr, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            points = batch["points"].to(self.device)
            labels = batch["labels"].to(self.device)
            box = batch["box"].to(self.device)
            gt_masks = batch["mask"].to(self.device)
            mask_inputs = batch["mask_prompt"].to(self.device)

            with autocast(device_type=self.device):
                features = self.model.forward_image(images)

                image_embed = features["vision_features"]
                high_res_features = [
                    features["backbone_fpn"][0],
                    features["backbone_fpn"][1],
                ]

                sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                    points=(points, labels),
                    boxes=box,
                    masks=mask_inputs,
                )

                image_pe = self.model.sam_prompt_encoder.get_dense_pe()

                low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )

                # 配合 SSIM Loss 捕捉细节, 上采样
                pred_masks_1024 = F.interpolate(
                    low_res_masks, 
                    size=(1024, 1024), 
                    mode="bilinear", 
                    align_corners=False
                )

                #pred_logits = low_res_masks[:, 0].unsqueeze(1)
                #gt_256 = F.interpolate(gt_masks.float(), size=(256, 256), mode="nearest")

                #loss = F.binary_cross_entropy_with_logits(pred_logits, gt_256)
                loss, loss_bce, loss_dice, loss_ssim = self.criterion(pred_masks_1024, gt_masks)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # self.scheduler.step()

            self.global_step += 1
            epoch_loss += loss.item()

            if itr % self.args.print_interval == 0:
                print(f"[Epoch {epoch}] Step {itr}: Total={loss.item():.4f} "
                      f"(BCE={loss_bce.item():.4f}, "
                      f"Dice={loss_dice.item():.4f}, "
                      f"SSIM={loss_ssim.item():.4f})"
                )

        if self.scheduler is not None:
            self.scheduler.step()
            print(f"LR: {self.scheduler.get_last_lr()}")

        return epoch_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()

        all_metrics = {}
        total_avg_loss = 0
        for loader in self.val_loader:
            dataset_name = loader.dataset.dataset_name
            evaluator = SODEvaluator() 
            dataset_loss = 0.0
            print(f"Validating on {dataset_name}...") 

            for batch in loader:
                images = batch["image"].to(self.device)
                points = batch["points"].to(self.device)
                labels = batch["labels"].to(self.device)
                box = batch["box"].to(self.device)
                gt_masks = batch["mask"].to(self.device)
                mask_inputs = batch["mask_prompt"].to(self.device)

                with autocast(device_type=self.device):
                    features = self.model.forward_image(images)
                    image_embed = features["vision_features"]
                    high_res_features = [
                        features["backbone_fpn"][0],
                        features["backbone_fpn"][1],
                    ]

                    sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                        points=(points, labels),
                        boxes=box,
                        masks=mask_inputs,
                    )

                    image_pe = self.model.sam_prompt_encoder.get_dense_pe()

                    low_res_masks, _, _, _ = self.model.sam_mask_decoder(
                        image_embeddings=image_embed,
                        image_pe=image_pe,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=high_res_features,
                    )

                    pred_masks_1024 = F.interpolate(
                        low_res_masks, 
                        size=(1024, 1024), 
                        mode="bilinear", 
                        align_corners=False
                    )

                    loss, _, _, _ = self.criterion(pred_masks_1024, gt_masks)
                    dataset_loss += loss.item()

                    evaluator.update(pred_masks_1024, gt_masks)

                    #pred_logits = low_res_masks[:, 0].unsqueeze(1)
                    #gt_256 = F.interpolate(gt_masks.float(), size=(256, 256), mode="nearest")

                    #loss = F.binary_cross_entropy_with_logits(pred_logits, gt_256)
                    #total_loss += loss.item()
                    #evaluator.update(pred_logits, gt_256)

            # 计算当前数据集的平均 Loss
            avg_dataset_loss = dataset_loss / len(loader)
            total_avg_loss += avg_dataset_loss
            
            metrics = evaluator.get_results()
            
            for k, v in metrics.items():
                all_metrics[f"{dataset_name}/{k}"] = v
            all_metrics[f"{dataset_name}/loss"] = avg_dataset_loss


        final_val_loss = total_avg_loss / len(self.val_loader)

        return final_val_loss / len(self.val_loader), all_metrics
    
    def run(self):
        for epoch in range(self.args.num_epochs):
            print(f"\n===== Epoch {epoch+1}/{self.args.num_epochs} =====")

            train_loss = self.train_one_epoch(epoch)
            val_loss, metrics = self.validate(epoch)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Avg Val Loss: {val_loss:.4f}")

            sorted_keys = sorted(metrics.keys())
            current_dataset = ""
            for k in sorted_keys:
                # k 格式如 "ECSSD/mae"
                dataset_name, metric_name = k.split('/')
                if dataset_name != current_dataset:
                    print(f"--- {dataset_name} ---")
                    current_dataset = dataset_name
                print(f"  {metric_name}: {metrics[k]:.4f}")

            # 保存最佳模型逻辑
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                save_path = os.path.join(self.args.save_dir, "best_model.pt")
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")

            # 定期保存
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(
                    self.args.save_dir, f"sam2_epoch_{epoch+1}.pt"
                )
                torch.save(self.model.state_dict(), save_path)


def main():
    args = get_argparser()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    trainer = Trainer(args)
    
    trainer.run()


if __name__ == "__main__":
    main()





