import torch
from torch import nn

import torch
import torch.nn as nn

class Adapter(nn.Module):
    """
    Adapter for Hiera MultiScaleBlock.
    Logic: Output = Block(x + Adapter(x))
    """
    def __init__(self, target_block, adapter_ratio=0.25):
        super().__init__()
        self.block = target_block
        
        # 冻结原始 Block
        for p in self.block.parameters():
            p.requires_grad = False
            
        # 自动获取维度
        try:
            dim = self.block.norm1.normalized_shape[0]
        except AttributeError:
            # 如果找不到 norm1，尝试找 attn.qkv
            dim = self.block.attn.qkv.in_features
            
        # 定义 bottleneck adapter
        hidden_dim = int(dim * adapter_ratio)
        # 避免 hidden_dim 太小
        hidden_dim = max(hidden_dim, 1) 
        
        self.adapter_layer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # 零初始化 (Zero Initialization) - 保证初始状态下，adapter 输出为 0，不影响预训练权重
        nn.init.zeros_(self.adapter_layer[-1].weight)
        nn.init.zeros_(self.adapter_layer[-1].bias)

    def forward(self, x, *args, **kwargs):
        # x shape: [B, H, W, C] (Hiera 默认是 Channel Last)
        
        # 计算 Adapter 增量
        prompt = self.adapter_layer(x)
        
        # Residual
        prompted_x = x + prompt
        
        # 输入原始 Block
        return self.block(prompted_x, *args, **kwargs)

def get_hiera_depths(total_blocks):
    """
    根据 Block 总数判断 Hiera 的规格，返回每个 Stage 的层数 (Depths)
    """
    if total_blocks == 48:
        return [2, 6, 36, 4]  # Hiera-Large (SAM 2 默认)
    elif total_blocks == 24:
        return [2, 3, 16, 3]  # Hiera-Base+
    elif total_blocks == 16:
        return [1, 2, 11, 2]  # Hiera-Small
    elif total_blocks == 12:
        return [1, 2, 7, 2]   # Hiera-Tiny
    else:
        raise ValueError(f"Unknown Hiera architecture with {total_blocks} blocks.")

def inject_adapters(sam_model, split_ratio=0.5):
    print("Injecting Adapters into Image Encoder (Hiera)...")
    
    # 锁定 Trunk
    trunk = sam_model.image_encoder.trunk

    # 检查是否有 blocks 属性
    if not hasattr(trunk, 'blocks'):
        raise AttributeError("Error: trunk does not have 'blocks' attribute.")


    total_blocks = len(trunk.blocks)
    depths = get_hiera_depths(total_blocks)
    
    print(f"  - Model Architecture Detected: Total {total_blocks} blocks")
    print(f"  - Stage Depths: {depths}")

    stage_end_indices = [sum(depths[:i+1]) - 1 for i in range(len(depths))]

    # trunk.blocks 是一个 nn.ModuleList
    # large - ratio=0.5=>8千万，需要在解冻几层才会到达1B，这里的想法是在后面哪里添加注意力机制会比较好（或者搞双模态，或者在适配器里加注意力机制，去找点多尺度交叉融合的方法（这个应该比较好））
    for i, block in enumerate(trunk.blocks):
        if i in stage_end_indices:
            print(f"  Block {i} is the end of a Stage. Injecting Adapter...")

            # === 注入 Adapter ===
        
            # 使用你的 Adapter 类
            # 注意：Hiera 不同 Stage 的维度不同 (L: 144 -> 288 -> 576 -> 1152)
            # 你的 Adapter 类需要能自动适应输入 block 的维度
            wrapped_block = Adapter(block, adapter_ratio=0.25)
            wrapped_block = wrapped_block.to(sam_model.device) # 确保设备一致
            
            trunk.blocks[i] = wrapped_block
            
            # 确保 Adapter 参数可训练
            for p in trunk.blocks[i].parameters():
                p.requires_grad = True

    return sam_model