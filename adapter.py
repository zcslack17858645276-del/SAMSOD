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
    

def inject_adapters(sam_model, split_ratio=0.5):
    print("Injecting Adapters into Image Encoder (Hiera)...")
    
    # 锁定 Trunk
    trunk = sam_model.image_encoder.trunk

    # 检查是否有 blocks 属性
    if not hasattr(trunk, 'blocks'):
        raise AttributeError("Error: trunk does not have 'blocks' attribute.")


    total_blocks = len(trunk.blocks)
    split_index = int(total_blocks * split_ratio)

    print(f"Total Blocks: {total_blocks}, Split Index: {split_index}")
    print(f"  - Blocks 0 to {split_index-1}: UNFREEZE (Full Fine-tuning)")
    print(f"  - Blocks {split_index} to {total_blocks-1}: FREEZE + ADAPTER")

    # trunk.blocks 是一个 nn.ModuleList(他没有stage参数，根据网络查询除了第三个stage，其他都是2个blocks，后续适配器可尝试添加在每个stage后)
    # large - ratio=0.5=>8千万，需要在解冻几层才会到达1B，这里的想法是在后面哪里添加注意力机制会比较好（或者搞双模态，或者在适配器里加注意力机制，去找点多尺度交叉融合的方法（这个应该比较好））
    # 先尝试适配器里加注意力机制，不过这个要怎么加，又或者怎么去做多尺度的连接
    # 欸，这样子搞，当前是最后一层输出到decoder，我在输入到decoder之前做一个新的融合，又或者说是加一些decoder，
    for i in range(total_blocks):
        block = trunk.blocks[i]
        
        if i < split_index:
            # === 前半部分：解冻 ===
            print(f"  Block {i}: Unfreezing parameters...")
            # 确保该 block 的所有参数需要梯度
            for param in block.parameters():
                param.requires_grad = True
                
        else:
            # === 后半部分：加 Adapter ===
            # 获取维度用于日志
            dim = block.norm1.normalized_shape[0] if hasattr(block, 'norm1') else "Unknown"
            print(f"  Block {i} (Dim {dim}): Injecting Adapter...")
            
            # 创建 Adapter (HieraBlockAdapter 的 __init__ 里会自动冻结原始 block)
            # 这里的 HieraBlockAdapter 使用定义的Adaptor类
            # adapter_ratio会影响Adapter里面隐藏层的维度
            wrapped_block = Adapter(block, adapter_ratio=0.25)
            wrapped_block = wrapped_block.to("cuda")
            
            # 替换
            trunk.blocks[i] = wrapped_block
        
    print("Adapter injection finished.")

    return sam_model