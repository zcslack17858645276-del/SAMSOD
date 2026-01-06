import torch

obj = torch.load("./checkpoints_finetuned/sam2_final.pt", map_location="cpu")

for k in obj.keys():
    print(k)
