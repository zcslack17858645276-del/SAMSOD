import numpy as np
import torch
from metric import prepare_data, MAE, Fmeasure, Smeasure, Emeasure, WeightedFmeasure


class SODEvaluator:
    def __init__(self):

        self.mae = MAE()
        self.fm = Fmeasure()
        self.wfm = WeightedFmeasure()
        self.sm = Smeasure()
        self.em = Emeasure()

    def update(self, pred_logits: torch.Tensor, gt_mask: torch.Tensor):
        """
        处理一个 batch 的数据
        pred_logits: [B, 1, H, W] (Tensor)
        gt_mask:     [B, 1, H, W] (Tensor)
        """
        # Sigmoid 激活
        pred_probs = torch.sigmoid(pred_logits)

        # [B, 1, H, W] -> [B, H, W]
        # 移除 channel=1 的维度，方便后续处理
        pred_np = (pred_probs * 255).squeeze(1).cpu().detach().numpy()
        gt_np = (gt_mask * 255).squeeze(1).cpu().detach().numpy().astype(np.uint8)
        # 问题就在这里，可让我好找(⚪^⚪)
        #print(f"pred: {pred_np}, gt: {gt_np}")

        # 遍历 Batch 中的每一张图
        batch_size = pred_np.shape[0]
        for i in range(batch_size):

            p = pred_np[i]
            g = gt_np[i]

            self.mae.step(p, g)
            self.fm.step(p, g)
            self.wfm.step(p, g)
            self.sm.step(p, g)
            self.em.step(p, g)

    def get_results(self):
        """汇总结果"""
        results = {}

        results['WFm'] = self.wfm.get_results()["wfm"]
        results['MaxF'] = self.fm.get_results()["mf"]
        results['Mae'] = self.mae.get_results()["mae"]
        results['Sm'] = self.sm.get_results()["sm"]
        em = self.em.get_results()["em"]
        results['MeanEm'] = em["curve"].mean()
        results['MaxEm'] = em["curve"].max()

        return results