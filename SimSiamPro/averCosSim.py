import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F

def averCosineSimilatiry(A, B):

    # param A: 表示特征图1:[N, C, H, W]
    # param B: 表示特征图2:[N, C, H, W]
    # return: 返回均值相似度:[N]
    
    N = A.shape[0]  # 表示当前批次中图片数量

    criterion_similarity = nn.CosineSimilarity(dim=1).cuda()

    A = F.adaptive_avg_pool2d(A, [1, 1])  # [N, C, 1, 1]
    B = F.adaptive_avg_pool2d(B, [1, 1])  # [N, C, 1, 1]

    A = A.view(A.shape[0], A.shape[1])  # [N, C]
    B = B.view(B.shape[0], B.shape[1])  # [N, C]

    A = F.normalize(A, dim=1)  # 如果需要使用欧氏距离损失函数,那么将向量进行l2标准化
    B = F.normalize(B, dim=1)
    
    similarity = criterion_similarity(A, B)

    return similarity
