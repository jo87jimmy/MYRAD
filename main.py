import torch  # 引入 PyTorch
from dataset import get_data_transforms  # 從 dataset.py 載入資料轉換函式
from torchvision.datasets import ImageFolder  # 用於影像資料夾的資料集
import numpy as np  # 數值計算套件
import random  # 亂數控制
import os  # 檔案系統操作
from torch.utils.data import DataLoader  # PyTorch 的資料載入器
from dataset import MVTecDataset  # MVTec 資料集類別
import torch.backends.cudnn as cudnn  # CUDA cuDNN 加速
import argparse  # 命令列參數處理
from test import evaluation, evaluation_draem, visualization, visualizationDraem, dream_evaluation, evaluation2, test  # 測試、評估與可視化函式
from torch.nn import functional as F  # 引入 PyTorch 的函式介面
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork  # 假設你的 DRAEM 定義在 models/draem.py
from student_models import StudentReconstructiveSubNetwork, StudentDiscriminativeSubNetwork
from loss import FocalLoss, SSIM
from data_loader import MVTecDRAEMTrainDataset
from torch import optim


def setup_seed(seed):
    # 設定隨機種子，確保實驗可重現
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保證結果可重現
    torch.backends.cudnn.benchmark = False  # 關閉自動最佳化搜尋


# 蒸餾損失函數
def distillation_loss(teacher_features, student_features):
    cos_loss = torch.nn.CosineSimilarity()
    if not isinstance(teacher_features, (list, tuple)):
        teacher_features, student_features = [teacher_features
                                              ], [student_features]

    loss = 0
    for i in range(len(teacher_features)):
        loss += torch.mean(1 - cos_loss(
            teacher_features[i].view(teacher_features[i].shape[0], -1),
            student_features[i].view(student_features[i].shape[0], -1)))
    return loss


def train(_arch_, _class_, epochs, save_pth_path):
    # 訓練流程
    print(f"🔧 類別: {_class_} | Epochs: {epochs}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 選擇裝置
    print(f"🖥️ 使用裝置: {device}")

    # 教師模型 (已載入權重並設為 eval 模式)
    teacher_model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    teacher_model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    # === Step 2: 載入 checkpoint ===
    teacher_model_ckpt = torch.load(
        "DRAEM_seg_large_ae_large_0.0001_800_bs8_bottle_.pckl",
        map_location=device,
        weights_only=True)
    teacher_model_seg_ckpt = torch.load(
        "DRAEM_seg_large_ae_large_0.0001_800_bs8_bottle__seg.pckl",
        map_location=device,
        weights_only=True)
    teacher_model.load_state_dict(teacher_model_ckpt)
    teacher_model_seg.load_state_dict(teacher_model_seg_ckpt)
    # 重要：載入權重後再移到設備
    teacher_model = teacher_model.to(device)
    teacher_model_seg = teacher_model_seg.to(device)
    teacher_model.eval()
    teacher_model_seg.eval()

    # 學生模型
    student_model = StudentReconstructiveSubNetwork(in_channels=3,
                                                    out_channels=3)
    student_model_seg = StudentDiscriminativeSubNetwork(in_channels=6,
                                                        out_channels=2)
    student_model = student_model.to(device)
    student_model_seg = student_model_seg.to(device)

    # === Step 3: 為學生模型定義優化器和學習率排程器 ===
    optimizer = torch.optim.Adam([{
        "params": student_model.parameters(),
        "lr": args.lr
    }, {
        "params": student_model_seg.parameters(),
        "lr": args.lr
    }])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.epochs * 0.8, args.epochs * 0.9],
        gamma=0.2)

    # optimizer = torch.optim.Adam(list(student_model.parameters()) +
    #                              list(student_model_seg.parameters()),
    #                              lr=learning_rate,
    #                              betas=(0.5, 0.999))

    # === Step 4: 定義所有需要的損失函數和蒸餾超參數 ===
    # 硬損失 (Hard Loss) - 學生與真實標籤比較
    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()

    # 蒸餾損失 (Distillation Loss) - 學生與教師比較
    loss_distill_recon_fn = torch.nn.modules.loss.MSELoss()
    # 重建部分，學生模仿老師的輸出，MSE 或 L1 都可以
    loss_kldiv = torch.nn.KLDivLoss(
        reduction='batchmean')  # 分割部分，用 KL 散度衡量機率分佈

    # 蒸餾超參數
    T = 4.0  # 溫度 (Temperature)，讓教師的輸出更平滑，通常 > 1
    alpha = 0.7  # 蒸餾權重，控制蒸餾損失在總損失中的佔比 (0.7 代表 70%)

    # === Step 5: 準備 Dataset 和 DataLoader ===
    print("Step 5: Preparing Dataset and DataLoader...")

    train_path = f'./mvtec/{_class_}/train'  # 訓練資料路徑
    anomaly_source_path = f'./dtd/images'
    dataset = MVTecDRAEMTrainDataset(train_path + "/good/",
                                     anomaly_source_path,
                                     resize_shape=[256, 256])
    dataloader = DataLoader(dataset,
                            batch_size=args.bs,
                            shuffle=True,
                            num_workers=8)

    # === Step 6: 實現核心訓練迴圈 ===
    print("Step 6: Starting the training loop...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        student_model.train()  # 確保學生模型處於訓練模式
        student_model_seg.train()

        running_loss = 0.0

        print(f"Epoch: {epoch+1}/{args.epochs}")
        for i_batch, sample_batched in enumerate(dataloader):
            # 準備資料
            gray_batch = sample_batched["image"].to(device)
            aug_gray_batch = sample_batched["augmented_image"].to(device)
            anomaly_mask = sample_batched["anomaly_mask"].to(device)

            # --- 教師模型前向傳播 (不計算梯度) ---
            with torch.no_grad():
                teacher_rec = teacher_model(aug_gray_batch)
                teacher_joined_in = torch.cat((teacher_rec, aug_gray_batch),
                                              dim=1)
                teacher_out_mask_logits = teacher_model_seg(teacher_joined_in)

            # --- 學生模型前向傳播 ---
            student_rec = student_model(aug_gray_batch)
            student_joined_in = torch.cat((student_rec, aug_gray_batch), dim=1)
            student_out_mask_logits = student_model_seg(student_joined_in)

            # --- 計算損失 ---
            # 1. 硬損失 (學生 vs. Ground Truth)
            loss_hard_l2 = loss_l2(student_rec, gray_batch)
            loss_hard_ssim = loss_ssim(student_rec, gray_batch)
            student_out_mask_sm = F.softmax(student_out_mask_logits, dim=1)
            loss_hard_segment = loss_focal(student_out_mask_sm, anomaly_mask)
            loss_hard = loss_hard_l2 + loss_hard_ssim + loss_hard_segment

            # 2. 蒸餾損失 (學生 vs. 教師)
            #   a. 重建蒸餾損失
            loss_distill_recon = loss_distill_recon_fn(student_rec,
                                                       teacher_rec)
            #   b. 分割蒸餾損失 (KL 散度)
            p_student = F.log_softmax(student_out_mask_logits / T, dim=1)
            p_teacher = F.softmax(teacher_out_mask_logits / T, dim=1)
            loss_distill_segment = loss_kldiv(p_student, p_teacher) * (
                T * T)  # 乘上 T^2 以保持梯度大小
            loss_distill = loss_distill_recon + loss_distill_segment

            # 3. 總損失 (加權組合)
            loss = (1 - alpha) * loss_hard + alpha * loss_distill

            # --- 反向傳播與優化 (只更新學生模型) ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # === 修改：累計當前 batch 的損失 ===
            running_loss += loss.item()

            if i_batch % 100 == 0:
                print(
                    f"  Batch {i_batch}/{len(dataloader)}, Total Loss: {loss.item():.4f}, "
                    f"Hard Loss: {loss_hard.item():.4f}, Distill Loss: {loss_distill.item():.4f}"
                )

        # 計算此 epoch 的平均損失
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")

        # 檢查是否為最佳損失，若是則儲存權重
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            student_run_name = f"{_arch_}_student_{_class_}"
            torch.save(student_model.state_dict(),
                       os.path.join(save_pth_path, student_run_name + ".pckl"))
            torch.save(
                student_model_seg.state_dict(),
                os.path.join(save_pth_path, student_run_name + "_seg.pckl"))

            print(f"🎉 找到新的最佳模型！Loss: {best_loss:.4f}。已儲存至 {save_pth_path}")

        scheduler.step()
    print("訓練完成！")


if __name__ == '__main__':
    import argparse
    import pandas as pd
    import os
    import torch

    # 解析命令列參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='bottle', type=str)  # 訓練類別
    parser.add_argument('--epochs', default=25, type=int)  # 訓練回合數
    parser.add_argument('--arch', default='wres50', type=str)  # 模型架構
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    args = parser.parse_args()

    setup_seed(111)  # 固定隨機種子
    save_pth_path = f"pths/best_{args.arch}_{args.category}"

    # 建立輸出資料夾
    save_pth_dir = save_pth_path if save_pth_path else 'pths/best'
    os.makedirs(save_pth_dir, exist_ok=True)

    # 開始訓練，並接收最佳模型路徑與結果
    train(args.arch, args.category, args.epochs, save_pth_path)
