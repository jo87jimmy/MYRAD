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
    learning_rate = 0.005  # 學習率
    # batch_size = 16 # 批次大小
    image_size = 256  # 輸入影像大小

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 選擇裝置
    print(f"🖥️ 使用裝置: {device}")

    # # 資料轉換
    # data_transform, gt_transform = get_data_transforms(image_size, image_size)
    # train_path = f'./mvtec/{_class_}/train'  # 訓練資料路徑
    # test_path = f'./mvtec/{_class_}'  # 測試資料路徑

    # # 載入訓練與測試資料
    # train_data = ImageFolder(root=train_path, transform=data_transform)
    # test_data = MVTecDataset(root=test_path,
    #                          transform=data_transform,
    #                          gt_transform=gt_transform,
    #                          phase="test")

    # # 建立 DataLoader
    # train_dataloader = torch.utils.data.DataLoader(train_data,
    #                                                batch_size=batch_size,
    #                                                shuffle=True)
    # test_dataloader = torch.utils.data.DataLoader(test_data,
    #                                               batch_size=1,
    #                                               shuffle=False)

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
    # test_path = f'./mvtec/{_class_}'  # 測試資料路徑
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
    n_iter = 0
    for epoch in range(args.epochs):
        student_model.train()  # 確保學生模型處於訓練模式
        student_model_seg.train()

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

            if i_batch % 100 == 0:  # 每 100 個 batch 打印一次 log
                print(
                    f"  Batch {i_batch}/{len(dataloader)}, Total Loss: {loss.item():.4f}, "
                    f"Hard Loss: {loss_hard.item():.4f}, Distill Loss: {loss_distill.item():.4f}"
                )
            n_iter += 1

        scheduler.step()

    # === Step 7: 儲存訓練好的學生模型 ===
    print("Step 7: Saving the trained student model...")
    # 為學生模型設定一個新的儲存名稱
    student_run_name = f"{_arch_}_student_{_class_}"
    torch.save(student_model.state_dict(),
               os.path.join(save_pth_path, student_run_name + ".pckl"))
    torch.save(student_model_seg.state_dict(),
               os.path.join(save_pth_path, student_run_name + "_seg.pckl"))

    print(f"✅ 模型已訓練並儲存至 {save_pth_path}")

    # 建立輸出資料夾
    # save_pth_dir = save_pth_path if save_pth_path else 'pths/best'
    # os.makedirs(save_pth_dir, exist_ok=True)

    # # 設定最佳權重檔案存放路徑
    # best_ckp_path = os.path.join(save_pth_dir, f'best_{_arch_}_{_class_}.pth')

    # # 初始化最佳分數
    # best_score = -1

    # # 訓練迴圈
    # for epoch in range(epochs):
    #     student_encoder.train()
    #     student_decoder.train()
    #     loss_list = []

    #     for img, label in train_dataloader:
    #         img = img.to(device)

    #         # 教師模型推理
    #         with torch.no_grad():
    #             teacher_recon = teacher_encoder(img)
    #             teacher_input = torch.cat([img, teacher_recon], dim=1)
    #             teacher_seg = teacher_decoder(teacher_input)

    #         # 學生模型推理
    #         student_recon = student_encoder(img)
    #         student_input = torch.cat([img, student_recon], dim=1)
    #         student_seg = student_decoder(student_input)

    #         # 蒸餾損失：比較相同語義的輸出
    #         recon_loss = distillation_loss(teacher_recon, student_recon)
    #         seg_loss = distillation_loss(teacher_seg, student_seg)

    #         total_loss = recon_loss + seg_loss

    #         optimizer.zero_grad()
    #         total_loss.backward()  # 修正：使用 total_loss
    #         optimizer.step()
    #         loss_list.append(total_loss.item())

    #     print(
    #         f"📘 Epoch [{epoch + 1}/{epochs}] | Loss: {np.mean(loss_list):.4f}")

    #     # 每個 epoch 都進行一次評估（使用學生模型）
    #     # 需要添加 bn 層
    #     #torch.nn.Identity() 作為一個恆等映射層，不會改變輸入數據，只是為了滿足 evaluation 函數的參數要求
    #     #由於 torch.nn.Identity() 不改變輸入，所以 bn(inputs) 等同於直接傳遞 inputs，這樣就能讓您的 DREAM 架構正常工作。
    #     # bn = torch.nn.Identity()  # 或者使用適當的 batch normalization 層
    #     auroc_px, auroc_sp, aupro_px = dream_evaluation(student_encoder,
    #                                             #   bn,
    #                                               student_decoder,
    #                                               test_dataloader, device)
    #     # auroc_px, auroc_sp, aupro_px = evaluation(student_encoder,
    #     #                                           student_decoder,
    #     #                                           test_dataloader, device)
    #     print(f"🔍 評估 | Pixel AUROC: {auroc_px:.3f}")

    #     # 如果表現更好則儲存學生模型
    #     if auroc_px > best_score:
    #         best_score = auroc_px
    #         torch.save(
    #             {
    #                 'encoder': student_encoder.state_dict(),
    #                 'decoder': student_decoder.state_dict()
    #             }, best_ckp_path)
    #         print(f"💾 更新最佳模型 → {best_ckp_path}")

    # # 訓練結束回傳最佳結果
    # return best_ckp_path, best_score, auroc_sp, aupro_px, student_encoder, student_decoder


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
    # save_visual_path = f"results/{args.arch}_{args.category}"
    save_pth_path = f"pths/best_{args.arch}_{args.category}"

    # 建立輸出資料夾
    save_pth_dir = save_pth_path if save_pth_path else 'pths/best'
    os.makedirs(save_pth_dir, exist_ok=True)

    # 開始訓練，並接收最佳模型路徑與結果
    train(args.arch, args.category, args.epochs, save_pth_path)
    # best_ckp, auroc_px, auroc_sp, aupro_px, bn, decoder = train(
    #     args.arch, args.category, args.epochs, save_pth_path)

    # print(f"最佳模型: {best_ckp}")

    # # 存訓練指標到 CSV
    # df_metrics = pd.DataFrame([{
    #     'Category': args.category,
    #     'Pixel_AUROC': auroc_px,
    #     'Sample_AUROC': auroc_sp,
    #     'Pixel_AUPRO': aupro_px,
    #     'Epochs': args.epochs
    # }])
    # metrics_name = f"metrics_{args.arch}_{args.category}.csv"
    # df_metrics.to_csv(metrics_name,
    #                   mode='a',
    #                   header=not os.path.exists(metrics_name),
    #                   index=False)

    # # 🔥 訓練結束後自動產生可視化結果
    # visualizationDraem(args.arch,
    #               args.category,
    #               ckp_path=best_ckp,
    #               save_path=save_visual_path)
