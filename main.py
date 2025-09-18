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
from data_loader_val import MVTecDRAEMValidationDataset
# 新增熱力圖可視化所需的函式庫
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tensorboard_visualizer import TensorboardVisualizer


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
    # 計算學生模型與教師模型特徵的 Cosine 相似度損失
    cos_loss = torch.nn.CosineSimilarity()  # 初始化 CosineSimilarity
    if not isinstance(teacher_features, (list, tuple)):
        # 如果輸入不是 list 或 tuple，就轉成 list，方便迭代
        teacher_features, student_features = [teacher_features
                                              ], [student_features]

    loss = 0  # 初始化總損失
    for i in range(len(teacher_features)):
        # 將特徵展平，計算每個 batch 的 1 - Cosine 相似度，再取平均
        loss += torch.mean(1 - cos_loss(
            teacher_features[i].view(teacher_features[i].shape[0], -1),
            student_features[i].view(student_features[i].shape[0], -1)))
    return loss  # 回傳總蒸餾損失


def generate_anomaly_map(student_rec,
                         gray_batch,
                         student_out_mask_sm,
                         mode='recon+seg'):
    """
    生成缺陷熱力圖
    mode:
        'recon'       : 使用重建誤差 L2
        'seg'         : 使用分割 softmax 的缺陷通道
        'recon+seg'   : L2 重建誤差與分割概率加權
    """
    if mode == 'recon':
        # L2 重建誤差
        recon_error = torch.mean((student_rec - gray_batch)**2,
                                 dim=1,
                                 keepdim=True)  # [B,1,H,W]，沿通道維度取平均
        anomaly_map = recon_error  # 缺陷熱力圖即重建誤差
    elif mode == 'seg':
        # 使用缺陷分割 softmax 的第 1 通道 (假設 0 是正常, 1 是缺陷)
        anomaly_map = student_out_mask_sm[:, 1:, :, :]
    elif mode == 'recon+seg':
        # 同時考慮重建誤差與分割概率
        recon_error = torch.mean((student_rec - gray_batch)**2,
                                 dim=1,
                                 keepdim=True)  # 計算重建誤差
        seg_prob = student_out_mask_sm[:, 1:, :, :]  # 取缺陷概率
        anomaly_map = recon_error + seg_prob  # 簡單加權相加
        anomaly_map = anomaly_map / anomaly_map.max()  # Normalize to [0,1]
    else:
        raise ValueError(f"Unknown mode {mode}")  # 若 mode 不合法，丟出錯誤

    return anomaly_map  # 回傳缺陷熱力圖


def train(_arch_, _class_, epochs, save_pth_path):
    # 訓練流程主函數
    print(f"🔧 類別: {_class_} | Epochs: {epochs}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 選擇運算裝置
    print(f"🖥️ 使用裝置: {device}")

    # 教師模型 (已載入權重並設為 eval 模式)
    teacher_model = ReconstructiveSubNetwork(in_channels=3,
                                             out_channels=3)  # 重建子網路
    teacher_model_seg = DiscriminativeSubNetwork(in_channels=6,
                                                 out_channels=2)  # 分割子網路

    # === Step 2: 載入 checkpoint ===
    teacher_model_ckpt = torch.load(
        "DRAEM_seg_large_ae_large_0.0001_800_bs8_bottle_.pckl",
        map_location=device,
        weights_only=True)  # 載入教師重建模型權重
    teacher_model_seg_ckpt = torch.load(
        "DRAEM_seg_large_ae_large_0.0001_800_bs8_bottle__seg.pckl",
        map_location=device,
        weights_only=True)  # 載入教師分割模型權重

    teacher_model.load_state_dict(teacher_model_ckpt)  # 將權重加載到模型
    teacher_model_seg.load_state_dict(teacher_model_seg_ckpt)

    # 重要：載入權重後再移到設備
    teacher_model = teacher_model.to(device)
    teacher_model_seg = teacher_model_seg.to(device)
    teacher_model.eval()  # 設為評估模式，不更新權重
    teacher_model_seg.eval()

    # 學生模型
    student_dropout_rate = 0.2  # Dropout 率，可調整
    student_model = StudentReconstructiveSubNetwork(
        in_channels=3, out_channels=3,
        dropout_rate=student_dropout_rate)  # 學生重建模型
    student_model_seg = StudentDiscriminativeSubNetwork(
        in_channels=6, out_channels=2,
        dropout_rate=student_dropout_rate)  # 學生分割模型

    student_model = student_model.to(device)  # 移到運算裝置
    student_model_seg = student_model_seg.to(device)

    # === Step 3: 定義學生模型優化器和學習率排程器 ===
    optimizer_weight_decay = 1e-5  # L2 正則化
    optimizer = torch.optim.Adam([
        {
            "params": student_model.parameters(),
            "lr": args.lr,
            "weight_decay": optimizer_weight_decay  # L2 正則化
        },
        {
            "params": student_model_seg.parameters(),
            "lr": args.lr,
            "weight_decay": optimizer_weight_decay  # L2 正則化
        }
    ])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.epochs * 0.8, args.epochs * 0.9],  # 學習率下降節點
        gamma=0.2)  # 每次下降乘以 0.2

    # === Step 4: 定義損失函數和蒸餾超參數 ===
    loss_l2 = torch.nn.modules.loss.MSELoss()  # L2 損失
    loss_ssim = SSIM()  # 結構相似性損失
    loss_focal = FocalLoss()  # Focal Loss，用於分割

    # 蒸餾損失
    loss_distill_recon_fn = torch.nn.modules.loss.MSELoss()  # 重建蒸餾損失
    loss_kldiv = torch.nn.KLDivLoss(reduction='batchmean')  # KL 散度，用於分割蒸餾

    # 蒸餾超參數
    T = 2.0  # 溫度
    alpha = 0.5  # 蒸餾權重

    # === Step 5: 準備 Dataset 和 DataLoader ===
    print("Step 5: Preparing Dataset and DataLoader...")

    train_path = f'./mvtec/{_class_}/train'  # 訓練資料路徑
    anomaly_source_path = f'./dtd/images'  # 缺陷樣本來源
    dataset = MVTecDRAEMTrainDataset(train_path + "/good/",
                                     anomaly_source_path,
                                     resize_shape=[256, 256])  # 訓練集
    dataloader = DataLoader(dataset,
                            batch_size=args.bs,
                            shuffle=True,
                            num_workers=8)  # DataLoader

    # 驗證集
    val_path = f'./mvtec/{_class_}/test'  # 驗證資料路徑
    val_dataset = MVTecDRAEMValidationDataset(val_path,
                                              resize_shape=[256, 256])
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,  # 驗證集不打亂
        num_workers=8)
    print("Validation DataLoader prepared.")

    visualizer_path = f'{_class_}/'
    visualizer = TensorboardVisualizer(log_dir=os.path.join(
        save_pth_path, visualizer_path))  # TensorBoard 可視化

    # === Step 6: 核心訓練迴圈 ===
    print("Step 6: Starting the training loop...")
    best_val_loss = float('inf')  # 初始化最佳驗證損失
    n_iter = 0  # 總迭代計數器

    for epoch in range(args.epochs):
        student_model.train()  # 訓練模式
        student_model_seg.train()

        running_loss = 0.0  # 累計損失

        print(f"Epoch: {epoch+1}/{args.epochs}")
        for i_batch, sample_batched in enumerate(dataloader):
            # 取得 batch 資料，每個 batch 包含原始灰階圖像、增強後圖像，以及異常遮罩
            orig_batch = sample_batched["orig_image"].to(device)  # 原始彩色圖
            gray_batch = sample_batched["image"].to(
                device)  # 原始灰階圖，送到 GPU 或 CPU
            aug_gray_batch = sample_batched["augmented_image"].to(
                device)  # 增強後的灰階圖像
            anomaly_mask = sample_batched["anomaly_mask"].to(device)  # 異常區域遮罩

            # --- 教師模型前向傳播 ---
            with torch.no_grad():  # 教師模型不更新權重，只做推論
                teacher_rec = teacher_model(aug_gray_batch)  # 教師模型對增強後圖像做重建
                # 將教師模型重建結果與增強圖像合併，形成輸入給分割模型
                teacher_joined_in = torch.cat((teacher_rec, aug_gray_batch),
                                              dim=1)
                # 教師模型分割頭輸出異常遮罩的 logits
                teacher_out_mask_logits = teacher_model_seg(teacher_joined_in)
                teacher_out_mask = F.softmax(teacher_out_mask_logits,
                                             dim=1)[:, 1, ...]  # 取異常類別概率

            # --- 學生模型前向傳播 ---
            student_rec = student_model(aug_gray_batch)  # 學生模型對增強後圖像做重建
            # 將學生模型重建結果與增強圖像合併，形成輸入給分割模型
            student_joined_in = torch.cat((student_rec, aug_gray_batch), dim=1)
            # 學生模型分割頭輸出異常遮罩的 logits
            student_out_mask_logits = student_model_seg(student_joined_in)
            student_out_mask = F.softmax(student_out_mask_logits,
                                         dim=1)[:, 1, ...]  # 取異常類別概率
            # --- 可視化 ---
            if i_batch == 0:  # 只顯示第一個 batch，避免顯示太多
                batch_idx = 0  # 顯示 batch 中第一張圖
                # 將各張圖轉成 numpy 格式，C,H,W -> H,W,C
                orig_img = orig_batch[batch_idx].permute(1, 2, 0).cpu().numpy()
                gray_img = gray_batch[batch_idx, 0].cpu().numpy()
                aug_gray_img = aug_gray_batch[batch_idx, 0].cpu().numpy()
                teacher_rec_img = teacher_rec[batch_idx, 0].cpu().numpy()
                student_rec_img = student_rec[batch_idx, 0].cpu().numpy()
                teacher_mask_img = teacher_out_mask[batch_idx].cpu().numpy()
                student_mask_img = student_out_mask[batch_idx].cpu().numpy()
                anomaly_mask_img = anomaly_mask[batch_idx, 0].cpu().numpy()

                # 用 visualizer 儲存
                visualizer.add_image("原始彩色圖", orig_img, epoch)
                visualizer.add_image("原始灰階圖", gray_img, epoch, cmap='gray')
                visualizer.add_image("增強後圖像", aug_gray_img, epoch, cmap='gray')
                visualizer.add_image("教師模型重建",
                                     teacher_rec_img,
                                     epoch,
                                     cmap='gray')
                visualizer.add_image("學生模型重建",
                                     student_rec_img,
                                     epoch,
                                     cmap='gray')
                visualizer.add_image("教師分割結果",
                                     teacher_mask_img,
                                     epoch,
                                     cmap='jet',
                                     alpha=0.7)
                visualizer.add_image("學生分割結果",
                                     student_mask_img,
                                     epoch,
                                     cmap='jet',
                                     alpha=0.7)
                visualizer.add_image("原始異常遮罩",
                                     anomaly_mask_img,
                                     epoch,
                                     cmap='jet',
                                     alpha=0.5)
            # --- 計算損失 ---
            # 1. 硬損失
            loss_hard_l2 = loss_l2(student_rec, gray_batch)  # L2 損失
            loss_hard_ssim = loss_ssim(student_rec, gray_batch)  # SSIM 損失
            student_out_mask_sm = F.softmax(student_out_mask_logits, dim=1)
            loss_hard_segment = loss_focal(student_out_mask_sm,
                                           anomaly_mask)  # 分割損失
            loss_hard = loss_hard_l2 + loss_hard_ssim + loss_hard_segment  # 總硬損失

            # 2. 蒸餾損失
            loss_distill_recon = loss_distill_recon_fn(student_rec,
                                                       teacher_rec)  # 重建蒸餾
            p_student = F.log_softmax(student_out_mask_logits / T, dim=1)
            p_teacher = F.softmax(teacher_out_mask_logits / T, dim=1)
            loss_distill_segment = loss_kldiv(p_student, p_teacher) * (
                T * T)  # 分割蒸餾
            loss_distill = loss_distill_recon + loss_distill_segment  # 總蒸餾損失

            # 3. 總損失
            loss = (1 - alpha) * loss_hard + alpha * loss_distill  # 加權組合

            # --- 反向傳播與優化 ---
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向傳播
            optimizer.step()  # 更新權重

            running_loss += loss.item()  # 累計 batch 損失
            n_iter += 1  # 更新總迭代次數

            if i_batch % 100 == 0:
                print(
                    f"  Batch {i_batch}/{len(dataloader)}, Total Loss: {loss.item():.4f}, "
                    f"Hard Loss: {loss_hard.item():.4f}, Distill Loss: {loss_distill.item():.4f}"
                )

        # 計算 epoch 平均損失
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")

        # === 驗證階段 ===
        student_model.eval()  # 設為評估模式
        student_model_seg.eval()
        val_running_loss = 0.0

        # 選擇一個 batch 用於可視化（確保含缺陷）
        visualize_batch_done = False

        with torch.no_grad():  # 驗證不更新權重
            for i_batch_val, sample_batched_val in enumerate(val_dataloader):
                gray_batch_val = sample_batched_val["image"].to(device)
                aug_gray_batch_val = sample_batched_val["augmented_image"].to(
                    device)
                anomaly_mask_val = sample_batched_val["anomaly_mask"].to(
                    device)

                # 教師前向傳播
                teacher_rec_val = teacher_model(aug_gray_batch_val)
                teacher_joined_in_val = torch.cat(
                    (teacher_rec_val, aug_gray_batch_val), dim=1)
                teacher_out_mask_logits_val = teacher_model_seg(
                    teacher_joined_in_val)

                # 學生前向傳播
                student_rec_val = student_model(aug_gray_batch_val)
                student_joined_in_val = torch.cat(
                    (student_rec_val, aug_gray_batch_val), dim=1)
                student_out_mask_logits_val = student_model_seg(
                    student_joined_in_val)

                # 計算驗證損失
                loss_hard_l2_val = loss_l2(student_rec_val, gray_batch_val)
                loss_hard_ssim_val = loss_ssim(student_rec_val, gray_batch_val)
                student_out_mask_sm_val = F.softmax(
                    student_out_mask_logits_val, dim=1)
                loss_hard_segment_val = loss_focal(student_out_mask_sm_val,
                                                   anomaly_mask_val)
                loss_hard_val = loss_hard_l2_val + loss_hard_ssim_val + loss_hard_segment_val

                loss_distill_recon_val = loss_distill_recon_fn(
                    student_rec_val, teacher_rec_val)
                p_student_val = F.log_softmax(student_out_mask_logits_val / T,
                                              dim=1)
                p_teacher_val = F.softmax(teacher_out_mask_logits_val / T,
                                          dim=1)
                loss_distill_segment_val = loss_kldiv(p_student_val,
                                                      p_teacher_val) * (T * T)
                loss_distill_val = loss_distill_recon_val + loss_distill_segment_val

                val_loss = (1 -
                            alpha) * loss_hard_val + alpha * loss_distill_val
                val_running_loss += val_loss.item()

                # 只生成第一個含缺陷的 batch 的熱力圖，避免畫出全是正常樣本的圖
                if not visualize_batch_done and anomaly_mask_val.sum() > 0:
                    # 生成異常熱力圖（重建誤差 + 分割結果）
                    anomaly_map_val = generate_anomaly_map(
                        student_rec_val,
                        gray_batch_val,
                        student_out_mask_sm_val,
                        mode='recon+seg')
                    # 可視化熱力圖
                    visualizer.visualize_image_batch(
                        anomaly_map_val, n_iter, image_name='val_anomaly_map')

                    # 提取分割結果異常通道
                    t_mask_val = student_out_mask_sm_val[:, 1:, :, :]

                    # 將增強圖送到 TensorBoard
                    visualizer.visualize_image_batch(
                        aug_gray_batch_val,
                        n_iter,
                        image_name='val_batch_augmented')
                    # 將重建目標（原始圖）送到 TensorBoard
                    visualizer.visualize_image_batch(
                        gray_batch_val,
                        n_iter,
                        image_name='val_batch_recon_target')
                    # 將學生模型重建輸出送到 TensorBoard
                    visualizer.visualize_image_batch(
                        student_rec_val,
                        n_iter,
                        image_name='val_batch_recon_out')
                    # 將真實異常掩碼送到 TensorBoard
                    visualizer.visualize_image_batch(
                        anomaly_mask_val, n_iter, image_name='val_mask_target')
                    # 將學生模型分割輸出異常通道送到 TensorBoard
                    visualizer.visualize_image_batch(t_mask_val,
                                                     n_iter,
                                                     image_name='val_mask_out')

                    visualize_batch_done = True  # 標記已生成熱力圖，避免重複

        epoch_val_loss = val_running_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} Average Validation Loss: {epoch_val_loss:.4f}")

        # 檢查是否最佳模型，若是就保存
        if epoch_val_loss < best_val_loss:  # ✅ 用驗證集損失
            best_val_loss = epoch_val_loss
            student_run_name = f"{_arch_}_student_{_class_}"
            torch.save(student_model.state_dict(),
                       os.path.join(save_pth_path, student_run_name + ".pckl"))
            torch.save(
                student_model_seg.state_dict(),
                os.path.join(save_pth_path, student_run_name + "_seg.pckl"))
            print(
                f"🎉 找到新的最佳模型！Validation Loss: {best_val_loss:.4f}。已儲存至 {save_pth_path}"
            )

        scheduler.step()  # 更新學習率
    print("訓練完成！")  # 訓練結束


def generate_anomaly_heatmap(image_path,
                             student_model,
                             student_model_seg,
                             device,
                             resize_shape=[256, 256]):
    """
    使用訓練好的學生模型為給定輸入影像生成異常熱力圖。

    Args:
        image_path (str): 輸入影像的路徑。
        student_model (torch.nn.Module): 訓練好的學生重建子網路。
        student_model_seg (torch.nn.Module): 訓練好的學生判別子網路。
        device (str or torch.device): 執行推論的裝置 ('cuda' 或 'cpu')。
        resize_shape (list): 影像縮放的目標尺寸 (高度, 寬度)。

    Returns:
        tuple: (numpy.ndarray, PIL.Image.Image) 異常熱力圖 (NumPy 陣列) 和原始影像 (已縮放的 PIL.Image 物件)。
    """
    student_model.eval()  # 設定模型為評估模式
    student_model_seg.eval()

    # 定義影像預處理轉換
    # 與 MVTecDRAEMTrainDataset 中的正規化保持一致 (通常是 [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 載入並預處理影像
    img = Image.open(image_path).convert('RGB')
    # 為了可視化，保留一個未正規化但已縮放的原始影像副本
    original_img_resized = img.resize((resize_shape[1], resize_shape[0]))
    img_tensor = transform(img).unsqueeze(0).to(device)  # 新增批次維度

    with torch.no_grad():  # 在推論時不計算梯度
        # 學生重建網路推論
        student_rec = student_model(img_tensor)

        # 準備學生判別網路的輸入 (重建影像 + 原始影像)
        student_joined_in = torch.cat((student_rec, img_tensor), dim=1)

        # 學生判別網路推論
        student_out_mask_logits = student_model_seg(student_joined_in)

        # 應用 Softmax 取得異常機率
        # 判別網路輸出有 2 個通道：[正常機率 logits, 異常機率 logits]
        # 我們取異常類別 (通道 1) 的機率
        anomaly_map = F.softmax(student_out_mask_logits,
                                dim=1)[:, 1, :, :].squeeze(0)

        # 移至 CPU 並轉換為 NumPy 陣列以便可視化
        anomaly_map_np = anomaly_map.cpu().numpy()

    return anomaly_map_np, original_img_resized


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
    # parser.add_argument('--test_image_path',
    #                     type=str,
    #                     help='路徑到一個用於生成熱力圖的測試影像 (訓練後執行)。',
    #                     default=None)  # 新增參數
    # parser.add_argument(
    #     '--student_dropout_rate',
    #     type=float,
    #     default=0.2,
    #     help='學生模型訓練和載入時使用的 Dropout Rate。')  # 將 Dropout Rate 可配置化
    args = parser.parse_args()

    setup_seed(111)  # 固定隨機種子
    save_pth_path = f"pths/best_{args.arch}_{args.category}"

    # 建立輸出資料夾
    save_pth_dir = save_pth_path if save_pth_path else 'pths/best'
    os.makedirs(save_pth_dir, exist_ok=True)

    # 開始訓練，並接收最佳模型路徑與結果
    train(args.arch, args.category, args.epochs, save_pth_path)

    # # --- 缺陷檢測熱力圖生成區塊 ---
    # test_path = f'./mvtec/{args.category}/test'  # 驗證資料路徑 (MVTec AD 的測試集)
    # if test_path:
    #     print("\n--- 正在生成缺陷檢測熱力圖 ---")
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #     # 載入學生模型權重
    #     # 這裡使用的 dropout_rate 應與訓練時使用的保持一致
    #     student_model = StudentReconstructiveSubNetwork(
    #         in_channels=3,
    #         out_channels=3,
    #         dropout_rate=args.student_dropout_rate).to(device)
    #     student_model_seg = StudentDiscriminativeSubNetwork(
    #         in_channels=6,
    #         out_channels=2,
    #         dropout_rate=args.student_dropout_rate).to(device)

    #     student_run_name = f"{args.arch}_student_{args.category}"
    #     student_model_path = os.path.join(save_pth_path,
    #                                       student_run_name + ".pckl")
    #     student_model_seg_path = os.path.join(save_pth_path,
    #                                           student_run_name + "_seg.pckl")

    #     if os.path.exists(student_model_path) and os.path.exists(
    #             student_model_seg_path):
    #         student_model.load_state_dict(
    #             torch.load(student_model_path, map_location=device))
    #         student_model_seg.load_state_dict(
    #             torch.load(student_model_seg_path, map_location=device))
    #         print(
    #             f"✅ 已成功載入學生模型權重: {student_model_path} 及 {student_model_seg_path}"
    #         )

    #         # 生成熱力圖
    #         anomaly_heatmap, original_image_resized = generate_anomaly_heatmap(
    #             test_path + args.test_image_path, student_model,
    #             student_model_seg, device)

    #         # 可視化結果
    #         plt.figure(figsize=(12, 6))

    #         plt.subplot(1, 2, 1)
    #         plt.imshow(original_image_resized)
    #         plt.title("原始影像 (Resized)")
    #         plt.axis('off')

    #         plt.subplot(1, 2, 2)
    #         # 將熱力圖疊加在原始影像上
    #         plt.imshow(original_image_resized, cmap='gray')  # 背景顯示原始影像
    #         plt.imshow(anomaly_heatmap, cmap='jet', alpha=0.5, vmin=0,
    #                    vmax=1)  # 疊加熱力圖
    #         plt.colorbar(label='異常機率')
    #         plt.title("缺陷熱力圖")
    #         plt.axis('off')

    #         plt.tight_layout()
    #         plt.show()

    #         # (可選) 儲存熱力圖到檔案
    #         heatmap_filename = os.path.join(
    #             save_pth_path,
    #             f"heatmap_{os.path.basename(args.test_image_path)}")
    #         plt.savefig(heatmap_filename)
    #         print(f"熱力圖已儲存至: {heatmap_filename}")

    #     else:
    #         print(
    #             f"❌ 找不到學生模型權重，請確認路徑: {student_model_path} 或 {student_model_seg_path} 是否存在。"
    #         )
    # else:
    #     print("\n如需生成熱力圖，請在命令列中指定 '--test_image_path <影像路徑>'。")
