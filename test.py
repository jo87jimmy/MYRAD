import torch  # 引入 PyTorch 深度學習框架
from dataset import get_data_transforms, load_data  # 從 dataset.py 匯入資料增強方法與資料載入函式
from torchvision.datasets import ImageFolder  # 匯入 PyTorch 官方圖片資料集讀取工具
import numpy as np  # 匯入數值計算函式庫 NumPy
from torch.utils.data import DataLoader  # PyTorch 的資料載入器 (batch/迭代器)
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2  # 匯入自定義的 ResNet 模型
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2  # 匯入 ResNet 的反卷積解碼器
from dataset import MVTecDataset  # 匯入 MVTec 資料集定義類別 (瑕疵檢測用)
from torch.nn import functional as F  # 匯入 PyTorch 常用函數 (如激活、卷積、插值等)
from sklearn.metrics import roc_auc_score  # 匯入 sklearn 的 ROC AUC 評估指標
import cv2  # OpenCV，用於影像處理
import matplotlib.pyplot as plt  # 繪圖工具 Matplotlib
from sklearn.metrics import auc  # AUC 計算函式
from skimage import measure  # 影像處理工具 (區域標記、區域屬性)
import pandas as pd  # 資料分析工具 Pandas
from numpy import ndarray  # 匯入 ndarray 型別別名
from statistics import mean  # Python 內建平均數計算
from scipy.ndimage import gaussian_filter  # 高斯濾波器
from sklearn import manifold  # 曼哈頓/流形學習工具 (t-SNE, Isomap 等)
from matplotlib.ticker import NullFormatter  # Matplotlib 格式工具
from scipy.spatial.distance import pdist  # 計算向量之間距離
import matplotlib  # Matplotlib 主套件
import pickle  # Python 內建物件序列化工具


# === 計算異常熱力圖 ===
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':  # 乘法模式
        anomaly_map = np.ones([out_size, out_size])  # 初始化為全 1
    else:  # 加法模式
        anomaly_map = np.zeros([out_size, out_size])  # 初始化為全 0
    a_map_list = []  # 儲存每一層的 anomaly map
    for i in range(len(ft_list)):
        fs = fs_list[i]  # 特徵來源
        ft = ft_list[i]  # 特徵目標

        a_map = 1 - F.cosine_similarity(fs, ft)  # 計算 cosine 相似度並轉成 anomaly 分數
        a_map = torch.unsqueeze(a_map, dim=1)  # 增加一個維度 (batch, channel, h, w)
        a_map = F.interpolate(a_map,
                              size=out_size,
                              mode='bilinear',
                              align_corners=True)  # 上採樣至輸出大小
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()  # 轉 numpy
        a_map_list.append(a_map)  # 保存 anomaly map
        if amap_mode == 'mul':  # 乘法聚合
            anomaly_map *= a_map
        else:  # 加法聚合
            anomaly_map += a_map
    return anomaly_map, a_map_list  # 回傳總體 anomaly map 以及每層的 anomaly map

def cal_anomaly_map2(fs_list, ft_list, out_size=224, amap_mode='mul'):
    """
    計算 anomaly map
    fs_list: student 特徵 tensor 或 list of tensor
    ft_list: target 特徵 tensor 或 list of tensor (原圖或教師重建)
    amap_mode: 'mul' 或 'add' 聚合多層 anomaly map
    """
    if not isinstance(fs_list, list):
        fs_list = [fs_list]
    if not isinstance(ft_list, list):
        ft_list = [ft_list]

    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])

    a_map_list = []

    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]

        # 檢查通道匹配
        if fs.shape[1] != ft.shape[1]:
            raise ValueError(f"fs channels ({fs.shape[1]}) != ft channels ({ft.shape[1]})")

        # 計算 cosine similarity
        a_map = 1 - F.cosine_similarity(fs, ft, dim=1)  # channel 維度

        # 增加 channel 維度
        a_map = torch.unsqueeze(a_map, dim=1)

        # 上採樣至輸出大小
        a_map = F.interpolate(a_map,
                              size=out_size,
                              mode='bilinear',
                              align_corners=True)

        a_map_np = a_map[0, 0, :, :].cpu().detach().numpy()
        a_map_list.append(a_map_np)

        # 聚合
        if amap_mode == 'mul':
            anomaly_map *= a_map_np
        else:
            anomaly_map += a_map_np

    return anomaly_map, a_map_list


# === 疊加 anomaly map 到原圖 ===
def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255  # 正規化後加在原圖上
    cam = cam / np.max(cam)  # 縮放到 0~1
    return np.uint8(255 * cam)  # 回傳 uint8 格式影像


# === 最小-最大正規化 ===
def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


# === 轉換成熱力圖 (colormap) ===
def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray),
                                cv2.COLORMAP_JET)  # OpenCV colormap
    return heatmap

# === 評估函式draem ===
def evaluation_draem(student_encoder, student_decoder, dataloader, device, _class_=None):
    """適合 DRAEM 架構的評估函數"""
    student_encoder.eval()  # 設為推論模式
    student_decoder.eval()
    gt_list_px = []  # pixel-level ground truth
    pr_list_px = []  # pixel-level prediction
    gt_list_sp = []  # image-level ground truth
    pr_list_sp = []  # image-level prediction
    aupro_list = []  # PRO 評估

    with torch.no_grad():
        for img, gt, label, _ in dataloader:  # 從 dataloader 取資料
            img = img.to(device)  # 把圖片送到 GPU/CPU

            # DRAEM 架構的推理流程
            student_recon = student_encoder(img)  # 重建影像
            student_input = torch.cat([img, student_recon], dim=1)  # 串接原圖與重建圖
            student_seg = student_decoder(student_input)  # 異常分割

            # 計算異常圖：使用重建誤差
            anomaly_map = torch.mean((img - student_recon) ** 2, dim=1).squeeze()
            anomaly_map = anomaly_map.cpu().numpy()
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # 高斯濾波平滑

            # 二值化 ground truth
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item() != 0:  # 如果是瑕疵類別
                aupro_list.append(
                    compute_pro(
                        gt.squeeze(0).cpu().numpy().astype(int),
                        anomaly_map[np.newaxis, :, :]))

            # 累積像素級 ground truth 與預測
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            # 累積圖片級 (是否有異常)
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)  # 計算像素級 AUC
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)  # 計算圖片級 AUC
    return auroc_px, auroc_sp, round(np.mean(aupro_list), 3)

# === 評估函式 ===
def evaluation2(student_encoder, student_decoder, dataloader, device, _class_=None, teacher_encoder=None):
    """
    評估學生模型 anomaly detection 表現
    teacher_encoder 可選，用於 anomaly map 計算
    """
    # student_encoder.eval()
    student_decoder.eval()

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)

            # 學生模型重建 & segmentation
            student_recon = student_encoder(img)  # [B,3,H,W]
            student_input = torch.cat([img, student_recon], dim=1)  # [B,6,H,W]
            student_seg = student_decoder(student_input)  # [B,2,H,W]

            # anomaly map 計算
            # 如果有教師模型可用教師重建
            if teacher_encoder is not None:
                teacher_recon = teacher_encoder(img)
                fs = student_recon
                ft = teacher_recon
            else:
                # 單純用學生重建 vs 原圖
                fs = student_recon
                ft = img

            anomaly_map, _ = cal_anomaly_map2(fs, ft, out_size=img.shape[-1], amap_mode='mul')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            # 二值化 ground truth
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            if label.item() != 0:  # 有瑕疵
                aupro_list.append(
                    compute_pro(
                        gt.squeeze(0).cpu().numpy().astype(int),
                        anomaly_map[np.newaxis, :, :])
                )

            # 累積像素級
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())

            # 累積圖片級
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        # 計算 AUROC
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)

    return auroc_px, auroc_sp, round(np.mean(aupro_list), 3)

# === 評估函式 ===
def evaluation(encoder, bn, decoder, dataloader, device, _class_=None):
    bn.eval()  # 設為推論模式
    decoder.eval()
    gt_list_px = []  # pixel-level ground truth
    pr_list_px = []  # pixel-level prediction
    gt_list_sp = []  # image-level ground truth
    pr_list_sp = []  # image-level prediction
    aupro_list = []  # PRO 評估
    with torch.no_grad():
        for img, gt, label, _ in dataloader:  # 從 dataloader 取資料

            img = img.to(device)  # 把圖片送到 GPU/CPU
            inputs = encoder(img)  # 取 encoder 特徵
            outputs = decoder(bn(inputs))  # 解碼器輸出
            anomaly_map, _ = cal_anomaly_map(inputs,
                                             outputs,
                                             img.shape[-1],
                                             amap_mode='a')  # 計算 anomaly map
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # 高斯濾波平滑

            # 二值化 ground truth
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item() != 0:  # 如果是瑕疵類別
                aupro_list.append(
                    compute_pro(
                        gt.squeeze(0).cpu().numpy().astype(int),
                        anomaly_map[np.newaxis, :, :]))
            # 累積像素級 ground truth 與預測
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            # 累積圖片級 (是否有異常)
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)  # 計算像素級 AUC
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)  # 計算圖片級 AUC
    return auroc_px, auroc_sp, round(np.mean(aupro_list), 3)

def dream_evaluation(student_encoder, student_decoder, test_dataloader, device, _class_=None):
    """
    專門為 DREAM 架構設計的評估函數
    處理重建+分割的兩階段流程
    """
    # student_encoder.eval()
    student_decoder.eval()

    gt_list_px = []  # pixel-level ground truth
    pr_list_px = []  # pixel-level prediction
    gt_list_sp = []  # image-level ground truth
    pr_list_sp = []  # image-level prediction
    aupro_list = []  # PRO 評估

    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            img = img.to(device)

            # 第一階段：重建
            reconstructed = student_encoder(img)  # 3通道重建圖像

            # 第二階段：分割（串接原圖和重建圖像）
            concat_input = torch.cat([img, reconstructed], dim=1)  # 6通道輸入
            # 用於蒸餾訓練而評估時仍使用重建比較，則可以移除該變數。
            segmentation = student_decoder(concat_input)  # 分割輸出

            # 計算異常圖：比較原圖和重建圖像
            anomaly_map, _ = cal_anomaly_map([img], [reconstructed],
                                           img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            # 二值化 ground truth
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            if label.item() != 0:  # 如果是瑕疵類別
                aupro_list.append(
                    compute_pro(
                        gt.squeeze(0).cpu().numpy().astype(int),
                        anomaly_map[np.newaxis, :, :]))

            # 累積像素級 ground truth 與預測
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())

            # 累積圖片級 (是否有異常)
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

    auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)

    return auroc_px, auroc_sp, round(np.mean(aupro_list), 3)


# === 測試函式 ===
def test(_class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 判斷是否使用 GPU
    print(device)
    print(_class_)

    data_transform, gt_transform = get_data_transforms(256, 256)  # 資料增強
    test_path = '../mvtec/' + _class_  # 測試資料路徑
    ckp_path = './checkpoints/' + 'rm_1105_wres50_ff_mm_' + _class_ + '.pth'  # 模型 checkpoint 路徑
    test_data = MVTecDataset(root=test_path,
                             transform=data_transform,
                             gt_transform=gt_transform,
                             phase="test")  # 測試資料集
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=1,
                                                  shuffle=False)  # 測試資料 loader
    encoder, bn = wide_resnet50_2(pretrained=True)  # 建立編碼器
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)  # 建立解碼器
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)  # 載入 checkpoint
    for k, v in list(ckp['bn'].items()):  # 移除 batchnorm 的 memory 欄位
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder,
                                              test_dataloader, device,
                                              _class_)  # 執行評估
    print(_class_, ':', auroc_px, ',', auroc_sp, ',', aupro_px)
    return auroc_px


import os  # 載入作業系統模組，用於檔案路徑處理、建立資料夾


def visualizationDraem(_arch_, _class_, save_path=None, ckp_path=None):
    print(f"🖼️ 開始可視化類別: {_class_}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = f'./mvtec/{_class_}'

    if ckp_path is None:
        ckp_path = f'./checkpoints/{_arch_}_{_class_}.pth'

    # 建立測試資料集
    test_data = MVTecDataset(root=test_path,
                             transform=data_transform,
                             gt_transform=gt_transform,
                             phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=1,
                                                  shuffle=False)

    # ✅ 修改：使用 DRAEM 的 ReconstructiveSubNetwork 和 DiscriminativeSubNetwork
    from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork

    student_encoder = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    student_decoder = DiscriminativeSubNetwork(in_channels=6, out_channels=2)

    student_encoder = student_encoder.to(device)
    student_decoder = student_decoder.to(device)
    student_encoder.eval()
    student_decoder.eval()

    # ✅ 修改：載入 DRAEM 格式的檢查點
    ckp = torch.load(ckp_path, map_location=device)
    student_encoder.load_state_dict(ckp['encoder'])
    student_decoder.load_state_dict(ckp['decoder'])

    # 建立輸出資料夾
    save_dir = save_path if save_path else f'results/{_class_}'
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            if label.item() == 0:  # 跳過正常樣本
                continue

            img = img.to(device)

            # ✅ 修改：使用 DRAEM 的推理流程
            student_recon = student_encoder(img)  # 重建影像
            student_input = torch.cat([img, student_recon], dim=1)  # 串接原圖與重建圖
            student_seg = student_decoder(student_input)  # 異常分割

            # 從分割結果提取異常圖
            anomaly_map = student_seg[:, 1:, :, :].cpu().numpy()[0, 0]  # 取異常通道
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map * 255)

            # 處理原圖
            img_np = img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255
            img_np = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_BGR2RGB)
            img_norm = np.uint8(min_max_norm(img_np) * 255)

            overlay = show_cam_on_image(img_norm, ano_map)

            # 儲存結果
            cv2.imwrite(f"{save_dir}/{count:03d}_org.png", img_norm)
            cv2.imwrite(f"{save_dir}/{count:03d}_ad.png", overlay)
            count += 1

    print(f"✅ 可視化完成，共儲存 {count} 張圖片至 {save_dir}")

# =============================
# 函式：可視化模型輸出結果
# =============================
def visualization(_arch_, _class_, save_path=None, ckp_path=None):
    print(f"🖼️ 開始可視化類別: {_class_}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 判斷使用 GPU 或 CPU

    data_transform, gt_transform = get_data_transforms(256,
                                                       256)  # 取得影像與標註的資料轉換方式
    test_path = f'./mvtec/{_class_}'  # 測試資料集路徑，與 train() 一致

    # ✅ 如果沒有外部傳入權重檔路徑，就使用預設值
    if ckp_path is None:
        ckp_path = f'./checkpoints/{_arch_}_{_class_}.pth'

    # 建立測試資料集
    test_data = MVTecDataset(root=test_path,
                             transform=data_transform,
                             gt_transform=gt_transform,
                             phase="test")
    # 建立 DataLoader (一次一張圖片，不打亂順序)
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=1,
                                                  shuffle=False)

    # 建立編碼器與 BatchNorm
    encoder, bn = wide_resnet50_2(
        pretrained=True)  # 使用 Wide ResNet50_2 作為 backbone
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()  # 設定為推理模式

    decoder = de_wide_resnet50_2(pretrained=False)  # 解碼器 (Decoder)
    decoder = decoder.to(device)

    # ✅ 載入已訓練好的模型權重
    ckp = torch.load(ckp_path, map_location=device)
    for k in list(ckp['bn'].keys()):
        if 'memory' in k:  # 移除 batch norm 的 "memory" 欄位，避免載入失敗
            del ckp['bn'][k]
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    # 建立輸出資料夾
    save_dir = save_path if save_path else f'results/{_class_}'
    os.makedirs(save_dir, exist_ok=True)

    count = 0  # 計數器：已處理圖片數
    with torch.no_grad():  # 關閉梯度，節省記憶體
        for img, gt, label, _ in test_dataloader:  # 逐張處理測試資料
            if label.item() == 0:  # 如果是正常樣本，跳過
                continue

            img = img.to(device)
            inputs = encoder(img)  # 編碼影像特徵
            outputs = decoder(bn(inputs))  # 解碼重建影像

            # 計算異常圖 (Anomaly Map)
            anomaly_map, _ = cal_anomaly_map([inputs[-1]], [outputs[-1]],
                                             img.shape[-1],
                                             amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # 高斯平滑
            ano_map = min_max_norm(anomaly_map)  # 正規化到 0~1
            ano_map = cvt2heatmap(ano_map * 255)  # 轉換成熱力圖

            # 將原圖轉為 numpy 格式，並標準化
            img_np = img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255
            img_np = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_BGR2RGB)
            img_norm = np.uint8(min_max_norm(img_np) * 255)

            overlay = show_cam_on_image(img_norm, ano_map)  # 疊加熱力圖

            # 儲存原圖與疊加熱力圖
            cv2.imwrite(f"{save_dir}/{count:03d}_org.png", img_norm)
            cv2.imwrite(f"{save_dir}/{count:03d}_ad.png", overlay)
            count += 1

    print(f"✅ 可視化完成，共儲存 {count} 張圖片至 {save_dir}")


# =============================
# 函式：另一種可視化 (nd)
# =============================
def vis_nd(name, _class_):
    print(name, ':', _class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # 權重檔路徑
    ckp_path = './checkpoints/' + name + '_' + str(_class_) + '.pth'
    train_dataloader, test_dataloader = load_data(name, _class_, batch_size=16)

    # 使用 ResNet18
    encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)

    # 載入權重
    ckp = torch.load(ckp_path)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    decoder.eval()
    bn.eval()

    gt_list_sp = []  # Ground Truth 標籤
    prmax_list_sp = []  # 最大異常分數
    prmean_list_sp = []  # 平均異常分數

    count = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            if img.shape[1] == 1:  # 如果是灰階，轉成 3 通道
                img = img.repeat(1, 3, 1, 1)

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            anomaly_map, amap_list = cal_anomaly_map(inputs,
                                                     outputs,
                                                     img.shape[-1],
                                                     amap_mode='a')

            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map * 255)

            img = cv2.cvtColor(
                img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255,
                cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img) * 255)

            # 儲存原圖
            cv2.imwrite(
                './nd_results/' + name + '_' + str(_class_) + '_' +
                str(count) + '_' + 'org.png', img)

            # 疊加熱力圖
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite(
                './nd_results/' + name + '_' + str(_class_) + '_' +
                str(count) + '_' + 'ad.png', ano_map)

            gt_list_sp.extend(label.cpu().data.numpy())  # 加入 GT
            prmax_list_sp.append(np.max(anomaly_map))  # 加入最大異常值
            prmean_list_sp.append(np.sum(anomaly_map))  # 加入總和

        # 將 GT 轉換為 0=正常, 1=異常
        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        # 正規化異常分數
        ano_score = (prmean_list_sp - np.min(prmean_list_sp)) / (
            np.max(prmean_list_sp) - np.min(prmean_list_sp))

        vis_data = {}
        vis_data['Anomaly Score'] = ano_score
        vis_data['Ground Truth'] = np.array(gt_list_sp)

        # 存成 pkl 檔案
        with open('vis.pkl', 'wb') as f:
            pickle.dump(vis_data, f, pickle.HIGHEST_PROTOCOL)


import numpy as np
import pandas as pd
from numpy import ndarray
from skimage import measure
from sklearn.metrics import auc
from statistics import mean
# =============================
# 函式：計算 PRO 指標 (Pixel-wise Recall Overlap)
# =============================


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> float:
    """計算每個區域重疊率（PRO）與 FPR 在 0~0.3 區間的 AUC"""

    # --- 資料驗證 ---
    assert isinstance(amaps, ndarray), "amaps 必須是 ndarray"
    assert isinstance(masks, ndarray), "masks 必須是 ndarray"
    assert amaps.ndim == 3 and masks.ndim == 3, "amaps 和 masks 必須是三維陣列"
    assert amaps.shape == masks.shape, "amaps 和 masks 的形狀必須一致"
    assert set(np.unique(masks)) <= {0, 1}, "masks 必須是二值 (0 或 1)"
    assert isinstance(num_th, int), "num_th 必須是整數"

    # --- 初始化 ---
    df = pd.DataFrame({
        "pro": pd.Series(dtype="float"),
        "fpr": pd.Series(dtype="float"),
        "threshold": pd.Series(dtype="float")
    })
    min_th, max_th = amaps.min(), amaps.max()
    thresholds = np.linspace(min_th, max_th, num_th)

    # --- 閾值掃描 ---
    for th in thresholds:
        binary_amaps = (amaps > th).astype(np.uint8)  # 閾值二值化

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            labeled_mask = measure.label(mask)  # 標記 mask 區域
            for region in measure.regionprops(labeled_mask):
                coords = region.coords
                tp_pixels = binary_amap[coords[:, 0], coords[:, 1]].sum()
                pros.append(tp_pixels / region.area)  # 區域內 TP 比例

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()  # 偽陽率

        new_row = pd.DataFrame([{
            "pro": mean(pros) if pros else 0,
            "fpr": fpr,
            "threshold": th
        }])

        # ✅ 避免 concat 空或全 NA 的 DataFrame
        if not new_row.isna().all(axis=None) and not new_row.empty:
            df = pd.concat([df, new_row], ignore_index=True)

    # --- FPR 正規化與 AUC 計算 ---
    df = df[df["fpr"] < 0.3]  # 只保留 FPR < 0.3
    if df.empty or df["fpr"].max() == 0:
        return 0.0

    df["fpr"] = df["fpr"] / df["fpr"].max()  # FPR 正規化
    pro_auc = auc(df["fpr"], df["pro"])  # 計算 AUC

    return pro_auc


# =============================
# 函式：異常檢測評估 (AUROC)
# =============================
def detection(encoder, bn, decoder, dataloader, device, _class_):
    bn.load_state_dict(bn.state_dict())
    bn.eval()
    decoder.eval()

    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []

    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            if img.shape[1] == 1:  # 灰階轉 RGB
                img = img.repeat(1, 3, 1, 1)
            label = label.to(device)

            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1],
                                             'acc')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # 高斯平滑

            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))  # 最大值
            prmean_list_sp.append(np.sum(anomaly_map))  # 總和

        # 轉換 GT 為二值 (0=正常, 1=異常)
        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        # 計算 ROC AUC
        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)

    return auroc_sp_max, auroc_sp_mean
