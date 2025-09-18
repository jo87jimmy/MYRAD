import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from PIL import Image


class MVTecDRAEMValidationDataset(Dataset):
    """
    用於 MVTec AD 資料集驗證階段的 Dataset。
    載入測試集的正常圖片和異常圖片，並生成對應的 ground truth 遮罩。
    """

    def __init__(
            self,
            test_data_path,  # 例如: './mvtec/hazelnut/test'
            resize_shape=None):
        super().__init__()
        self.test_data_path = test_data_path
        self.resize_shape = resize_shape

        # 圖片轉換：調整大小、轉為 Tensor、正規化
        self.image_transform = transforms.Compose([
            transforms.Resize(self.resize_shape),
            transforms.ToTensor(),
            # 使用 ImageNet 的均值和標準差進行正規化，請確保與訓練集一致
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # 遮罩轉換：調整大小、轉為 Tensor、二值化
        self.mask_transform = transforms.Compose(
            [transforms.Resize(self.resize_shape),
             transforms.ToTensor()])

        self.image_paths = []  # 儲存所有圖片的路徑
        self.anomaly_masks = []  # 儲存異常遮罩的路徑 (正常圖片為 None)
        self.is_anomaly = []  # 標記該樣本是否為異常

        # 1. 載入正常測試圖片
        good_image_dir = os.path.join(test_data_path, "good")
        if os.path.exists(good_image_dir):
            for img_name in sorted(os.listdir(good_image_dir)):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(
                        os.path.join(good_image_dir, img_name))
                    self.anomaly_masks.append(None)  # 正常圖片沒有異常遮罩
                    self.is_anomaly.append(False)
        else:
            print(f"警告: 在 {good_image_dir} 找不到 'good' 目錄。")

        # 2. 載入異常測試圖片及其 ground truth 遮罩
        # 假設 ground_truth 目錄與 test_data_path 的父目錄平行
        gt_base_path = os.path.join(os.path.dirname(test_data_path),
                                    "ground_truth")

        for anomaly_type_dir in sorted(os.listdir(test_data_path)):
            # 遍歷 'test' 目錄下的所有子目錄 (除了 'good' 之外，這些是異常類別)
            if anomaly_type_dir == "good":
                continue  # 已處理正常圖片

            anomaly_type_path = os.path.join(test_data_path, anomaly_type_dir)
            if os.path.isdir(anomaly_type_path):
                # 檢查是否有對應的 ground_truth 目錄
                gt_mask_dir = os.path.join(gt_base_path, anomaly_type_dir)
                if not os.path.exists(gt_mask_dir):
                    print(
                        f"警告: 在 {gt_mask_dir} 找不到 {anomaly_type_dir} 的 Ground truth 目錄。"
                    )
                    continue

                for img_name in sorted(os.listdir(anomaly_type_path)):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        base_name = os.path.splitext(img_name)[0]
                        # MVTec 遮罩命名慣例通常是 '_mask.png'
                        mask_name = base_name + '_mask.png'
                        mask_path = os.path.join(gt_mask_dir, mask_name)

                        if os.path.exists(mask_path):
                            self.image_paths.append(
                                os.path.join(anomaly_type_path, img_name))
                            self.anomaly_masks.append(mask_path)
                            self.is_anomaly.append(True)
                        else:
                            print(f"警告: 在 {mask_path} 找不到圖片 {img_name} 的遮罩。")

        print(
            f"已為類別 '{os.path.basename(os.path.dirname(test_data_path))}' 載入 {len(self.image_paths)} 個驗證樣本。"
        )
        print(f" - 正常圖片: {self.is_anomaly.count(False)} 張")
        print(f" - 異常圖片: {self.is_anomaly.count(True)} 張")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        is_anomalous = self.is_anomaly[idx]

        # 載入圖片並轉換為 RGB
        image = Image.open(img_path).convert('RGB')

        # 'image' (對應 'gray_batch'): 經過轉換的原始圖片，作為重建目標
        original_image_tensor = self.image_transform(image)

        # 'augmented_image' (對應 'aug_gray_batch'):
        # 驗證階段通常不需要額外的數據增強，可以直接使用處理後的原始圖片。
        # 這樣，模型會直接接收測試圖片作為輸入。
        augmented_image_tensor = original_image_tensor

        # 載入或生成異常遮罩
        if is_anomalous:
            mask_path = self.anomaly_masks[idx]
            anomaly_mask = Image.open(mask_path).convert('L')  # 轉為灰度圖
            anomaly_mask_tensor = self.mask_transform(anomaly_mask)
            # 二值化遮罩 (0 或 1)
            anomaly_mask_tensor[anomaly_mask_tensor > 0.5] = 1
            anomaly_mask_tensor[anomaly_mask_tensor <= 0.5] = 0
        else:
            # 正常圖片的異常遮罩為全零
            anomaly_mask_tensor = torch.zeros(
                (1, self.resize_shape[0], self.resize_shape[1]),
                dtype=torch.float32)

        return {
            "image": original_image_tensor,  # 原始圖片 (用於重建損失的目標)
            "augmented_image":
            augmented_image_tensor,  # 模型輸入 (augmented_image_tensor)
            "anomaly_mask": anomaly_mask_tensor  # 異常遮罩 (用於分割損失的目標)
        }
