import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import timm
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd # 统一使用 pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 动态数据增强包装类
class AugmentedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, label
        
    def __len__(self):
        return len(self.subset)

# 2. 混淆矩阵绘制函数
def plot_and_save_cm(y_true, y_pred, species_list, model_label):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=species_list, yticklabels=species_list,
                cbar_kws={'label': 'Number of images'})
    plt.title(f'Confusion Matrix: {model_label} (15 Epochs)', fontsize=14, pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig("cm_swin_15epochs.png")
    plt.close()
    print("✅ 混淆矩阵已保存为: cm_swin_15epochs.png")

# 3. 主程序
def main():
    # 自动识别设备
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🌟 当前运行设备: {device}")
        
    dataset_dir = "./data"
    if not os.path.exists(dataset_dir):
        print(f"❌ 错误：找不到 '{dataset_dir}' 文件夹！")
        return

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(degrees=(-15, 15))], p=0.7),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 数据集加载
    base_dataset = datasets.ImageFolder(root=dataset_dir)
    species_list = base_dataset.classes
    num_classes = len(species_list)
    
    train_size = int(0.8 * len(base_dataset))
    test_size = len(base_dataset) - train_size
    train_split, test_split = random_split(base_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(AugmentedDataset(train_split, train_transform), batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(AugmentedDataset(test_split, test_transform), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    
    # 模型加载
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(15):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}/15 完成")
        
    # 评估与保存
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, preds = torch.max(model(images), 1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            
    plot_and_save_cm(y_true, y_pred, species_list, "Swin Transformer")
    
    # ✅ 重点：保存权重
    torch.save(model.state_dict(), "swin_final.pth")
    print("💾 模型权重已保存至: swin_final.pth")
    print("🚀 实验流程闭环完成！")

if __name__ == "__main__":
    main()