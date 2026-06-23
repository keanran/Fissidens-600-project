import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import timm
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 1. 动态数据增强包装类（完美对齐你的生态学增强方案设计）
# =====================================================================
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

# =====================================================================
# 2. 混淆矩阵热力图绘制函数（高级学术苔藓绿）
# =====================================================================
def plot_and_save_cm(y_true, y_pred, species_list, model_label):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7), dpi=300) # 300 DPI 高清期刊印刷级别
    
    # 🎨 替换为 'Greens' 绿色渐变配色
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=species_list, yticklabels=species_list,
                cbar_kws={'label': 'Number of images'})
    
    plt.title(f'Confusion Matrix: {model_label} (15 Epochs Fine-tuned)', fontsize=14, pad=15, fontweight='bold')
    plt.ylabel('True Species (Ground Truth)', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Species', fontsize=12, labelpad=10)
    plt.xticks(rotation=35, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = "cm_resnet50_15epochs.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 绿意盎然！全新的混淆矩阵图已成功保存为：'{save_path}'")

# =====================================================================
# 3. 主训练与评估流程
# =====================================================================
def main():
    # 🚀 激活 Mac 芯片图形硬件加速 (MPS)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"🌟 硬件加速通道开启，当前运行设备：{device}")
        
    dataset_dir = "./data"
    if not os.path.exists(dataset_dir):
        print(f"❌ 错误：在当前目录下找不到 '{dataset_dir}' 文件夹，请核对。")
        return

    # 🎨 【训练集增强流】严格对齐你的概率和参数设计
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224), 
        transforms.RandomHorizontalFlip(p=0.5), # 50% 概率水平翻转
        transforms.RandomVerticalFlip(p=0.5),   # 50% 概率垂直翻转
        transforms.RandomApply([
            transforms.RandomRotation(degrees=(-15, 15)) # 70% 概率触发 [-15°, 15°] 小角度旋转
        ], p=0.7),
        transforms.RandomApply([
            # 80% 概率触发 [0.7, 1.3] 随机亮度和对比度调节
            transforms.ColorJitter(brightness=0.3, contrast=0.3) 
        ], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ⚠️ 【测试集流】保持干净，不做花哨增强，确保期末考试公平
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 📂 严格进行 8:2 随机划分，锁死随机种子 Seed 42，保证试卷和大模型完全一致
    base_dataset = datasets.ImageFolder(root=dataset_dir)
    species_list = base_dataset.classes
    num_classes = len(species_list)
    print(f"🌿 成功载入凤尾藓数据集，共 {num_classes} 个物种: {species_list}")
    
    train_size = int(0.8 * len(base_dataset))
    test_size = len(base_dataset) - train_size
    train_split, test_split = random_split(base_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_dataset = AugmentedDataset(train_split, train_transform)
    test_dataset = AugmentedDataset(test_split, test_transform)
    
    # 🏎️ 换装极限加速的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"📊 数据划分完成 -> 训练集: {len(train_dataset)} 张 | 测试集: {len(test_dataset)} 张")
    print("-" * 75)

    # 🔄 从 timm 加载工业级 ResNet50 并自动将最后一层魔改为 5 分类
    print("🔄 正在加载预训练 ResNet50 并配置分类层...")
    model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # 📌 引入余弦退火学习率退火器，T_max 严格对齐 15 轮
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    # 🎬 锁定 15 轮训练
    epochs = 15
    print(f"🎬 开始 {epochs} 轮有监督微调训练，模型正在努力认领形态学特征...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        # 📌 每轮结束时执行学习率衰减，帮助模型在后期精细调整
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
    # 📝 考卷封闭评估
    model.eval()
    y_true, y_pred_top1 = [], []
    top3_correct, total = 0, 0
    
    print("\n📝 15轮微调结束，正在对测试集进行期末考试评估...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, top3_preds = torch.topk(outputs, k=3, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred_top1.extend(top3_preds[:, 0].cpu().tolist())
            
            for i in range(labels.size(0)):
                if labels[i] in top3_preds[i]:
                    top3_correct += 1
                total += 1
                
    # 📊 指标清算
    top1_acc = sum(1 for gt, pr in zip(y_true, y_pred_top1) if gt == pr) / total
    top3_acc = top3_correct / total
    macro_f1 = f1_score(y_true, y_pred_top1, average='macro')
    
    # 🎨 自动画图
    plot_and_save_cm(y_true, y_pred_top1, species_list, "ResNet50")
    
    # ... (前面的评估代码和打印战报保持不变)

    # 🏆 打印单项终极战报
    print("\n" + "="*15 + " 📈 ResNet50 (15强力轮) 最终战报 " + "="*15)
    report_data = {
        "Model": ["ResNet50 (15 Epochs + Aug)"],
        "Top-1 Acc": [f"{top1_acc:.2%}"],
        "Top-3 Acc": [f"{top3_acc:.2%}"],
        "Macro F1-score": [f"{macro_f1:.2%}"]
    }
    print(pd.DataFrame(report_data).to_string(index=False))
    print("=" * 60)
    
    # ✅ 在 main 函数末尾添加这一行，保存 ResNet50 权重
    torch.save(model.state_dict(), "resnet50_final.pth")
    print("💾 ResNet50 模型权重已保存至: resnet50_final.pth")

if __name__ == "__main__":
    main()
    