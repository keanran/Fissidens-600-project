import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import pandas as pd
import os

def evaluate_model(model_name, weight_path, num_classes):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 1. 加载模型结构
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. 准备数据 (使用你训练时完全相同的测试集)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 注意：这里需要你确保 dataset_dir 路径正确
    dataset = datasets.ImageFolder(root="./data", transform=transform)
    _, test_split = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))], 
                                 generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_split, batch_size=32, shuffle=False)

    # 3. 评估指标计算
    y_true, y_pred_top1 = [], []
    top3_correct, total = 0, 0
    
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
                
    top1_acc = sum(1 for gt, pr in zip(y_true, y_pred_top1) if gt == pr) / total
    top3_acc = top3_correct / total
    macro_f1 = f1_score(y_true, y_pred_top1, average='macro')
    
    return top1_acc, top3_acc, macro_f1

# 执行评估
if __name__ == "__main__":
    # 评估 Swin
    s1, s3, sf = evaluate_model("swin_base_patch4_window7_224", "swin_final.pth", 5)
    # 评估 ResNet50
    r1, r3, rf = evaluate_model("resnet50.a1_in1k", "resnet50_final.pth", 5)
    
    # 打印最终战报
    df = pd.DataFrame({
        "Model": ["Swin Transformer", "ResNet50"],
        "Top-1 Acc": [f"{s1:.2%}", f"{r1:.2%}"],
        "Top-3 Acc": [f"{s3:.2%}", f"{r3:.2%}"],
        "Macro F1": [f"{sf:.2%}", f"{rf:.2%}"]
    })
    print(df.to_string(index=False))