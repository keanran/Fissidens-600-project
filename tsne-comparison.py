import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.manifold import TSNE
import warnings

# 忽略底层线性代数库可能抛出的除零与数值溢出运行时警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 确保中文字体在 Matplotlib 中正常显示（兼容 Windows 与 Mac）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 核心：低时延、无害化特征提取函数
# ==========================================
@torch.no_grad()
def extract_features(model, dataloader, device, model_type='cnn'):
    model.eval()
    features_list = []
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        if model_type == 'cnn':
            # ResNet50: 提取 AvgPool 层的输出并展平
            outputs = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(imgs)))))))))
            outputs = torch.flatten(outputs, 1)
        elif model_type == 'swin':
            # Swin Transformer (timm 库): 显式处理 3 维张量流
            outputs = model.forward_features(imgs)
            # 如果输出是 [Batch_Size, Sequence_Length, Features] (例如 [32, 49, 1024])
            if len(outputs.shape) == 3:
                outputs = outputs.mean(dim=1)  # 对序列维度取平均，降到 [Batch_Size, Features]
            # 终极保险：强行展平可能残留的任何一维空间冗余，确保形态严格为 2 维
            outputs = outputs.view(outputs.size(0), -1)
        elif model_type == 'bioclip':
            # BioCLIP 2.5: 直接调用图像编码器
            outputs = model.encode_image(imgs)
            # 保险：确保 CLIP 传回的也是标准的 2 维特征
            outputs = outputs.view(outputs.size(0), -1)
            
        features_list.append(outputs.cpu().numpy())
        
    # 拼接所有批次的数据
    final_embeddings = np.concatenate(features_list, axis=0)
    return final_embeddings


# ==========================================
# 主程序入口（Mac 多进程环境必须加上这把保护伞）
# ==========================================
if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    data_dir = './data'  # 对应你的 data 文件夹
    img_size = 224       # 统一降采样为 224 以适配 Swin Base 的输入规范

    # 归一化标准：ResNet 和 Swin 均适用标准 ImageNet 归一化
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 自动读取子文件夹名称作为类别标签
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    # Mac环境安全起见将 num_workers 设为 0，防止多进程冲突
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    class_names = dataset.classes
    labels_all = np.array(dataset.targets)

    embeddings_dict = {}

    # ---- 模型 ①: 未训练的 ResNet50 ----
    print("--> 正在提取: 未训练的 ResNet50 特征...")
    resnet_untrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    embeddings_dict['未训练 ResNet50'] = extract_features(resnet_untrained, dataloader, device, 'cnn')

    # ---- 模型 ②: 微调后的 ResNet50 ----
    print("--> 正在提取: 微调后的 ResNet50 特征...")
    resnet_tuned = models.resnet50()
    num_ftrs = resnet_tuned.fc.in_features
    resnet_tuned.fc = nn.Linear(num_ftrs, len(class_names))
    resnet_tuned.load_state_dict(torch.load('resnet50_final.pth', map_location=device))
    resnet_tuned = resnet_tuned.to(device)
    embeddings_dict['微调后 ResNet50'] = extract_features(resnet_tuned, dataloader, device, 'cnn')

    # ---- 模型 ③: 微调后的 Swin Transformer（已适配您的 timm 训练配置） ----
    print("--> 正在提取: 微调后的 Swin Transformer 特征...")
    try:
        import timm
        # 1. 严格使用您当初训练时一模一样的 timm 初始化底座
        swin_tuned = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=len(class_names))
        # 2. 完美无缝加载您的本地权重
        swin_tuned.load_state_dict(torch.load('swin_final.pth', map_location=device))
        swin_tuned = swin_tuned.to(device)
        # 3. 提取特征
        embeddings_dict['微调后 Swin'] = extract_features(swin_tuned, dataloader, device, 'swin')
    except ImportError:
        print("❌ 错误: 未检测到 timm 库，微调后 Swin 将被跳过！请在终端运行: pip install timm")

    # ---- 模型 ④: BioCLIP 2.5 ----
    print("--> 正在提取: BioCLIP 2.5 特征...")
    try:
        import open_clip
        try:
            # 策略 A：尝试通过 Hugging Face 社区标准路径加载 (最稳妥)
            print("尝试通过 HF 托管路径加载 BioCLIP...")
            bioclip_model, _, _ = open_clip.create_model_and_transforms(
                'hf-hub:imageomics/bioclip', 
                pretrained=''
            )
        except Exception:
            # 策略 B：如果失败，退回到最初的官方标签尝试
            print("HF 路径失败，尝试使用默认标签加载...")
            bioclip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='bioclip')
            
        bioclip_model = bioclip_model.to(device)
        embeddings_dict['BioCLIP 2.5'] = extract_features(bioclip_model, dataloader, device, 'bioclip')
    except Exception as e:
        print(f"提示: BioCLIP 2.5 加载失败 ({e})，本次画图将跳过该模型。")

    # ==========================================
    # 4. 运行 t-SNE 降维算法并绘制 4 联图
    # ==========================================
    print("--> 所有模型特征提取完毕！正在启动 t-SNE 空间降维...")

    num_plots = len(embeddings_dict)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5.5))
    if num_plots == 1: axes = [axes]

    # 定义 5 种凤尾藓对应的专属学术多色系
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, (model_name, emb) in enumerate(embeddings_dict.items()):
        print(f"正在对 {model_name} 的特征空间进行 2D 流形降维...")
        
        # 打印调试日志，确保传入 t-SNE 的矩阵是 2 维的
        print(f"调试信息 -> {model_name} 的特征矩阵最终形状为: {emb.shape}")
        
        # 运行 t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        emb_2d = tsne.fit_transform(emb)
        
        ax = axes[idx]
        for class_idx, class_nm in enumerate(class_names):
            mask = (labels_all == class_idx)
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                       c=colors[class_idx], label=class_nm, 
                       alpha=0.7, edgecolors='none', s=25)
        
        ax.set_title(model_name, fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 统一将图例摆放在整张大图的正下方
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=12, frameon=True)
    plt.tight_layout()

    # 保存高清大图
    output_path = 'fissidens_tsne_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✨ 成功！高清特征对比散点图已安全保存至: {output_path}")
    plt.show()