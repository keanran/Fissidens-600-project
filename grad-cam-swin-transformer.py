import torch
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np
import math

def swin_reshape_transform(tensor):
    """
    针对 Swin Transformer 的特征图重构函数。
    将一维序列 (B, L, C) 正确还原并排列为 pytorch-grad-cam 所需的二维空间张量 (B, C, H, W)。
    """
    # 如果输出已经是 B, H, W, C 格式
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    
    # 如果输出是 B, L, C 格式
    B, L, C = tensor.shape
    H = W = int(math.sqrt(L)) # 从序列长度 L 反推特征图的 H 和 W
    
    # 调整轴的顺序：B, L, C -> B, H, W, C -> B, C, H, W
    result = tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return result

def preprocess_image_robust(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224)) 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0), np.array(img) / 255.0

def main():
    # 使用 CPU 确保兼容性，避免 Mac MPS 的一些潜在限制
    device = "cpu"
    
    # 初始化模型
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=5)
    weight_path = "swin_final.pth" 
    
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"✅ 已成功加载 Swin 权重: {weight_path}")
    else:
        print(f"⚠️ 错误：未找到 {weight_path}，请检查权重文件路径！")
        return
        
    model = model.to(device).eval()
    
    # 设置路径
    folder_path = "./grad-cam"
    output_dir = "./grad-cam/results_swin_final"
    os.makedirs(output_dir, exist_ok=True)
    
    # 【Target Layer 选择说明】：
    # 还原空间排列后，建议直接选择最后一层的特征（如下方的 model.layers[-1]...）。
    # 这样获得的语义最强，且几乎没有网格线条。
    target_layers = [model.layers[-1].blocks[-1].norm1]
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"🔍 准备生成 Swin 修正后的热力图，共 {len(files)} 张图片。")
    
    for img_name in files:
        img_path = os.path.join(folder_path, img_name)
        input_tensor, original_img = preprocess_image_robust(img_path)
        
        # 初始化 GradCAM，必须传入 reshape_transform 修正参数
        cam = GradCAM(
            model=model, 
            target_layers=target_layers, 
            reshape_transform=swin_reshape_transform
        )
        
        # 生成热力图
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        visualization = show_cam_on_image(original_img, grayscale_cam[0, :], use_rgb=True)
        
        # 尺寸恢复与平滑缩放
        vis_resized = Image.fromarray(visualization).resize((512, 512), Image.Resampling.BICUBIC)
        
        save_path = os.path.join(output_dir, f"swin_{img_name}")
        vis_resized.save(save_path, quality=95)
        print(f"✅ 已保存正确热力图: {save_path}")

if __name__ == "__main__":
    main()