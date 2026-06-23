import torch
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np

def preprocess_image_robust(img_path):
    # 使用 PIL 读取并强制转换为 RGB，避免通道数报错
    img = Image.open(img_path).convert('RGB')
    # 统一调整至模型输入大小
    img = img.resize((224, 224)) 
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 返回 Tensor 和用于热力图叠加的归一化 numpy 数组
    return transform(img).unsqueeze(0), np.array(img) / 255.0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 初始化模型 (与你训练时的结构保持一致)
    model = timm.create_model("resnet50.a1_in1k", pretrained=False, num_classes=5)
    
    # 【请在此处确认你的权重文件名】
    weight_path = "resnet50_final.pth" 
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"✅ 已加载权重: {weight_path}")
    else:
        print(f"⚠️ 未找到 {weight_path}，当前模型权重为随机初始化。")
        
    model = model.to(device)
    model.eval()
    
    # 2. 设置路径
    folder_path = "./grad-cam"
    output_dir = "./grad-cam/results"
    os.makedirs(output_dir, exist_ok=True)
        
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"🔍 准备生成热力图，共 {len(files)} 张图片。")
    
    # 3. 批量处理
    for img_name in files:
        img_path = os.path.join(folder_path, img_name)
        input_tensor, original_img = preprocess_image_robust(img_path)
        input_tensor = input_tensor.to(device)
        
        # 使用最后一层卷积层
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # 生成热力图
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        visualization = show_cam_on_image(original_img, grayscale_cam[0, :], use_rgb=True)
        
        # 4. 强制放大并以高质量保存
        vis_pil = Image.fromarray(visualization)
        vis_resized = vis_pil.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        save_path = os.path.join(output_dir, f"cam_{img_name}")
        vis_resized.save(save_path, quality=95)
        print(f"✅ 已保存放大版: {save_path}")

if __name__ == "__main__":
    main()