import torch
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np

def preprocess_image_robust(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224)) 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0), np.array(img) / 255.0

def main():
    # 保持 CPU 模式，这是你目前最稳妥的跑通环境
    device = "cpu"
    
    # 初始化 ResNet50
    model = timm.create_model("resnet50.a1_in1k", pretrained=False, num_classes=5)
    weight_path = "resnet50_final.pth" 
    
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"✅ 已加载 ResNet50 权重: {weight_path}")
    else:
        print(f"⚠️ 错误：未找到 {weight_path}，请确认文件名！")
        return
        
    model = model.to(device).eval()
    
    # 设置路径
    folder_path = "./grad-cam"
    output_dir = "./grad-cam/results_resnet50" # 建立专属文件夹
    os.makedirs(output_dir, exist_ok=True)
        
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"🔍 准备生成热力图，共 {len(files)} 张图片。")
    
    # 批量处理
    for img_name in files:
        img_path = os.path.join(folder_path, img_name)
        input_tensor, original_img = preprocess_image_robust(img_path)
        
        # ResNet50 最后一层卷积
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        visualization = show_cam_on_image(original_img, grayscale_cam[0, :], use_rgb=True)
        
        # 强制放大并以高质量保存 (1024x1024)
        vis_resized = Image.fromarray(visualization).resize((1024, 1024), Image.Resampling.LANCZOS)
        
        save_path = os.path.join(output_dir, f"resnet50_{img_name}")
        vis_resized.save(save_path, quality=95)
        print(f"✅ 已保存: {save_path}")

if __name__ == "__main__":
    main()