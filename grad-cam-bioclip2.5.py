import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image
import open_clip

# 引入 pytorch-grad-cam 库
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================================
# 1. 初始化 BioCLIP 2.5 模型与配置
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载 BioCLIP 2.5 (Huge / ViT-H/14 或 ViT-L/14)
# 提示: 如果你本地使用的是本地权重，请将 model_path 指向你的 checkpoint 
model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:imageomics/bioclip',  # 如果是2.5对应特定分支，可替换为具体的hf路径或本地路径
    precision='fp32'              # 算梯度建议使用 fp32 保证数值稳定
)
model = model.to(device)
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

# ==========================================
# 2. 定制适用于 BioCLIP ViT 架构的适配器
# ==========================================
# ViT 最后一层的特征是 (Batch, Tokens, Channels) 的一维序列（通常含 Class Token）
# 我们需要去除 Class Token，并将剩余的 Patch Tokens 重塑为 2D 图像网格
def vit_reshape_transform(tensor):
    # tensor 形状通常为: [B, Grid_H*Grid_W + 1, Channels] 
    # 去除第0个位置的 Class Token
    result = tensor[:, 1:, :]
    
    # 计算网格大小（如 ViT-H/14 在 224x224 输入下，网格大小为 14x14 = 196）
    num_patches = result.size(1)
    grid_size = int(np.sqrt(num_patches))
    
    # 重塑为 [B, Grid_H, Grid_W, Channels] -> 转换为 PyTorch 标准 [B, Channels, Grid_H, Grid_W]
    result = result.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

# 包装模型：让前向传播直接输出“图像特征与目标文本的相似度分数”
class BioCLIPTargetWrapper(torch.nn.Module):
    def __init__(self, image_encoder, text_embedding):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_embedding = text_embedding # 事先提取好的 "Fissidens" 文本特征
        
    def forward(self, x):
        image_features = self.image_encoder(x)
        # 归一化特征向量
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # 计算余弦相似度作为 Grad-CAM 的 Target Score
        similarity = image_features @ self.text_embedding.T
        return similarity

# ==========================================
# 3. 准备目标文本（以凤尾藓属 "Fissidens" 为目标）
# ==========================================
target_text = "A photo of Fissidens, a genus of mosses"
text_token = tokenizer([target_text]).to(device)
with torch.no_grad():
    text_embedding = model.encode_text(text_token)
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

# 提取视觉编码器，并指定目标特征层（通常选择最后一层 Transformer Block 结束后的 LayerNorm）
image_encoder = model.visual
target_layers = [image_encoder.transformer.resblocks[-1].ln_1]

# 实例化可求导的包裹模型
wrapped_model = BioCLIPTargetWrapper(image_encoder, text_embedding)

# ==========================================
# 4. 批量处理 grad-cam 文件夹下的图片
# ==========================================
input_dir = "grad-cam"
output_dir = "grad-cam_results"
os.makedirs(output_dir, exist_ok=True)

# 获取文件夹下所有的 jpg 图片
img_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
print(f"Found {len(img_paths)} images to process.")

# 初始化 GradCAM 算子
cam = GradCAM(
    model=wrapped_model, 
    target_layers=target_layers, 
    reshape_transform=vit_reshape_transform
)

for img_path in img_paths:
    img_name = os.path.basename(img_path)
    print(f"Processing: {img_name}...")
    
    # 读取原始图片用于后期叠加显示
    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        continue
    # 转换为 0-1 范围的 RGB 浮点图
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    rgb_img_float = np.float32(orig_rgb) / 255.0
    
    # 预处理输入供模型推理
    pil_img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True) # 启用输入梯度
    
    # 计算 Grad-CAM (Target = None 默认回传模型输出的第一个标量，即我们的相似度分数)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :] # 提取单张图的 2D 热力图
    
    # 强制将热力图无损缩放到与原始图片一模一样的物理分辨率（如 1024x1024 级高精出版级要求）
    grayscale_cam_resized = cv2.resize(grayscale_cam, (orig_bgr.shape[1], orig_bgr.shape[0]))
    
    # 将热力图叠加到原图上
    cam_image = show_cam_on_image(rgb_img_float, grayscale_cam_resized, use_rgb=True)
    cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    
    # 保存结果
    output_path = os.path.join(output_dir, f"cam_{img_name}")
    cv2.imwrite(output_path, cam_image_bgr)

print(f"Done! All results are saved in '{output_dir}/' folder.")