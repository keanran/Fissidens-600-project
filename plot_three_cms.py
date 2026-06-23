import os
import torch
import open_clip
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_cm_for_model(model_label, model_path, dataset_dir, species_list, device):
    print(f"\n[🚀 正在处理] -> {model_label} ...")
    
    # 1. 加载模型
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_path)
        tokenizer = open_clip.get_tokenizer(model_path)
        model = model.to(device).eval()
    except Exception as e:
        print(f"⚠️ 模型 {model_label} 加载失败，跳过。错误: {e}")
        return

    text_prompts = [f"a photo of {sp}, a type of moss" for sp in species_list]
    text_tokens = tokenizer(text_prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    y_true = []
    y_pred = []

    # 2. 预测 600 张图
    for true_idx, species_name in enumerate(species_list):
        folder_path = os.path.join(dataset_dir, species_name)
        if not os.path.exists(folder_path):
            continue
        img_names = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        for img_name in img_names:
            try:
                img = preprocess(Image.open(os.path.join(folder_path, img_name))).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_features = model.encode_image(img)
                    img_features /= img_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * img_features @ text_features.T).softmax(dim=-1)
                    pred_idx = similarity[0].argmax().item()
                
                y_true.append(true_idx)
                y_pred.append(pred_idx)
            except:
                continue

    # 3. 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 4. 绘制并保存热力图
    plt.figure(figsize=(8, 6), dpi=300) # 300 DPI 达到期刊印刷级别
    
    # 使用 Blues 渐变色，annot=True 显示具体图片张数，fmt='d' 保证显示整数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=species_list, yticklabels=species_list,
                cbar_kws={'label': 'Number of images'})
    
    plt.title(f'Confusion Matrix: {model_label}', fontsize=14, pad=15, fontweight='bold')
    plt.ylabel('True Species (Ground Truth)', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Species', fontsize=12, labelpad=10)
    plt.xticks(rotation=35, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 文件名格式化，例如：cm_bioclip_2.5.png
    safe_name = model_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_path = f"cm_{safe_name}.png"
    
    plt.savefig(save_path)
    plt.close() # 关闭当前画布，防止多图重叠
    print(f"✅ 成功生成并保存：{save_path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_dir = "./data" 
    
    if not os.path.exists(dataset_dir):
        print(f"❌ 错误：找不到 '{dataset_dir}' 文件夹。")
        return

    # 获取你的 5 个凤尾藓学名
    species_list = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print(f"🌿 检测到物种池: {species_list}")

    # 三个模型的配置字典
    models_to_run = {
        "BioCLIP 初代": "hf-hub:imageomics/bioclip",
        "BioCLIP 2": "hf-hub:imageomics/bioclip-2",
        "BioCLIP 2.5": "hf-hub:imageomics/bioclip-2.5-vith14"
    }

    # 循环调用绘图函数
    for label, path in models_to_run.items():
        generate_cm_for_model(label, path, dataset_dir, species_list, device)
        
    print("\n🏁 所有模型的混淆矩阵图已全部生成完毕！快去项目根目录下查看吧。")

if __name__ == "__main__":
    main()