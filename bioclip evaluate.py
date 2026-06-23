import os
import torch
import open_clip
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score

def evaluate_on_dataset(model_name, dataset_dir, species_list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[正在初始化] -> {model_name} ...")
    
    # 1. 加载模型与对应的图像预处理流
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    # 2. 构建针对这 5 个凤尾藓特定物种的 Prompt
    text_prompts = [f"a photo of {sp}, a type of moss" for sp in species_list]
    text_tokens = tokenizer(text_prompts).to(device)

    # 预先计算文本特征并归一化
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    y_true = []      # 真实标签索引
    y_pred_top1 = [] # 预测的 Top-1 标签索引
    top3_correct = 0
    total_images = 0

    print(f"正在使用 {model_name} 推理图片...")
    # 3. 遍历你的 5 个凤尾藓文件夹
    for true_idx, species_name in enumerate(species_list):
        folder_path = os.path.join(dataset_dir, species_name)
        if not os.path.exists(folder_path):
            print(f"⚠️ 警告: 找不到文件夹 {folder_path}，已跳过。")
            continue
            
        # 支持常见图像格式
        img_names = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            try:
                img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # 提取图像特征并归一化
                    img_features = model.encode_image(img)
                    img_features /= img_features.norm(dim=-1, keepdim=True)
                    
                    # 计算图文匹配概率
                    similarity = (100.0 * img_features @ text_features.T).softmax(dim=-1)
                    
                    # 提取前 3 名索引
                    _, top_indices = torch.topk(similarity[0], k=3)
                    top_indices = top_indices.cpu().tolist()

                # 记录统计结果
                y_true.append(true_idx)
                y_pred_top1.append(top_indices[0])
                
                if true_idx in top_indices:
                    top3_correct += 1
                total_images += 1
                
            except Exception as e:
                print(f"⚠️ 图片 {img_name} 读取或推理失败，已跳过。错误信息: {e}")
                continue

    # 4. 计算三大核心学术指标
    if total_images == 0:
        print(f"❌ 模型 {model_name} 未检测到任何可运行的图像。")
        return 0, 0, 0

    # Top-1 Accuracy
    top1_correct = sum(1 for gt, pr in zip(y_true, y_pred_top1) if gt == pr)
    top1_acc = top1_correct / total_images
    
    # Top-3 Accuracy
    top3_acc = top3_correct / total_images
    
    # Macro F1-score
    macro_f1 = f1_score(y_true, y_pred_top1, average='macro')

    print(f"✨ {model_name} 评估完成！总图片数: {total_images} | Top-1: {top1_acc:.2%} | Top-3: {top3_acc:.2%} | Macro F1: {macro_f1:.2%}")
    return top1_acc, top3_acc, macro_f1

def main():
    # 📌 完美契合你的截图：将数据根目录指定为当前路径下的 "./data"
    dataset_dir = "./data" 
    
    if not os.path.exists(dataset_dir):
        print(f"❌ 错误：在当前目录下找不到 '{dataset_dir}' 文件夹，请确认脚本运行路径是否正确。")
        return

    # 自动获取并排序这 5 个物种文件夹的名字
    species_list = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print(f"🌿 成功关联测试的 5 个凤尾藓属物种: {species_list}\n" + "-"*60)

    # 待评测的三代生物学多模态大模型
    models_to_test = {
       "BioCLIP (初代)": "hf-hub:imageomics/bioclip",
        "BioCLIP 2": "hf-hub:imageomics/bioclip-2",
        "BioCLIP 2.5": "hf-hub:imageomics/bioclip-2.5-vith14"
    }

    results = {}
    
    # 依次跑完三个模型
    for label, path in models_to_test.items():
        t1, t3, f1 = evaluate_on_dataset(path, dataset_dir, species_list)
        if t1 or t3 or f1: # 排除完全没读到图的异常情况
            results[label] = {
                "Top-1 Acc": f"{t1:.2%}",
                "Top-3 Acc": f"{t3:.2%}",
                "Macro F1-score": f"{f1:.2%}"
            }

    # 5. 自动打印精美的三代模型性能大比拼表格
    if results:
        print("\n" + "="*18 + " 📊 三代 BioCLIP 凤尾藓分类终极报告 " + "="*18)
        df_report = pd.DataFrame(results).T
        print(df_report.to_string())
        print("="*70)
    else:
        print("未生成任何有效数据，请检查各个物种文件夹下是否存在图片。")

if __name__ == "__main__":
    main()