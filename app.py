import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib
import matplotlib.pyplot as plt
import io

# 设置页面
st.set_page_config(page_title="VGG11烧伤分类器", layout="wide")
st.title("基于VGG11的烧伤程度分类系统")

# 烧伤类型信息（中文）
BURN_TYPES = {
    0: {"name": "正常皮肤", "color": "#FFD700"},
    1: {"name": "浅二度烧伤", "color": "#FF6347"},
    2: {"name": "深二度烧伤", "color": "#CD5C5C"},
    3: {"name": "三度烧伤", "color": "#8B0000"},
    4: {"name": "电击烧伤", "color": "#4682B4"},
    5: {"name": "火焰烧伤", "color": "#FF4500"}
}

# 生成烧伤示意图的函数
def generate_burn_diagram(burn_type):
    img = Image.new('RGB', (300, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # 绘制皮肤示意图
    draw.rectangle([50, 50, 250, 150], fill="#FFE4B5", outline="black")
    
    # 根据烧伤类型添加不同效果
    if burn_type == 0:  # 正常皮肤
        draw.text((100, 80), "正常皮肤", fill="black")
    elif burn_type == 1:  # 浅二度烧伤
        draw.rectangle([50, 50, 250, 150], fill=BURN_TYPES[1]["color"], outline="black")
        draw.text((90, 80), "浅二度烧伤", fill="white")
    elif burn_type == 2:  # 深二度烧伤
        for i in range(5):
            draw.rectangle([50+i*40, 50, 90+i*40, 150], fill=BURN_TYPES[2]["color"], outline="black")
        draw.text((90, 80), "深二度烧伤", fill="white")
    elif burn_type == 3:  # 三度烧伤
        draw.rectangle([50, 50, 250, 150], fill=BURN_TYPES[3]["color"], outline="black")
        draw.text((100, 80), "三度烧伤", fill="white")
    elif burn_type == 4:  # 电击烧伤
        for i in range(3):
            draw.line([70+i*60, 50, 70+i*60, 150], fill=BURN_TYPES[4]["color"], width=5)
        draw.text((90, 80), "电击烧伤", fill=BURN_TYPES[4]["color"])
    elif burn_type == 5:  # 火焰烧伤
        for i in range(3):
            draw.polygon([150, 50+i*30, 100, 100+i*30, 200, 100+i*30], 
                         fill=BURN_TYPES[5]["color"], outline="black")
        draw.text((90, 80), "火焰烧伤", fill=BURN_TYPES[5]["color"])
    
    return img

# 模型加载函数
@st.cache_resource
def load_models():
    # 加载VGG11模型
    vgg11 = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
    vgg11.eval()
    
    # 加载分类器
    classifier = joblib.load('pretrained_classifier.pkl')
    
    return vgg11, classifier

# 图像预处理
def preprocess_image(image):
    # 确保图像是RGB格式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# 特征提取
def extract_features(model, image_tensor):
    with torch.no_grad():
        x = model.features(image_tensor)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        # 获取倒数第二层特征(4096维)
        for i in range(6):  # 跳过最后一层
            x = model.classifier[i](x)
    return x.numpy()

# 主函数
def main():
    # 加载模型
    vgg11_model, classifier = load_models()
    
    # 文件上传
    uploaded_file = st.file_uploader("上传烧伤部位图片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # 显示上传的图片
            image = Image.open(uploaded_file)
            st.image(image, caption='上传的烧伤图片', use_column_width=True)
            
            # 预处理和预测
            image_tensor = preprocess_image(image)
            features = extract_features(vgg11_model, image_tensor)
            
            # 检查特征维度
            if features.shape[1] != classifier.n_features_in_:
                st.error(f"特征维度不匹配！需要 {classifier.n_features_in_} 维，得到 {features.shape[1]} 维")
                return
            
            # 预测
            prediction = classifier.predict(features)
            prediction_proba = classifier.predict_proba(features)
            
            # 显示结果
            st.subheader("预测结果")
            
            # 获取预测的烧伤类型
            predicted_type = BURN_TYPES[prediction[0]]["name"]
            st.success(f"预测烧伤类型: **{predicted_type}**")
            
            # 生成并显示示意图
            diagram = generate_burn_diagram(prediction[0])
            st.image(diagram, caption=f"{predicted_type}示意图", width=300)
            
            st.subheader("各类别概率分布")
            
            # 创建概率条形图
            fig, ax = plt.subpl