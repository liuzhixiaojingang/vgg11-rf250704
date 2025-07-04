import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib
import os

# 设置页面
st.set_page_config(page_title="VGG11图像分类器", layout="wide")
st.title("基于VGG11的图像分类预测")

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
    uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # 显示上传的图片
            image = Image.open(uploaded_file)
            st.image(image, caption='上传的图片', use_column_width=True)
            
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
            st.write(f"预测类别: {prediction[0]}")
            
            st.subheader("类别概率")
            for i, prob in enumerate(prediction_proba[0]):
                st.write(f"类别 {i}: {prob:.4f}")
        
        except Exception as e:
            st.error(f"处理图像时出错: {str(e)}")

if __name__ == "__main__":
    main()