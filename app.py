import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
print('⚙️  torth 설정 완료')
from sklearn.metrics import average_precision_score
import numpy as np
from tqdm import tqdm
import collections
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
from public_function import *
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
print('⚙️  환경설정 완료')
import streamlit as st

if 'transform' not in st.session_state:
    st.session_state.transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 이미지 크기를 512x512로 고정 📐
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])    
if 'train_dataset' not in st.session_state:
    st.session_state.train_dataset = torchvision.datasets.ImageFolder(root='my_dataset', transform=st.session_state.transform)
if 'sampler' not in st.session_state:
    st.session_state.sampler = MPerClassSampler(st.session_state.train_dataset.targets, m=4, length_before_new_iter=len(st.session_state.train_dataset))
if 'train_dataloader' not in st.session_state:
    st.session_state.train_dataloader = torch.utils.data.DataLoader(
        st.session_state.train_dataset, batch_size=44, shuffle=False, sampler=st.session_state.sampler
    )
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if 'model' not in st.session_state:
    # 기본적인 모델
    class ResNet18FeatureExtractor(nn.Module):
        def __init__(self):
            super(ResNet18FeatureExtractor, self).__init__()
            resnet = torchvision.models.resnet18(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        def forward(self, x):
            x = self.feature_extractor(x)
            return x.view(x.size(0), -1)
    st.session_state.model = ResNet18FeatureExtractor()
    st.session_state.model.load_state_dict(torch.load("trained_model.pth", map_location=st.session_state.device))
    st.session_state.model = st.session_state.model.to(st.session_state.device)
    st.session_state.model.eval()
    

def extract(model, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="특징 추출"):
            # torch에 넣기
            inputs = inputs.to(st.session_state.device)
            outputs = model(inputs)
            
            # cpu로 전환 (cpu가 더 빠름)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
            
    # 반환
    return np.concatenate(features), np.concatenate(labels)    

if 'feature_extract' not in st.session_state:
    st.session_state.feature_extract, st.session_state.label = extract(st.session_state.model, st.session_state.train_dataloader)
    
    
def extract_one(image):
    image = Image.open(image)
    image = image.convert('RGB')
    x = st.session_state.transform(image).unsqueeze(0)
    dummy_label = torch.tensor([0])

    dataset = TensorDataset(x, dummy_label)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    feature, label = extract(st.session_state.model, dataloader)
    return feature, label

# 유클리드 거리
def euclidean_distance(query_feature, database_features):
    distances = np.linalg.norm(database_features - query_feature, axis=1)
    return distances

# 검색 결과 얻기
def get_top_k_results(query, features, labels, k=5):
    distances = euclidean_distance(query, features)
    indices = np.argsort(distances)[:k]
    return indices, labels[indices]


# 이미지 업로드 받기
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    feature, _ = extract_one(uploaded_file)
    idx, labels = get_top_k_results(feature,  st.session_state.feature_extract,  st.session_state.label, k=15)
    for i in idx:
        path, label_index =  st.session_state.train_dataset.samples[i]
        label_name =  st.session_state.train_dataset.classes[label_index]  # 문자 라벨명
        st.write(f'### 🔖라벨: {label_name}')
        st.image(path)