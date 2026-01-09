import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler

def prepare_data(window_size=7):
    # 固定的数据路径
    base_path = "/Users/Administrator/Desktop/大数据大作业"
    file_path = os.path.join(base_path, "2023_MCM_Problem_C_Data.xlsx")
    
    # 加载与清洗
    df = pd.read_excel(file_path, header=1).iloc[:, 1:]
    df.columns = ['Date', 'Contest_number', 'Word', 'Total_Reported', 'Hard_Mode', 
                  'Try_1', 'Try_2', 'Try_3', 'Try_4', 'Try_5', 'Try_6', 'Try_7_plus']

    # 计算目标值
    df['Mean_Tries'] = (df['Try_1']*1 + df['Try_2']*2 + df['Try_3']*3 + 
                        df['Try_4']*4 + df['Try_5']*5 + df['Try_6']*6 + 
                        df['Try_7_plus']*7) / 100

    # 特征工程
    df['vowels'] = df['Word'].apply(lambda x: sum(1 for c in str(x).lower() if c in 'aeiou'))
    df['unique_chars'] = df['Word'].apply(lambda x: len(set(str(x).lower())))
    
    features = df[['Hard_Mode', 'vowels', 'unique_chars', 'Total_Reported']].values
    target = df['Mean_Tries'].values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 序列化
    X, y = [], []
    for i in range(len(features_scaled) - window_size):
        X.append(features_scaled[i:i+window_size])
        y.append(target[i+window_size])
    
    return (torch.tensor(np.array(X), dtype=torch.float32), 
            torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1), 
            target[window_size:])