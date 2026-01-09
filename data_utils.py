import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler

def prepare_data(window_size=7):
    # 路径设置
    base_path = "/Users/Administrator/Desktop/大数据大作业"
    file_path = os.path.join(base_path, "2023_MCM_Problem_C_Data.xlsx")
    
    # 1. 加载与清洗
    df = pd.read_excel(file_path, header=1).iloc[:, 1:]
    df.columns = ['Date', 'Contest_number', 'Word', 'Total_Reported', 'Hard_Mode', 
                  'Try_1', 'Try_2', 'Try_3', 'Try_4', 'Try_5', 'Try_6', 'Try_7_plus']

    # 2. 特征工程 (计算目标值与多源特征)
    df['Mean_Tries'] = (df['Try_1']*1 + df['Try_2']*2 + df['Try_3']*3 + 
                        df['Try_4']*4 + df['Try_5']*5 + df['Try_6']*6 + 
                        df['Try_7_plus']*7) / 100

    df['vowels'] = df['Word'].apply(lambda x: sum(1 for c in str(x).lower() if c in 'aeiou'))
    df['unique_chars'] = df['Word'].apply(lambda x: len(set(str(x).lower())))
    
    # 保存清洗后的数据到 CSV
    cleaned_data_path = os.path.join(base_path, "cleaned_wordle_data.csv")
    df.to_csv(cleaned_data_path, index=False)
    print(f"预处理完成，清洗后的数据已保存至: {cleaned_data_path}")

    # 3. 准备模型输入
    features = df[['Hard_Mode', 'vowels', 'unique_chars', 'Total_Reported']].values
    target = df['Mean_Tries'].values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X, y = [], []
    for i in range(len(features_scaled) - window_size):
        X.append(features_scaled[i:i+window_size])
        y.append(target[i+window_size])
    
    return (torch.tensor(np.array(X), dtype=torch.float32), 
            torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1), 
            target[window_size:])
