import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = "/Users/Administrator/Desktop/大数据大作业"
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from model import EnhancedWordleLSTM
from data_utils import prepare_data

def main():
    # 1. 准备数据
    X, y, actual_vals = prepare_data()

    # 2. 初始化模型
    model = EnhancedWordleLSTM(input_size=4, hidden_size=128, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # 3. 训练过程
    print("正在进行模块化 Bi-LSTM 训练...")
    for epoch in range(500):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/500], Loss: {loss.item():.4f}')

    # 4. 模型评估
    model.eval()
    with torch.no_grad():
        preds = model(X).numpy()
    
    mae = mean_absolute_error(actual_vals, preds)
    rmse = mean_squared_error(actual_vals, preds)**0.5
    print(f"\n训练完成! 评估指标:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # 5. 保存结果图
    plt.figure(figsize=(12, 6))
    plt.plot(actual_vals, label='Actual Data', alpha=0.5, color='blue')
    plt.plot(preds, label='Bi-LSTM Prediction', linestyle='--', color='red')
    plt.title(f"Wordle Prediction Analysis (MAE: {mae:.4f})")
    plt.xlabel("Timeline")
    plt.ylabel("Mean Tries")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(BASE_DIR, "final_result.png")
    plt.savefig(save_path)
    print(f"分析结果图已保存至: {save_path}")

if __name__ == "__main__":
    main()
