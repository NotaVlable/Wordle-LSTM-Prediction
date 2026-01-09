import torch
import torch.nn as nn

class EnhancedWordleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EnhancedWordleLSTM, self).__init__()
        # 双向 LSTM，捕捉波动更灵敏
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])