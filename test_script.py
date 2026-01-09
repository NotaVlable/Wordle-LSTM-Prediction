import unittest
import torch
import os
import sys

# 路径修复
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import EnhancedWordleLSTM

class TestWordleProject(unittest.TestCase):
    def test_model_output(self):
        # 测试模型输出形状是否正确
        model = EnhancedWordleLSTM(input_size=4, hidden_size=64, num_layers=1, output_size=1)
        dummy_input = torch.randn(5, 7, 4) # batch=5, seq=7, feat=4
        output = model(dummy_input)
        self.assertEqual(output.shape, (5, 1))
        print("单元测试：模型输出维度检查通过！")

if __name__ == '__main__':
    unittest.main()