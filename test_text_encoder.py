#!/usr/bin/env python
"""
测试文本编码器加载和初始化
"""
import os
import sys
import torch

os.environ['CURRENT_GPU_ID'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("=" * 60)
print("测试文本编码器加载")
print("=" * 60)
sys.stdout.flush()

print("步骤1: 导入模块...")
sys.stdout.flush()
from sentence_transformers import SentenceTransformer

print("步骤2: 加载模型...")
sys.stdout.flush()
model_name = "/home3/zhiyu/code-5/CRKT/five-thinkkt/hf_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

print("步骤3: 创建SentenceTransformer对象...")
sys.stdout.flush()
try:
    model = SentenceTransformer(model_name, device=str(device))
    print("步骤4: 模型对象创建成功")
    sys.stdout.flush()
    
    print("步骤5: 测试编码（首次调用可能较慢）...")
    sys.stdout.flush()
    test_text = "这是一个测试文本"
    embed = model.encode(test_text, convert_to_tensor=True)
    print(f"步骤6: 编码成功，输出维度: {embed.shape}")
    sys.stdout.flush()
    
    print("步骤7: 再次编码测试...")
    sys.stdout.flush()
    embed2 = model.encode("另一个测试文本", convert_to_tensor=True)
    print(f"步骤8: 第二次编码成功")
    sys.stdout.flush()
    
    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()

