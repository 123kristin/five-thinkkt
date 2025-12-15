#!/usr/bin/env python
"""
快速测试 CoTGenerator 初始化过程，定位卡住的位置
"""
import os
import sys
import torch

# 设置环境变量
os.environ['CURRENT_GPU_ID'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("=" * 60)
print("开始测试 CoTGenerator 初始化")
print("=" * 60)
sys.stdout.flush()

print("步骤1: 导入模块...")
sys.stdout.flush()
sys.path.insert(0, 'scripts_training2testing/examples')
from pykt.models.our_model.cot.cot_generator import CoTGenerator

print("步骤2: 创建 CoTGenerator 对象...")
sys.stdout.flush()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
sys.stdout.flush()

try:
    cot_gen = CoTGenerator(
        mllm_name="/home3/zhiyu/code-5/CRKT/five-thinkkt/hf_models/Qwen/Qwen2-VL-3B-Instruct",
        text_encoder_name="/home3/zhiyu/code-5/CRKT/five-thinkkt/hf_models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        d_cot=384,
        cache_dir="cot_cache",
        device=device,
        use_cache=True
    )
    print("步骤3: CoTGenerator 初始化完成！")
    sys.stdout.flush()
    
    print("步骤4: 测试文本编码器加载...")
    sys.stdout.flush()
    cot_gen._load_text_encoder()
    print("步骤5: 文本编码器加载完成！")
    sys.stdout.flush()
    
    print("=" * 60)
    print("所有步骤完成，没有卡住")
    print("=" * 60)
    sys.stdout.flush()
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()

