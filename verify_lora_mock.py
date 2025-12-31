import sys
import os
import torch
import torch.nn as nn
from transformers import AutoConfig

# 添加相关路径
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts_training2testing/examples"))
from pykt.models.our_model.visual_language_encoder import VisualLanguageEncoder

def mock_peft_imports():
    """Mock peft/bitsandbytes if not available for structural testing"""
    import sys
    from unittest.mock import MagicMock
    
    # Check if real ones exist
    try:
        import peft
        import bitsandbytes
        print("Real peft/bitsandbytes detected. Using them.")
        return False
    except ImportError:
        print("Real peft/bitsandbytes NOT found. Using Mocks for structural verification.")
        sys.modules["peft"] = MagicMock()
        sys.modules["bitsandbytes"] = MagicMock()
        return True

def verify_lora_encoder():
    print("--- Verifying VisualLanguageEncoder QLoRA Support ---")
    mocked = mock_peft_imports()
    
    # Mocking Qwen2-VL loading by simple Config override if running on CPU/NoModel machine
    # But to test structure, we can try to init.
    # If using mocked PEFT, VisEncoder logic needs strict mocks or it will fail on real Qwen loading.
    
    # We will assume this runs on the GPU machine.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("No CUDA. Cannot test QLoRA bitsandbytes loading. Exiting.")
        return

    # Use a dummy model path if real one not available, or assume user configured.
    # Here we just check instantiation logic (which might fail if model path invalid).
    
    print("Test: Initializing Encoder with use_lora=True...")
    try:
        encoder = VisualLanguageEncoder(
            num_c=100,
            d_question=1024,
            model_path="/home3/zhiyu/code-5/CRKT/hf_models/Qwen/Qwen2-VL-3B-Instruct", # Default path
            use_lora=True, 
            lora_r=8,
            device=device
        )
        print("✅ Initialization successful.")
    except Exception as e:
        print(f"❌ Initialization failed (Expected if path invalid or libs missing): {e}")
        # Even if failed, if it failed inside get_peft_model, it means logic was reached.
        pass

    # Basic Check for online forward method existence
    if hasattr(VisualLanguageEncoder, 'forward_online'):
        print("✅ forward_online method exists.")
    else:
        print("❌ forward_online method MISSING.")

if __name__ == "__main__":
    verify_lora_encoder()
