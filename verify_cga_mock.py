
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts_training2testing/examples')))

from pykt.models.our_model.vcrkt import VCRKT

def test_cga_forward():
    config = {
        'num_q': 50,
        'num_c': 10,
        'dim_qc': 200,
        'd_question': 1024,
        'd_question_repr': 200, # Expected for CGA
        'dropout': 0.1,
        'dataset_name': 'test_dataset',
        'question_rep_type': 'cga',
        'emb_type': 'qkcs',
        'mllm_name': 'dummy', # prevent loading real encoder
        'cache_dir': 'features'
    }
    
    print("Initializing VCRKT with CGA mode...")
    model = VCRKT(config)
    model.to(model.device) # Move entire model (including VCRKTNet) to GPU
    
    # Mock visual encoder (since we don't load real one)
    # We need to manually monkeypatch visual encoder or just mock its call?
    # Or rely on build_img_path_dict being empty and it failing?
    # Wait, if visual encoder fails to load or img_path_dict is empty, vcrkt will return None for visual?
    # vcrkt:306 v_raw, _ = self.visual_encoder(qids, ...)
    # If we don't want to load Qwen, we should mock self.visual_encoder
    
    class MockEncoder(torch.nn.Module):
        def forward(self, qids, img_paths, return_kc=False):
            bz, seq = qids.shape
            return torch.randn(bz, seq, 1024), None
            
    model.visual_encoder = MockEncoder()
    model.visual_proj = torch.nn.Linear(1024, 200).to(model.device) # Re-init on correct device
    
    # Create dummy data
    bz, seq_len = 2, 5
    data = {
        'qseqs': torch.randint(0, 50, (bz, seq_len)),
        'cseqs': torch.randint(0, 10, (bz, seq_len, 3)), # 3 concepts per question
        'rseqs': torch.randint(0, 2, (bz, seq_len)),
        'shft_qseqs': torch.randint(0, 50, (bz, seq_len)),
        'shft_rseqs': torch.randint(0, 2, (bz, seq_len)),
        'smasks': torch.ones(bz, seq_len).bool()
    }
    
    print("Running train_one_step...")
    try:
        y, loss = model.train_one_step(data)
        print(f"Success! Output shape: {y.shape}, Loss: {loss.item()}")
    except Exception as e:
        print(f"Failed! Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cga_forward()
