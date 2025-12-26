
import os
import sys
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Ensure we can import the project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.join(current_dir, 'scripts_training2testing/examples')
sys.path.insert(0, examples_dir)

# Define a Dummy class to replace VisualLanguageEncoder
class DummyVisualEncoder(nn.Module):
    def __init__(self, num_c, d_question, model_path, cache_dir, dataset_name, use_cache, device, **kwargs):
        super().__init__()
        print(f"[DummyVisualEncoder] Initialized with d_question={d_question}")
        self.d_question = d_question
        self.device = device
        
    def forward(self, qids, img_path_dict, return_kc=False):
        # qids is [bz, seq_len]
        bz, seq_len = qids.shape
        # Output shape: (bz, seq_len, d_question)
        v_t = torch.randn(bz, seq_len, self.d_question).to(self.device)
        k_t = None
        return v_t, k_t

def test_vcrkt_vq_mode():
    with patch('pykt.models.our_model.vcrkt.VisualLanguageEncoder', DummyVisualEncoder):
        from pykt.models.our_model.vcrkt import VCRKT
        
        print("Testing VCRKT with question_rep_type='v&q'...")
        
        config = {
            'emb_type': 'qkcs',
            'question_rep_type': 'v&q',
            'dataset_name': 'DBE_KT22',
            'd_question': 1024,
            'num_c': 100,
            'num_q': 100,
            'dim_qc': 200,
            'dropout': 0.1,
            'mllm_name': 'dummy_path',
        }
        
        data_config = {
            'dpath': '/dummy/path', 
        }
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Initialize model
        model = VCRKT(config, data_config).to(device)
        print("Model initialized successfully.")
        
        # Verify both QEmbs and Visual Components are present
        assert model.QEmbs is not None
        assert model.visual_proj is not None
        assert model.visual_encoder is not None
        print("Components verified: QEmbs, VisualProj, VisualEncoder present.")
        
        # Verify d_question_repr calculation in Net
        # v&q mode: 200 (QID) + 200 (Vis Proj) = 400
        assert model.model.d_question_repr == 400
        print(f"Verified VCRKTNet d_question_repr = {model.model.d_question_repr}")
        
        # Create dummy data (3D concepts!)
        bz = 4
        seq_len = 10
        max_concepts = 4
        
        qseqs = torch.randint(0, 100, (bz, seq_len)).to(device)
        cseqs = torch.randint(0, 100, (bz, seq_len, max_concepts)).to(device)
        rseqs = torch.randint(0, 2, (bz, seq_len)).to(device)
        shft_qseqs = torch.randint(0, 100, (bz, seq_len)).to(device)
        shft_rseqs = torch.randint(0, 2, (bz, seq_len)).to(device)
        smasks = torch.ones(bz, seq_len).bool().to(device)
        
        data = {
            'qseqs': qseqs,
            'cseqs': cseqs,
            'rseqs': rseqs,
            'shft_qseqs': shft_qseqs,
            'shft_rseqs': shft_rseqs,
            'smasks': smasks
        }
        
        # Run train_one_step
        print("Running train_one_step...")
        y, loss = model.train_one_step(data)
        print(f"train_one_step output shape: {y.shape}, loss: {loss.item()}")
        
        # Run predict_one_step
        print("Running predict_one_step...")
        y_pred = model.predict_one_step(data)
        print(f"predict_one_step output shape: {y_pred.shape}")
        
        print("Test passed!")

if __name__ == "__main__":
    test_vcrkt_vq_mode()
