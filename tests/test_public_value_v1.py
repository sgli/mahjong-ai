import json
from pathlib import Path
import numpy as np
import torch
from mahjong.model.policy import build_policy_model
from mahjong.training.ppo import init_ppo_from_bc

def model(coeff):
    return build_policy_model('feature-v3',781,325,(16,),model_config={
        'tile_embedding_dim':4,'river_hidden_size':8,'value_public_coefficients':coeff})

def test_public_value_matches_frozen_numpy_and_isolates_actor():
    from scripts.analyze_phase10_2_7_dev import public_features
    coeff=json.loads(Path('experiments/ppo_v4_bc_critic/phase10_2_9_residual_v1/result.json').read_text())['public_coefficients']
    m=model(coeff);x=torch.randn(3,781);x[:,397:]=37;x[:,385:389]=torch.eye(4)[:3]
    x[:,389:393]=torch.tensor([[25.,25.,25.,25.],[32.,19.,25.,24.],[18.,38.,22.,22.]])
    mask=torch.ones(3,325,dtype=torch.bool)
    expected=public_features(x.numpy())@np.asarray(coeff)
    out=m(x,mask)
    assert np.allclose(out.value.detach().numpy(),expected,atol=1e-5)
    out.value.sum().backward()
    for name,p in m.named_parameters():
        if name.startswith(('trunk.','action_head.','tile_embedding.','river_encoder.')):
            assert p.grad is None,name
    state=m.state_dict();again=model(coeff);again.load_state_dict(state,strict=True)
    assert torch.equal(out.value,again(x,mask).value)

def test_public_value_preserves_bc_actor_at_initialization():
    coeff=[0.]*17;m=model(coeff);bc=model(None)
    init_ppo_from_bc(m,bc.state_dict())
    x=torch.randn(2,781);x[:,397:]=37;x[:,385:389]=torch.eye(4)[:2]
    mask=torch.ones(2,325,dtype=torch.bool)
    assert torch.equal(m(x,mask).logits,bc(x,mask).logits)
