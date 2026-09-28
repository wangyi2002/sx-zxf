"""Strict initialization and state checks for frozen-baseline SRDA."""
import hashlib
from pathlib import Path
import torch

def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

def frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in unwrap(model).state_dict().items():
        if not name.startswith('depth_adapter.'):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def initialize_frozen_srda(model, checkpoint_path, output_dir):
    model = unwrap(model)
    if model.depth_adapter is None:
        raise ValueError('Frozen experiment requires SRDA')
    source = Path(checkpoint_path).resolve()
    for filename in ('best_epoch.pth.tr', 'latest_epoch.pth.tr'):
        output = (Path(output_dir) / filename).resolve()
        if source == output or (output.exists() and source.samefile(output)):
            raise ValueError('Baseline source must not be an output checkpoint')
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    state = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    expected = {k for k in model.state_dict() if not k.startswith('depth_adapter.')}
    if set(state) != expected:
        raise ValueError(f'Expected clean baseline. Missing={sorted(expected-set(state))}; extra={sorted(set(state)-expected)}')
    merged = model.state_dict()
    merged.update(state)
    model.load_state_dict(merged, strict=True)
    last = model.depth_adapter.mlp[-1]
    if torch.count_nonzero(last.weight) or torch.count_nonzero(last.bias):
        raise ValueError('Adapter must be freshly zero initialized')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('depth_adapter.'))
    model.eval()
    model.depth_adapter.train()
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return str(source), digest.hexdigest()

def set_frozen_training_mode(model):
    model.eval()
    unwrap(model).depth_adapter.train()

def check_zero_output(model, inputs):
    model.eval()
    with torch.no_grad():
        full = model(inputs)
        raw = model(inputs, depth_adapter_alpha=0.0)
    if not torch.isfinite(full).all() or not torch.equal(full, raw):
        raise RuntimeError('Zero SRDA must match loaded baseline output exactly')
