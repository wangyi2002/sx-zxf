"""CPU-only shape/finite/parameter check with an explicit reference scan."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tools.cpu_scan_reference import enable_cpu_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/h36m/MotionAGFormer-large.yaml')
    parser.add_argument('--output', default='depth_outputs/cpu_smoke.json')
    opts = parser.parse_args()
    enable_cpu_reference()
    from utils.learning import load_model
    from utils.tools import get_config
    torch.manual_seed(0)
    torch.set_num_threads(2)
    config = get_config(opts.config)
    model = load_model(config).cpu().eval()
    inputs = torch.randn(1, config.n_frames, config.num_joints, config.dim_in)
    inputs[..., 2] = torch.rand_like(inputs[..., 2])
    with torch.inference_mode():
        output = model(inputs)
    expected = (1, config.n_frames, config.num_joints, config.dim_out)
    assert tuple(output.shape) == expected, (output.shape, expected)
    assert torch.isfinite(output).all(), 'NaN/Inf in model output'
    result = dict(config=opts.config, input_shape=list(inputs.shape), output_shape=list(output.shape),
                  input_finite=bool(torch.isfinite(inputs).all()), output_finite=True,
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  scan='explicit CPU reference; production model files unchanged',
                  added_model_parameters=0)
    path = Path(opts.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
