import sys
import torch

from cldm.utils import create_model
from ldm.util import instantiate_from_config


def init_model(sd_weights_path, config_path, output_path):
    pretrained_weights = torch.load(sd_weights_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']
    model = create_model(config_path=config_path)
    scratch_dict = model.state_dict()
    target_dict = {}
    for sk in scratch_dict.keys():
        if sk.replace('stable_pose_adapter.', 'model.diffusion_model.') in pretrained_weights.keys():
            target_dict[sk] = pretrained_weights[sk.replace('stable_pose_adapter.', 'model.diffusion_model.')].clone()
        else:
            target_dict[sk] = scratch_dict[sk].clone()
            print('new params: {}'.format(sk))
    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_path)
    print('Done.')


if __name__ == '__main__':
    assert len(sys.argv) == 4, 'Args are wrong.'
    input_path = sys.argv[1]
    config_path = sys.argv[2]
    output_path = sys.argv[3]
    init_model(input_path, config_path, output_path)