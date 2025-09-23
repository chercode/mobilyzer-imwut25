import torch
from .MST_Plus_Plus import MST_Plus_Plus

def model_generator(method, pretrained_model_path=None, compressed=False, bands=68):
    if method == 'mst_plus_plus':
        if compressed:
            model = MST_Plus_Plus(in_channels=4, out_channels=bands, n_feat=bands//2, msab_stages=2, stage=1)
        else:
            model = MST_Plus_Plus(in_channels=4, out_channels=bands)

    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)
    return model
