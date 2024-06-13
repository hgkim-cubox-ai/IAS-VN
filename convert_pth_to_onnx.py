import torch
from collections import OrderedDict

from models import LBPModel
# from models.lbp_model_infer import LBPModel


def main():
    model = LBPModel(
        {
            'backbone': 'resnet50',
            'regressor': [2048, 256, 16, 1],
            'lbp_in_model': False,
            'Data': {'batch_size': 1}
        }
    )
    tmp = torch.load('models/ias_model.pth')['state_dict']
    state_dict = OrderedDict()
    for n, v in tmp.items():
        if n == 'module.lbp_layer.lbp_kernel.weight':
            continue
        state_dict[n[7:]] = v
    model.load_state_dict(state_dict)
    
    dummy_img = torch.randn(1, 3, 144, 224)
    dummy_lbp_hist = torch.randn(1, 256)
    
    torch.onnx.export(
        model,
        (dummy_img, dummy_lbp_hist),
        'models/ias_model.onnx',
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['image', 'lbp_hist'],
        output_names=['output'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'lbp_hist': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    # model = LBPModel()
    # tmp = torch.load('models/ias_model.pth')['state_dict']
    # state_dict = OrderedDict()
    # for n, v in tmp.items():
    #     state_dict[n[7:]] = v
    # model.load_state_dict(state_dict)
    
    # dummy = torch.randn(1, 3, 144, 224)
    # torch.onnx.export(
    #     model,
    #     dummy,
    #     'models/ias_infer.onnx',
    #     export_params=True,
    #     opset_version=15,
    #     do_constant_folding=True,
    #     input_names=['input'],
    #     output_names=['output'],
    #     dynamic_axes={
    #         'input': {0: 'batch_size'},
    #         'output': {0: 'batch_size'}
    #     }
    # )


if __name__ == '__main__':
    main()
    print('Done')