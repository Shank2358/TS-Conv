from collections import OrderedDict

import torch
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from . import model_resnet


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int):
        """The function takes in a backbone and a number of channels and returns a body with the layers of
        the backbone

        Parameters
        ----------
        backbone : nn.Module
            the backbone network
        num_channels : int
            the number of channels in the output feature map.

        """
        super().__init__()
        for name, parameter in backbone.named_parameters():
            # print(name, parameter.shape)
            if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)

        # return_layers = {"layer2": "3", "layer3": "5", "layer4": "2"}
        return_layers = {"layer2": "3", "layer3": "22", "layer4": "2"}

        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x):
        xs = self.backbone(x)

        fmp_list = []
        for name, fmp in xs.items():
            fmp_list.append(fmp)

        return fmp_list[0], fmp_list[1], fmp_list[2]


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, pretrained: bool, dilation: bool, norm_type: str):
        if norm_type == "BN":
            norm_layer = nn.BatchNorm2d
        elif norm_type == "FrozeBN":
            norm_layer = FrozenBatchNorm2d
        # NOTE: get the backbone network
        backbone = getattr(model_resnet, name)(
            replace_stride_with_dilation=[False, False, dilation], pretrained=pretrained, norm_layer=norm_layer
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, num_channels)


# def build_resnet(model_name="resnet18", pretrained=False, norm_type="BN"):
#     if model_name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnext101_32x8d"]:
#         backbone = Backbone(model_name, pretrained, dilation=False, norm_type=norm_type)
#     elif model_name in ["resnet50-d", "resnet101-d"]:
#         backbone = Backbone(model_name[:-2], pretrained, dilation=True, norm_type=norm_type)

#     # return backbone, backbone.num_channels
#     return backbone


# A function that returns a model.
def Resnet50(pretrained=False, norm_type="BN"):
    """`Resnet50` is a function that returns a `Backbone` object with the following parameters:

    - `name`: "resnet50"
    - `pretrained`: False
    - `dilation`: False
    - `norm_type`: "BN"

    The `Backbone` object is a class that is defined in `backbone.py`

    Parameters
    ----------
    pretrained, optional
        Whether to use a pretrained model.
    norm_type, optional
        BN, GN, or SN

    Returns
    -------
        A backbone object with the following parameters:
        - name: resnet50
        - pretrained: False
        - dilation: False
        - norm_type: BN

    """

    return Backbone("resnet50", pretrained, dilation=False, norm_type=norm_type)

# A function that returns a model.
def Resnet101(pretrained=False, norm_type="BN"):
    """`Resnet50` is a function that returns a `Backbone` object with the following parameters:

    - `name`: "resnet50"
    - `pretrained`: False
    - `dilation`: False
    - `norm_type`: "BN"

    The `Backbone` object is a class that is defined in `backbone.py`

    Parameters
    ----------
    pretrained, optional
        Whether to use a pretrained model.
    norm_type, optional
        BN, GN, or SN

    Returns
    -------
        A backbone object with the following parameters:
        - name: resnet50
        - pretrained: False
        - dilation: False
        - norm_type: BN

    """

    return Backbone("resnet101", pretrained, dilation=False, norm_type=norm_type)

if __name__ == "__main__":
    model = Resnet101(pretrained=False, norm_type="BN")
    model_list = model.state_dict().keys()
    # print(model_list)
    weight = torch.load(
        "D:\\Github\\v2\\weight\\resnet101-cd907fc2.pth"
    )
    # print(weight.keys())
    new_weight = OrderedDict()
    # # zip 默认遍历最少的list
    for model_key, weight_key, weight_value in zip(model_list, weight.keys(), weight.values()):
        if model_key[9:] == weight_key:
            new_weight[model_key] = weight_value
    model.load_state_dict(new_weight)


    # print('hello world')
    # print(type(feat_dim))
    # x = torch.randn(3, 3, 800, 800)
    # x_s, x_m, x_l = model(x)
    # print(x_s.size())
    # print(x_m.size())
    # print(x_l.size())
    # from rich import print
    # model_keys = list(model.state_dict().keys())
    # print(model_keys)
