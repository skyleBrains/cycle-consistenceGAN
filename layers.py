import torch
import torchvision
from torch import nn
import functools
from torch.nn import init
from   torch.optim import lr_scheduler

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,
                                       track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('Normalization layer [%s] is not found' %norm_type)
    return norm_layer

def init_weights(net, init_type='normal', gain=0.02):
    def init_function(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain=0)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError("[%s] was not implemented")
            if hasattr(m, 'bias'):
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('Initialize network with %s' %init_type)
    net.apply(init_function)

# def general_conv(input_conv, filters=64, kernel=4, stride=1, padding=1, use_bias = True,
#                  use_sigmoid = False, use_relu = True, norm_layer = nn.BatchNorm2d, norm_type = 'instance'):
#
#     return subnet