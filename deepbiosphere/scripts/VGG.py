# shamelessly pulled down from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py on 1/9/2021

import math
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast
import deepbiosphere.scripts.GEOCLEF_Config as config

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        pretrained: str,
        num_spec : int,
        num_gen : int,
        num_fam : int,
        init_weights : bool = True,
    ) -> None:
        super(VGG, self).__init__()

        self.pretrained = pretrained
        self.num_spec = num_spec
        self.num_gen = num_gen
        self.num_fam = num_fam
        # https://stackoverflow.com/questions/62629114/how-to-modify-resnet-50-with-4-channels-as-input-using-pre-trained-weights-in-py        
        if pretrained == 'finetune':
            # convolves 4 band RGB-I down to 3 channels of 224x224 dimension
            self.conv4band = nn.Conv2d(4, 3, kernel_size=7, stride=1, padding=3) 
            # initialize He-style
            nn.init.kaiming_normal_(self.conv4band.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.conv4band.bias, 0)                        
            # TODO: make sure that conv4band has gradient flow

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.intermediate = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.spec = nn.Linear(4096, num_spec)
        self.gen = nn.Linear(4096, num_gen)
        self.fam = nn.Linear(4096, num_fam)
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.pretrained == 'finetune':
            x = self.conv4band(x)
        # cheap trick to get around channels dimension problem, but ah well
        # just cut out the infrared band since regular pre-trained model can
        # only handle RGB
        elif self.pretrained == 'feat_ext':
            x = x[:,:3]
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.intermediate(x)
        spec = self.spec(x)
        gen = self.gen(x)
        fam = self.fam(x)
        return (spec, gen, fam)

    def _initialize_weights(self) -> None:

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

                
                
class VGG_No_FC(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        pretrained: str,
        num_spec : int,
        num_gen : int,
        num_fam : int,
        init_weights : bool = True,
    ) -> None:
        super(VGG_No_FC, self).__init__()

        self.pretrained = pretrained
        self.num_spec = num_spec
        self.num_gen = num_gen
        self.num_fam = num_fam
        # https://stackoverflow.com/questions/62629114/how-to-modify-resnet-50-with-4-channels-as-input-using-pre-trained-weights-in-py        
        if pretrained == 'finetune':
            # convolves 4 band RGB-I down to 3 channels of 224x224 dimension
            self.conv4band = nn.Conv2d(4, 3, kernel_size=7, stride=1, padding=3) 
            # initialize He-style
            nn.init.kaiming_normal_(self.conv4band.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.conv4band.bias, 0)                        
            # TODO: make sure that conv4band has gradient flow

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # only here so feature extraction code doesn't complain
        self.intermediate = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )        
        self.spec = nn.Linear(512 * 7 * 7, num_spec)
        self.gen = nn.Linear(512 * 7 * 7, num_gen)
        self.fam = nn.Linear(512 * 7 * 7, num_fam)
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.pretrained == 'finetune':
            x = self.conv4band(x)
        # cheap trick to get around channels dimension problem, but ah well
        # just cut out the infrared band since regular pre-trained model can
        # only handle RGB
        elif self.pretrained == 'feat_ext':
            x = x[:,:3]
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        spec = self.spec(x)
        gen = self.gen(x)
        fam = self.fam(x)
        return (spec, gen, fam)

    def _initialize_weights(self) -> None:

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                

# can only use with training from scratch
class VGG_Scaled_FC(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        pretrained: str,
        num_spec : int,
        num_gen : int,
        num_fam : int,
        init_weights : bool = True,
    ) -> None:
        super(VGG_Scaled_FC, self).__init__()

        self.pretrained = pretrained
        self.num_spec = num_spec
        self.num_gen = num_gen
        self.num_fam = num_fam
        # https://stackoverflow.com/questions/62629114/how-to-modify-resnet-50-with-4-channels-as-input-using-pre-trained-weights-in-py        
        if pretrained == 'finetune':
            # convolves 4 band RGB-I down to 3 channels of 224x224 dimension
            self.conv4band = nn.Conv2d(4, 3, kernel_size=7, stride=1, padding=3) 
            # initialize He-style
            nn.init.kaiming_normal_(self.conv4band.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.conv4band.bias, 0)                        
            # TODO: make sure that conv4band has gradient flow

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        size = size_adaptive_bottleneck(num_spec, num_gen, num_fam)
        self.intermediate = nn.Sequential(
            nn.Linear(512 * 7 * 7, size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.spec = nn.Linear(size, num_spec)
        self.gen = nn.Linear(size, num_gen)
        self.fam = nn.Linear(size, num_fam)
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.pretrained == 'finetune':
            x = self.conv4band(x)
        # cheap trick to get around channels dimension problem, but ah well
        # just cut out the infrared band since regular pre-trained model can
        # only handle RGB
        elif self.pretrained == 'feat_ext':
            x = x[:,:3]
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.intermediate(x)
        spec = self.spec(x)
        gen = self.gen(x)
        fam = self.fam(x)
        return (spec, gen, fam)

    def _initialize_weights(self) -> None:

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
def size_adaptive_bottleneck(num_spec : int, num_gen : int, num_fam : int):
    scaling_factor = 4096/1000
    num_classes = num_spec + num_gen + num_fam
    return math.ceil(num_classes * scaling_factor)

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

def make_layers(cfg: List[Union[str, int]], pretrained : str, batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    if pretrained == 'none':
        in_channels = 4 # use all RGB-I if training from scratch
    else:
        in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: str, base_dir : str, progress: bool, arch_type : str, num_spec : int, num_gen : int, num_fam : int, **kwargs: Any):
    
    # I should be able to deal with model architecture here b/c load_state_dict should keymatch properly
    if pretrained is not 'none':
        kwargs['init_weights'] = False
    if arch_type == 'plain':
        model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, pretrained=pretrained), num_spec=num_spec, num_gen=num_gen, num_fam=num_fam, pretrained=pretrained, **kwargs)
    elif arch_type == 'remove_fc':
        model = VGG_No_FC(make_layers(cfgs[cfg], batch_norm=batch_norm, pretrained=pretrained), num_spec=num_spec, num_gen=num_gen, num_fam=num_fam, pretrained=pretrained, **kwargs)
    else:
        model = VGG_Scaled_FC(make_layers(cfgs[cfg], batch_norm=batch_norm, pretrained=pretrained), num_spec=num_spec, num_gen=num_gen, num_fam=num_fam, pretrained=pretrained, **kwargs)
    if pretrained != 'none':
        # going to be lazy and use load_state_dict_from_url, might change in the future
        dirr = config.setup_pretrained_dirs(base_dir) + 'VGGNet/'
        torch.hub.set_dir(dirr)
        state_dict = load_state_dict_from_url(model_urls[arch],progress=progress)
        model.load_state_dict(state_dict, strict=False)
    if pretrained == 'feat_ext':
            # for vgg         self.features = features
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.intermediate = nn.Sequential(
        set_parameter_requires_grad(model.features)
        set_parameter_requires_grad(model.avgpool)
        set_parameter_requires_grad(model.intermediate)        

    return model


def vgg11(num_spec : int, num_gen : int, num_fam : int, base_dir : str, arch_type : str, pretrained: str = 'none', progress: bool = True, **kwargs: Any):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, base_dir, progress, arch_type, num_spec=num_spec, num_gen=num_gen, num_fam=num_fam, **kwargs)


def vgg11_bn(num_spec : int, num_gen : int, num_fam : int, base_dir : str, arch_type : str, pretrained: str = 'none', progress: bool = True, **kwargs: Any):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, base_dir, progress, arch_type, num_spec=num_spec, num_gen=num_gen, num_fam=num_fam, **kwargs)



def vgg16(num_spec : int, num_gen : int, num_fam : int, base_dir : str, arch_type : str, pretrained: str = 'none', progress: bool = True, **kwargs: Any):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, base_dir, progress, arch_type, num_spec=num_spec, num_gen=num_gen, num_fam=num_fam, **kwargs)



def vgg16_bn(num_spec : int, num_gen : int, num_fam : int, base_dir : str, arch_type : str, pretrained: str = 'none', progress: bool = True, **kwargs: Any):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, base_dir, progress, arch_type, num_spec=num_spec, num_gen=num_gen, num_fam=num_fam, **kwargs)


