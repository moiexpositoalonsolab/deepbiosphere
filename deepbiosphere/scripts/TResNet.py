# shamelessly pulled down from on 1/9/2021

import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
import torch.nn.functional as F
from torch.nn import Module as Module
from collections import OrderedDict
from inplace_abn import InPlaceABN
import deepbiosphere.scripts.GEOCLEF_Config as config

# from inplace_abn import ABN

model_files = {
    'MSCOCO_TResnetL': 'MS_COCO_TRresNet_L_448_86.6.pth',
}




class bottleneck_head(nn.Module):
    def __init__(self, num_features, num_classes, bottleneck_features=200):
        super(bottleneck_head, self).__init__()
        self.embedding_generator = nn.ModuleList()
        self.embedding_generator.append(nn.Linear(num_features, bottleneck_features))
        self.embedding_generator = nn.Sequential(*self.embedding_generator)
        self.FC = nn.Linear(bottleneck_features, num_classes)

    def forward(self, x):
        self.embedding = self.embedding_generator(x)
        logits = self.FC(self.embedding)
        return logits


def conv2d(ni, nf, stride):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(nf),
        nn.ReLU(inplace=True)
    )


def conv2d_ABN(ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
                  bias=False),
        InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param)
    )


class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                self.conv1 = nn.Sequential(conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv2d_ABN(planes, planes, stride=1, activation="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None: out = self.se(out)

        out += residual

        out = self.relu(out)

        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation="leaky_relu",
                                activation_param=1e-3)
        if stride == 1:
            self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu",
                                    activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation="leaky_relu",
                                        activation_param=1e-3)
            else:
                self.conv2 = nn.Sequential(conv2d_ABN(planes, planes, kernel_size=3, stride=1,
                                                      activation="leaky_relu", activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1,
                                activation="identity")

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None: out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = self.relu(out)

        return out


class TResNet(Module):

    
    def __init__(self, layers, in_chans, pretrained, num_spec, num_gen, num_fam, width_factor=1.0,
                 do_bottleneck_head=False,bottleneck_features=512):
        super(TResNet, self).__init__()

        # JIT layers
        space_to_depth = SpaceToDepthModule() # TODO: remove
        anti_alias_layer = AntiAliasDownsampleLayer
        global_pool_layer = FastAvgPool2d(flatten=True)

        self.pretrained = pretrained
        self.num_spec = num_spec
        self.num_gen = num_gen
        self.num_fam = num_fam        
        
        # TResnet stages
        if pretrained == 'finetune':
            # convolves 4 band RGB-I down to 3 channels of 224x224 dimension
            self.conv4band = nn.Conv2d(4, 3, kernel_size=7, stride=1, padding=3) 
            in_chans = 3
            # initialize He-style
            nn.init.kaiming_normal_(self.conv4band.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.conv4band.bias, 0)
        elif pretrained == 'feat_ext':
            in_chans = 3
        
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 56x56
        layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 28x28
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 14x14
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
                                  anti_alias_layer=anti_alias_layer)  # 7x7

        #
        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', space_to_depth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)]))

        # head
        self.embeddings = []
        self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
        self.num_features = (self.planes * 8) * Bottleneck.expansion # expansion is just 4, magic number
        print("num features is ", self.num_features)
        # Ignore this for now, code will break for bottleneck_head but don't think will use
        if do_bottleneck_head:
            fc = bottleneck_head(self.num_features, num_classes,
                                 bottleneck_features=bottleneck_features)
        else:
            self.spec = nn.Linear(self.num_features, num_spec)
            self.gen = nn.Linear(self.num_features, num_gen)
            self.fam = nn.Linear(self.num_features, num_fam)        

        # get rid of this and instead put in place f, g, spec fcs
        # model initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
                                  activation="identity")]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(
            block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        
        if self.pretrained == 'finetune':
            x = self.conv4band(x)        
        elif self.pretrained == 'feat_ext':
            x = x[:,:3]
        # if self.pretrained != 'none':
            # below is if the data isn't being mean-centered. Now that it is being scaled to 0-1
            # then don't need this any more
            # x = x / 255.0 # from TResNet inference code ¯\_(ツ)_/¯
        x = self.body(x)
        self.embeddings = self.global_pool(x)
        spec = self.spec(self.embeddings)
        gen = self.gen(self.embeddings)
        fam = self.fam(self.embeddings)
        return (spec, gen, fam)
    
    
    
    

class Joint_TResNet(Module):

    
    def __init__(self, layers, in_chans, pretrained, num_spec, num_gen, num_fam, env_rasters, width_factor=1.0,
                 do_bottleneck_head=False,bottleneck_features=512):
        super(Joint_TResNet, self).__init__()

        # JIT layers
        space_to_depth = SpaceToDepthModule()
        anti_alias_layer = AntiAliasDownsampleLayer
        global_pool_layer = FastAvgPool2d(flatten=True)

        self.pretrained = pretrained
        self.num_spec = num_spec
        self.num_gen = num_gen
        self.num_fam = num_fam        
        self.env_rasters = env_rasters
        self.mlp_choke1 = 64
        self.mlp_choke2 = 128        
        # TResnet stages
        if pretrained == 'finetune':
            # convolves 4 band RGB-I down to 3 channels of 224x224 dimension
            self.conv4band = nn.Conv2d(4, 3, kernel_size=7, stride=1, padding=3) 
            in_chans = 3
            # initialize He-style
            nn.init.kaiming_normal_(self.conv4band.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.conv4band.bias, 0)
        elif pretrained == 'feat_ext':
            in_chans = 3
        
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 56x56
        layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 28x28
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 14x14
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
                                  anti_alias_layer=anti_alias_layer)  # 7x7

        #
        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', space_to_depth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)]))

        # head
        self.embeddings = []
        self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
        self.num_features = (self.planes * 8) * Bottleneck.expansion # expansion is just 4, magic number
        print("num features is ", self.num_features) 
        
        self.intermediate1 = nn.Sequential(
            nn.Linear(self.num_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
#             nn.Dropout(),
        )
        
        
        
        self.mlp = nn.Sequential(
            nn.Linear(env_rasters, self.mlp_choke1),
            nn.Linear(self.mlp_choke1, self.mlp_choke2),
            nn.Linear(self.mlp_choke2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
        )        
        
        self.choke = 2048
        self.intermediate2 = nn.Sequential(
            nn.Linear(2048 * 2, self.choke), # todo: try different powers of 2 here 
            nn.ReLU(True),
#             nn.Dropout() # may need to remove???
        )
        self.spec = nn.Linear(self.choke, num_spec)
        self.gen = nn.Linear(self.choke, num_gen)
        self.fam = nn.Linear(self.choke, num_fam)
  
        # model initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
                                  activation="identity")]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(
            block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x, rasters):
        
        if self.pretrained == 'finetune':
            x = self.conv4band(x)        
        elif self.pretrained == 'feat_ext':
            x = x[:,:3]
            # don't need below now that 0-1 centering everyting
        #if self.pretrained != 'none':
        #    x = x / 255.0 # from TResNet inference code ¯\_(ツ)_/¯
        x = self.body(x)
#         self.embeddings = self.global_pool(x)
        x = self.global_pool(x)
#         x = torch.flatten(self.embeddings, 1)
        x = torch.flatten(x, 1)
        
        x = self.intermediate1(x)
        rasters = self.mlp(rasters)
        x = torch.cat((x, rasters), dim=1)
        x = self.intermediate2(x)
        spec = self.spec(x)
        gen = self.gen(x)
        fam = self.fam(x)
        return (spec, gen, fam)


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False
        
def _tresnet(arch, layers, pretrained: str, num_spec : int, num_gen : int, num_fam : int, base_dir : str, width_factor : int
) -> TResNet:
    in_chans = 4
    model = TResNet(layers, in_chans, pretrained, num_spec, num_gen, num_fam, width_factor=width_factor)
    if pretrained != 'none':
        if arch == 'TResNetL':
            # going to be lazy and use load_state_dict_from_url, might change in the future
            dirr = config.setup_pretrained_dirs(base_dir) + 'TResNet/'
            file = dirr + model_files['MSCOCO_TResnetL']
            state = torch.load(file, map_location='cpu')
            model.load_state_dict(state['model'], strict=False)
        else:
            raise NotImplementedError('no pretrained model on disk for this architecture yet!')
        
    if pretrained == 'feat_ext':
        set_parameter_requires_grad(model.body)        
 
    return model

def _joint_tresnet(arch, layers, pretrained: str, num_spec : int, num_gen : int, num_fam : int, env_rasters : int, base_dir : str, width_factor : int
) -> Joint_TResNet:
    in_chans = 4
    model = Joint_TResNet(layers, in_chans, pretrained, num_spec, num_gen, num_fam, env_rasters, width_factor=width_factor)
    if pretrained != 'none':
        if arch == 'TResNetL':
            # going to be lazy and use load_state_dict_from_url, might change in the future
            dirr = config.setup_pretrained_dirs(base_dir) + 'TResNet/'
            file = dirr + model_files['MSCOCO_TResnetL']
            state = torch.load(file, map_location='cpu')
            model.load_state_dict(state['model'], strict=False)
        else:
            raise NotImplementedError('no pretrained model on disk for this architecture yet!')
        
    if pretrained == 'feat_ext':
        set_parameter_requires_grad(model.body)        
 
    return model

def Joint_TResNetM(pretrained, num_spec, num_gen, num_fam, env_rasters, base_dir):
    return _joint_tresnet('TResNetM', [3, 4, 11, 3], pretrained, num_spec, num_gen, num_fam, env_rasters, base_dir, width_factor=1.0)
    
def Joint_TResNetL(pretrained, num_spec, num_gen, num_fam, env_rasters, base_dir):
    return _joint_tresnet('TResNetL', [4, 5, 18, 3], pretrained, num_spec, num_gen, num_fam, env_rasters, base_dir, width_factor=1.2)



def TResnetM(pretrained, num_spec, num_gen, num_fam, base_dir):
    """Constructs a medium TResnet model.
    """
    
    return _tresnet('TResNetM', [3, 4, 11, 3], pretrained, num_spec, num_gen, num_fam, base_dir, width_factor=1.0)


def TResnetL(pretrained, num_spec, num_gen, num_fam, base_dir):
    """Constructs a large TResnet model.
    """
    return _tresnet('TResNetL', [4, 5, 18, 3], pretrained, num_spec, num_gen, num_fam, base_dir, width_factor=1.2)




def TResnetXL(pretrained, num_spec, num_gen, num_fam, base_dir):
    """Constructs a xlarge TResnet model.
    """
    return _tresnet('TResNetXL', [4, 5, 24, 3], pretrained, num_spec, num_gen, num_fam, base_dir, width_factor=1.3)


class AntiAliasDownsampleLayer(nn.Module):
    def __init__(self, remove_model_jit: bool = False, filt_size: int = 3, stride: int = 2,
                 channels: int = 0):
        super(AntiAliasDownsampleLayer, self).__init__()
        if not remove_model_jit:
            self.op = DownsampleJIT(filt_size, stride, channels)
        else:
            self.op = Downsample(filt_size, stride, channels)

    def forward(self, x):
        return self.op(x)


@torch.jit.script
class DownsampleJIT(object):
    def __init__(self, filt_size: int = 3, stride: int = 2, channels: int = 0):
        self.stride = stride
        self.filt_size = filt_size
        self.channels = channels

        assert self.filt_size == 3
        assert stride == 2
        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        # ah, there's the offending cuda, in filt!
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1)).half()

    def __call__(self, input: torch.Tensor):
        if input.dtype != self.filt.dtype:
            self.filt = self.filt.float() 
        self.filt = self.filt.to(input.device)
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=2, padding=0, groups=input.shape[1])


class Downsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels


        assert self.filt_size == 3
        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))

    def forward(self, input):
        self.filt = self.filt.to(input.device)
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])
    
    
    
class FastAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)    
        
        
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class SpaceToDepthModule(nn.Module):
    def __init__(self, remove_model_jit=False):
        super().__init__()
        if not remove_model_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    def forward(self, x):
        return self.op(x)


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


@torch.jit.script
class SpaceToDepthJit(object):
    def __call__(self, x: torch.Tensor):
        # assuming hard-coded that block_size==4 for acceleration
        N, C, H, W = x.size()
        x = x.view(N, C, H // 4, 4, W // 4, 4)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * 16, H // 4, W // 4)  # (N, C*bs^2, H//bs, W//bs)
        return x


class hard_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hard_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.add_(3.).clamp_(0., 6.).div_(6.)
        else:
            return F.relu6(x + 3.) / 6.


class SEModule(nn.Module):

    def __init__(self, channels, reduction_channels, inplace=True):
        super(SEModule, self).__init__()
        self.avg_pool = FastAvgPool2d()
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=inplace)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, padding=0, bias=True)
        # self.activation = hard_sigmoid(inplace=inplace)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se2 = self.fc1(x_se)
        x_se2 = self.relu(x_se2)
        x_se = self.fc2(x_se2)
        x_se = self.activation(x_se)
        return x * x_se        
