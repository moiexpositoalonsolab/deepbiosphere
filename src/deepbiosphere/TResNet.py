# torch functions
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.nn import Module as Module

# misc functions
import numpy as np
from enum import Enum
from collections import OrderedDict
from inplace_abn import InPlaceABN
import deepbiosphere.Utils as utils
from types import SimpleNamespace


'''
pulled down from github https://github.com/Alibaba-MIIL/TResNet on 1/9/2021
'''

# ---------- Types ---------- #


PRETRAINED_MODELS = SimpleNamespace(
    MSCOCO_TResnetL = 'MS_COCO_TRresNet_L_448_86.6.pth')

class Pretrained(Enum, metaclass=utils.MetaEnum):
    NONE = None
    FEAT_EXT = 'feat_ext'
    FINETUNE = 'finetune'
    
class Architecture(Enum, metaclass=utils.MetaEnum):
    TRESNETM = 'TResnetM'
    TRESNETL = 'TResnetL'
    TRESNETXL = 'TResnetXL'
    
# ---------- helper methods ---------- #
    
def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

# ---------- ResNet components ---------- #
        
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

# ---------- Remote Sensing-only CNN ---------- #

class RS_TResNet(Module):


    def __init__(self, layers, in_chans, num_spec, num_gen, num_fam, pretrained, width_factor=1.0,
                 do_bottleneck_head=False,bottleneck_features=512, encode=False):
        super(RS_TResNet, self).__init__()

        # JIT layers
        space_to_depth = SpaceToDepthModule() 
        anti_alias_layer = AntiAliasDownsampleLayer
        global_pool_layer = FastAvgPool2d(flatten=True)

        self.pretrained = Pretrained[pretrained]
        self.encode = encode
        self.num_spec = num_spec
        self.num_gen = num_gen
        self.num_fam = num_fam

        # TResnet stages
        if self.pretrained is Pretrained.FINETUNE:
            # convolves 4 band RGB-I down to 3 channels of 224x224 dimension
            self.conv4band = nn.Conv2d(utils.NAIP_CHANS, utils.IMAGENET_CHANS, kernel_size=7, stride=1, padding=3)
            in_chans = utils.IMAGENET_CHANS
            # initialize He-style
            nn.init.kaiming_normal_(self.conv4band.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.conv4band.bias, 0)
        elif self.pretrained is Pretrained.FEAT_EXT:
            in_chans = utils.IMAGENET_CHANS

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
        # Ignore this for now, code will break for bottleneck_head but don't think will use
        if do_bottleneck_head:
            fc = bottleneck_head(self.num_features, num_classes,
                                 bottleneck_features=bottleneck_features)
        else:
            self.spec = nn.Linear(self.num_features, num_spec)
            if (self.num_gen != -1):
                self.gen = nn.Linear(self.num_features, num_gen)
            if (self.num_fam != -1):
                self.fam = nn.Linear(self.num_features, num_fam)

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

        if self.pretrained is Pretrained.FINETUNE:
            x = self.conv4band(x)
        elif self.pretrained is Pretrained.FEAT_EXT:
            x = x[:,:utils.IMAGENET_CHANS]

        x = self.body(x)
        self.embeddings = self.global_pool(x)
        # if we're encoding, we want
        # everything in forward() except output layers
        if self.encode:
            return self.embeddings
        # TODO: in eval mode, only return species
        spec = self.spec(self.embeddings)
        if not self.training: # TODO: make sure inference et al handles this!
            return spec
        # use all 3 taxonomic levels for training
        if (self.num_gen != -1) & (self.num_fam != -1):
            gen = self.gen(self.embeddings)
            fam = self.fam(self.embeddings)
            return (spec, gen, fam)
        # use only species genus taxonomic levels
        elif (self.num_fam != -1):
            gen = self.gen(self.embeddings)
            return (spec, gen)
        # use only species taxonomic level
        else:
            return spec

def _tresnet(arch, layers, num_spec : int, num_gen : int, num_fam : int, pretrained: str, base_dir : str, width_factor : int
) -> RS_TResNet:
    model = RS_TResNet(layers, utils.NAIP_CHANS, num_spec, num_gen, num_fam, pretrained, width_factor=width_factor)
    
    # type check
    pretrained = Pretrained[pretrained]
    arch = Architecture[arch]
    if pretrained is not Pretrained.NONE:
        if arch is Architecture.TRESNETL:
            # going to be lazy and use load_state_dict_from_url, might change in the future
            dirr = utils.setup_pretrained_dirs(base_dir) + 'TResNet/'
            file = dirr + model_files['MSCOCO_TResnetL']
            state = torch.load(file, map_location='cpu')
            model.load_state_dict(state['model'], strict=False)
        else:
            raise NotImplementedError('no pretrained model on disk for this architecture yet!')

    if pretrained is Pretrained.FEAT_EXT:
        set_parameter_requires_grad(model.body)

    return model

def TResnetM(num_spec, num_gen, num_fam, pretrained,  base_dir):
    """Constructs a medium TResnet model.
    """

    return _tresnet('TRESNETM', [3, 4, 11, 3], num_spec, num_gen, num_fam, pretrained, base_dir, width_factor=1.0)


def TResnetL(num_spec, num_gen, num_fam, pretrained,  base_dir):
    """Constructs a large TResnet model.
    """
    return _tresnet('TRESNETL', [4, 5, 18, 3], num_spec, num_gen, num_fam, pretrained, base_dir, width_factor=1.2)

def TResnetXL(num_spec, num_gen, num_fam, pretrained,  base_dir):
    """Constructs a xlarge TResnet model.
    """
    return _tresnet('TRESNETXL', [4, 5, 24, 3], num_spec, num_gen, num_fam, pretrained, base_dir, width_factor=1.3)
        
# ---------- Remote Sensing + climate CNN ---------- #

class Joint_TResNet(Module):


    def __init__(self, layers, in_chans, num_spec, num_gen, num_fam, env_rasters, pretrained, nlayers=4, drop=.25, width_factor=1.0,
                 do_bottleneck_head=False,bottleneck_features=512, encode=False):
        super(Joint_TResNet, self).__init__()

        # JIT layers
        space_to_depth = SpaceToDepthModule()
        anti_alias_layer = AntiAliasDownsampleLayer
        global_pool_layer = FastAvgPool2d(flatten=True)

        self.pretrained = Pretrained[pretrained]
        self.encode = encode
        self.num_spec = num_spec
        self.num_gen = num_gen
        self.num_fam = num_fam
        self.env_rasters = env_rasters
        self.mlp_choke1 = 1000
        self.mlp_choke2 = 2000
        self.unification = 2048
        self.elu = nn.ELU()
        
        # setting up pretrained models
        if self.pretrained is Pretrained.FINETUNE:
            # convolves 4 band RGB-I down to 3 channels of 224x224 dimension
            self.conv4band = nn.Conv2d(utils.NAIP_CHANS, utils.IMAGENET_CHANS, kernel_size=7, stride=1, padding=3)
            in_chans = utils.IMAGENET_CHANS
            # initialize He-style
            nn.init.kaiming_normal_(self.conv4band.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.conv4band.bias, 0)
        elif self.pretrained is Pretrained.FEAT_EXT:
            in_chans = utils.IMAGENET_CHANS

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

        # set up ResNet
        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', space_to_depth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)]))

        # ResNet head
        self.embeddings = []
        self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
        self.num_features = (self.planes * 8) * Bottleneck.expansion # expansion is just 4, magic number
        # downscale CNN predictions to same size as MLP predictions
        self.intermediate1 = nn.Sequential(
            nn.Linear(self.num_features, self.unification),
            nn.BatchNorm1d(self.unification),
            nn.ReLU(),
        )
        # same MLP as used for training with just bioclim rasters
        layers = []
        layers.append(nn.Linear(env_rasters, self.mlp_choke1))
        layers.append(self.elu)
        # smaller layers
        for i in range(1, (nlayers//2)):
            layers.append(nn.Linear(self.mlp_choke1, self.mlp_choke1))
            layers.append(self.elu)
        # setup to bigger layers
        layers.append(nn.Linear(self.mlp_choke1, self.mlp_choke2))
        layers.append(self.elu)
        # and dropout
        layers.append(nn.Dropout(drop))
        # bigger layers
        for i in range((nlayers//2)+1, nlayers):
            layers.append(nn.Linear(self.mlp_choke2, self.mlp_choke2))
            layers.append(self.elu)
        layers.append(nn.Linear(self.mlp_choke2, self.unification))
        layers.append(nn.BatchNorm1d(self.unification))
        layers.append(self.elu)
        self.mlp = nn.Sequential(*layers)
        # mixing of predictions of CNN head and MLP head
        self.intermediate2 = nn.Sequential(
            nn.Linear(self.unification * 2, self.unification),
            nn.ReLU(),
        )
        # finally, make predictions
        self.spec = nn.Linear(self.unification, num_spec)
        if (self.num_gen != -1):
            self.gen = nn.Linear(self.unification, num_gen)
        if (self.num_fam != -1):
            self.fam = nn.Linear(self.unification, num_fam)

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

    def forward(self, X):
        # necessary to reuse code for training
        # other CNNs. Input is a tuple with the first
        # input being the images and the second the raster values
        x, rasters = X
        if self.pretrained is Pretrained.FINETUNE:
            x = self.conv4band(x)
        elif self.pretrained is Pretrained.FEAT_EXT:
            x = x[:,:utils.IMAGENET_CHANS]
        x = self.body(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.intermediate1(x)
        rasters = self.mlp(rasters)
        # concatenate RS, clim embeddings
        x = torch.cat((x, rasters), dim=1)
        x = self.intermediate2(x)
        
        # if we're encoding, we want
        # everything in forward() except output layers
        if self.encode:
            return x
        
        # use all 3 taxonomic levels for training
        spec = self.spec(x)
        if not self.training: # TODO: make sure inference et al handles this!
            return spec
        if (self.num_gen != -1) & (self.num_fam != -1):
            gen = self.gen(x)
            fam = self.fam(x)
            return (spec, gen, fam)
        # use only species genus taxonomic levels
        elif (self.num_fam != -1):
            gen = self.gen(x)
            return (spec, gen)
        # use only species taxonomic level
        # still pass same way as others during training
        else:
            return (spec)



def _joint_tresnet(arch, layers, num_spec : int, num_gen : int, num_fam : int, env_rasters : int, pretrained: str, base_dir : str, width_factor : int
) -> Joint_TResNet:

    model = Joint_TResNet(layers, utils.NAIP_CHANS, num_spec, num_gen, num_fam, env_rasters, pretrained, width_factor=width_factor)

    pretrained = Pretrained[pretrained]
    arch = Architecture[arch]
    if pretrained is not Pretrained.NONE: 
        if arch is Architecture.TRESNETL:
            # going to be lazy and use load_state_dict_from_url, might change in the future
            dirr = utils.setup_pretrained_dirs(base_dir) + 'TResNet/'
            file = dirr + model_files['MSCOCO_TResnetL']
            state = torch.load(file, map_location='cpu')
            model.load_state_dict(state['model'], strict=False)
        else:
            # TODO: get TResNetM pretrained weights off github
            raise NotImplementedError('no pretrained model on disk for this architecture yet!')

    if pretrained is Pretrained.FEAT_EXT:
        set_parameter_requires_grad(model.body)

    return model

def Joint_TResNetM(num_spec, num_gen, num_fam, env_rasters, pretrained,  base_dir):
    return _joint_tresnet('TRESNETM', [3, 4, 11, 3], num_spec, num_gen, num_fam, env_rasters, pretrained, base_dir, width_factor=1.0)

def Joint_TResNetL(num_spec, num_gen, num_fam, env_rasters, pretrained,  base_dir):
    return _joint_tresnet('TRESNETL', [4, 5, 18, 3], num_spec, num_gen, num_fam, pretrained, env_rasters, base_dir, width_factor=1.2)


# -------- TResNet modules ------------- #


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
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])



class FastAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        in_size = x.size()
        if self.flatten:
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
# this is the problem becuase of the flatten=false
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
