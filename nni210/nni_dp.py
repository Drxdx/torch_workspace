import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
from opacus import PrivacyEngine
from torchvision import models


import nni.retiarii.strategy as strategy
search_strategy = strategy.Random(dedup=True)  # dedup=False if deduplication is not wanted
from opacus.validators import ModuleValidator

from opacus.layers import DPLSTM
# darts_v2_model = DARTS.load_searched_model('darts-v2', pretrained=True, download=True)
# print(next(resnet18.parameters()).device)
# model = resnet18.cuda()
# print(next(resnet18.parameters()).device)
# darts_conv = darts_v2_model.stages[0][0].preprocessor.pre0[1]
#darts_conv = darts_v2_model.stem[1]

from opacus.grad_sample import GradSampleModule
import nni
from copy import deepcopy
from collections import OrderedDict
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
from nni.retiarii.hub.pytorch import DARTS,ProxylessNAS

resnet18 = models.resnet18(num_classes=10)
P_model = ProxylessNAS.load_searched_model('acenas-m1', pretrained=True, download=True)
P_fix_model = ModuleValidator.fix(P_model, strict=False)
p_conv = P_model.stem[0]
p_bn = P_model.stem[1]
print(P_model)
print("="*50+'NAS'+'=='*50)
print(p_bn)
print(p_conv,p_bn)
print(ModuleValidator.validate(p_conv, strict=False))
print(ModuleValidator.validate(p_bn, strict=False))
print(ModuleValidator.fix(p_bn))
# print("                            ")
# print("="*50+'resnet'+'=='*50)
# print(resnet18.conv1,resnet18.bn1)
# a = ModuleValidator.validate(resnet18.conv1, strict=False)
# b= ModuleValidator.validate(resnet18.bn1, strict=False)
# c = ModuleValidator.validate(ModuleValidator.fix(resnet18.bn1),strict=False)
# print(a)
# print(b)
# print(c)


import nni.retiarii.nn.pytorch as nn
from nni.retiarii.nn.pytorch.nn import functional as F
import torch.optim

#darts_v2_model = DARTS.load_searched_model('darts-v2', pretrained=True, download=True)
# darts_v2_model = ModuleValidator.fix(P_model, strict=False)
# print(darts_v2_model)


#print(P_model)
#print(ModuleValidator.fix(resnet18, strict=False))
#print(P_model.__class__)
# resnet_conv = resnet18.conv1
#resnet_bn = resnet18.bn1
#re = resnet18.layer1[0].conv1

# p_conv = P_model.stem[0]
# p_bn = P_model.stem[1]
# print(ModuleValidator.validate(P_model, strict=False))
# print(ModuleValidator.FIXERS)
# print(type(p_bn))

# fn = ModuleValidator.FIXERS[type(p_bn)]
# print(fn(p_bn))
# print(ModuleValidator.fix(p_bn))
# print(ModuleValidator.fix(P_model, strict=False))
# print(p_bn, type(p_bn),ModuleValidator.FIXERS)
# print(type(p_bn) in ModuleValidator.FIXERS)


# print(ModuleValidator.FIXERS)
# print(type(p_bn) in ModuleValidator.FIXERS)

#print(isinstance(p_bn, nn.modules.batchnorm.BatchNorm2d))


#resnet181 = ModuleValidator.fix(P_model, strict=False)
#print(resnet181)
#print("======", len(ModuleValidator.validate(resnet181)))
#print(ModuleValidator.is_valid(resnet181))


# errors = ModuleValidator.validate(resnet18, strict=False)
# print(errors)
# resnet18 = ModuleValidator.fix(P_model, strict=False)
#print(resnet18)

# #print(ModuleValidator.is_valid(model))
# #model = GradSampleModule(model)





