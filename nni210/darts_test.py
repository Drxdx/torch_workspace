from nni.retiarii.hub.pytorch import DARTS as DartsSpace
from nni.retiarii.hub.pytorch import ENAS
import nni
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from nni.retiarii.evaluator.pytorch import DataLoader
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#################################################数据层面##########################################
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
model_space = DartsSpace(
    width=16,           # the initial filters (channel number) for the model
    num_cells=8,        # the number of stacked cells in total
    dataset='cifar'     # to give a hint about input resolution, here is 32x32
)

fast_dev_run = False#fast_dev_run设置为False以重现我们声称的结果。否则，将只运行几个小批量

import numpy as np
from nni.retiarii.evaluator.pytorch import Classification
from torch.utils.data import SubsetRandomSampler

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

train_data = nni.trace(CIFAR10)(root='/home/xuedaxuan/torch_workspace/pt.darts/data', train=True, download=True, transform=transform)

num_samples = len(train_data)
indices = np.random.permutation(num_samples)
split = num_samples // 2

search_train_loader = DataLoader(
    train_data, batch_size=128, num_workers=6,
    sampler=SubsetRandomSampler(indices[:split]),
)

search_valid_loader = DataLoader(
    train_data, batch_size=128, num_workers=6,
    sampler=SubsetRandomSampler(indices[split:]),
)

#################################################评估器##########################################
evaluator = Classification(#Evaluator that is used for classification.
    learning_rate=1e-3,
    weight_decay=1e-4,
    train_dataloaders=search_train_loader,
    val_dataloaders=search_valid_loader,
    max_epochs=10,
    gpus=1,
    fast_dev_run=fast_dev_run,
)

#################################################搜索策略##########################################
from nni.retiarii.strategy import DARTS as DartsStrategy
#单次NAS尽管计算效率很高，但也存在许多缺点。我们推荐权重共享神经结构搜索
strategy = DartsStrategy()

#################################################开始训练##########################################
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig

config = RetiariiExeConfig(execution_engine='oneshot')
experiment = RetiariiExperiment(model_space, evaluator=evaluator, strategy=strategy)
experiment.run(config)

exported_arch = experiment.export_top_models()[0]
print("#####################arch###########################")
print(exported_arch)
#torch.save(exported_arch, '/home/xuedaxuan/torch_workspace/nni210/exported_arch/exported_arch.pth')
print("################################################")

#########################################Retrain the searched model#########################################################################
# from nni.retiarii import fixed_arch
#
# with fixed_arch(exported_arch):
#     final_model = DartsSpace(width=16, num_cells=8, dataset='cifar')
#
# print("####################model############################")
# print(final_model)
# print("################################################")
# train_loader = DataLoader(train_data, batch_size=96, num_workers=6)  # Use the original training data
# transform_valid = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
# ])
# valid_data = nni.trace(CIFAR10)(root='/home/xuedaxuan/torch_workspace/pt.darts/data', train=False, download=True, transform=transform_valid)
# valid_loader = DataLoader(valid_data, batch_size=256, num_workers=1)
#
# max_epochs = 1
#
# evaluator = Classification(
#     learning_rate=1e-3,
#     weight_decay=1e-4,
#     train_dataloaders=train_loader,
#     val_dataloaders=valid_loader,
#     max_epochs=max_epochs,
#     gpus=1,
#     export_onnx=False,          # Disable ONNX export for this experiment
#     fast_dev_run=fast_dev_run   # Should be false for fully training
# )
#
# evaluator.fit(final_model)
# loss=0.989, v_num=10, train_loss=1.090, train_acc=0.587, val_loss=1.090, val_acc=0.608
#print("################################################")