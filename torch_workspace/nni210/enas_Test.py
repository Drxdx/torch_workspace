from nni.retiarii.hub.pytorch import ENAS as EnasSpace
from nni.retiarii.hub.pytorch import DARTS as DARTSSpace
import nni
import torch
from nni.retiarii.evaluator.pytorch import Lightning, Trainer
from torchvision import transforms
from torchvision.datasets import CIFAR10
from nni.retiarii.evaluator.pytorch import DataLoader
from nni.retiarii.strategy import ENAS as EnasStrategy
from nni.retiarii.strategy import DARTS as DartsStrategy
from nni.retiarii.evaluator.pytorch import ClassificationModule
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii import fixed_arch
import numpy as np
from nni.retiarii.oneshot.pytorch import EnasTrainer
from nni.retiarii.evaluator.pytorch import Classification
from torch.utils.data import SubsetRandomSampler
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

fast_dev_run = False

#单次NAS尽管计算效率很高，但也存在许多缺点。我们推荐权重共享神经结构搜索
strategy = EnasStrategy(reward_metric_name='val_acc')
#strategy = DartsStrategy()
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

search_train_loader = DataLoader(train_data, batch_size=128, num_workers=6,sampler=SubsetRandomSampler(indices[:split]),)
search_valid_loader = DataLoader(train_data, batch_size=128, num_workers=6,sampler=SubsetRandomSampler(indices[split:]),)

model_space = EnasSpace(
    width=16,           # 16-the initial filters (channel number) for the model
    num_cells=6,        # the number of stacked cells in total
    dataset='cifar'     # to give a hint about input resolution, here is 32x32
)
evaluator = Classification(
    learning_rate=1e-3,
    weight_decay=1e-4,
    train_dataloaders=search_train_loader,
    val_dataloaders=search_valid_loader,
    max_epochs=1,
    gpus=1,
    fast_dev_run=fast_dev_run,

)

config = RetiariiExeConfig(execution_engine='oneshot')
experiment = RetiariiExperiment(model_space, evaluator=evaluator, strategy=strategy)
experiment.run(config)

exported_arch = experiment.export_top_models()[0]
print("#####################arch###########################")
print(exported_arch)
torch.save(exported_arch, '/home/xuedaxuan/torch_workspace/nni210/exported_arch/enas_notrick_reduction_10StdAct_reluconvbn_stdsep_all_5node_50_200.pth')
print("################################################")

#########################################Retrain the searched model#########################################################################

# exported_arch = torch.load('/home/xuedaxuan/torch_workspace/nni210/exported_arch/enas_notrick_reduction_10StdAct_reluconvbn_stdsep_all_5node_50_200.pth')
# with fixed_arch(exported_arch):
#     final_model = EnasSpace(width=16, num_cells=6, dataset='cifar')
#     print(final_model)
#
# train_loader = DataLoader(train_data, batch_size=96, num_workers=6)  # Use the original training data
# transform_valid = transforms.Compose([transforms.ToTensor(),transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])
# valid_data = nni.trace(CIFAR10)(root='/home/xuedaxuan/torch_workspace/pt.darts/data', train=False, download=True, transform=transform_valid)
# valid_loader = DataLoader(valid_data, batch_size=256, num_workers=1)
# max_epochs = 100
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
# print("#####################end!!###########################")


###########trick#########

# from nni.retiarii.hub.pytorch import ENAS as EnasSpace
# from nni.retiarii.hub.pytorch import DARTS as DARTSSpace
# import nni
# import torch
# from torchvision import transforms
# from nni.retiarii.evaluator.pytorch import ClassificationModule
# from torchvision.datasets import CIFAR10
# from nni.retiarii.evaluator.pytorch import DataLoader
# from nni.retiarii.evaluator.pytorch import Lightning, Trainer
# from nni.retiarii.strategy import ENAS as EnasStrategy
# from nni.retiarii.strategy import DARTS as DartsStrategy
# from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
# from nni.retiarii import fixed_arch
# import numpy as np
# from nni.retiarii.evaluator.pytorch import Classification
# from torch.utils.data import SubsetRandomSampler
# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
#
# CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
# CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
#
# fast_dev_run = False
#
# # strategy = DartsStrategy(gradient_clip_val=5.)
# strategy = EnasStrategy(reward_metric_name='val_acc')
#
# transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
# ])
#
# train_data = nni.trace(CIFAR10)(root='/home/xuedaxuan/torch_workspace/pt.darts/data', train=True, download=True, transform=transform)
#
# num_samples = len(train_data)
# indices = np.random.permutation(num_samples)
# split = num_samples // 2
#
# search_train_loader = DataLoader(train_data, batch_size=64, num_workers=6,sampler=SubsetRandomSampler(indices[:split]),)
# search_valid_loader = DataLoader(train_data, batch_size=64, num_workers=6,sampler=SubsetRandomSampler(indices[split:]),)
#
# model_space = EnasSpace(
#     width=16,           # 16-the initial filters (channel number) for the model
#     num_cells=6,        # the number of stacked cells in total
#     dataset='cifar'     # to give a hint about input resolution, here is 32x32
# )
#
# class DartsClassificationModule(ClassificationModule):
#     def __init__(
#         self,
#         learning_rate: float = 0.001,
#         weight_decay: float = 0.,
#         auxiliary_loss_weight: float = 0.4,
#         max_epochs: int = 600
#     ):
#         self.auxiliary_loss_weight = auxiliary_loss_weight
#         # Training length will be used in LR scheduler
#         self.max_epochs = max_epochs
#         super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, export_onnx=False)
#
#     def configure_optimizers(self):
#         """Customized optimizer with momentum, as well as a scheduler."""
#         optimizer = torch.optim.SGD(
#             self.parameters(),
#             momentum=0.9,
#             lr=self.hparams.learning_rate,
#             weight_decay=self.hparams.weight_decay
#         )
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-3)
#         }
#
#
#     def training_step(self, batch, batch_idx):
#         """Training step, customized with auxiliary loss."""
#         x, y = batch
#         if self.auxiliary_loss_weight:
#             y_hat, y_aux = self(x)
#             loss_main = self.criterion(y_hat, y)
#             loss_aux = self.criterion(y_aux, y)
#             self.log('train_loss_main', loss_main)
#             self.log('train_loss_aux', loss_aux)
#             loss = loss_main + self.auxiliary_loss_weight * loss_aux
#         else:
#             y_hat = self(x)
#             loss = self.criterion(y_hat, y)
#         self.log('train_loss', loss, prog_bar=True)
#         for name, metric in self.metrics.items():
#             self.log('train_' + name, metric(y_hat, y), prog_bar=True)
#         return loss
#
#     def on_train_epoch_start(self):
#         # Set drop path probability before every epoch. This has no effect if drop path is not enabled in model.
#         self.model.set_drop_path_prob(self.model.drop_path_prob * self.current_epoch / self.max_epochs)
#
#         # Logging learning rate at the beginning of every epoch
#         self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
# # evaluator = Classification(
# #     learning_rate=1e-3,
# #     weight_decay=1e-4,
# #     train_dataloaders=search_train_loader,
# #     val_dataloaders=search_valid_loader,
# #     max_epochs=10,
# #     gpus=1,
# #     fast_dev_run=fast_dev_run,
# #)
# max_epochs = 50
# evaluator = Lightning(
#     DartsClassificationModule(0.025, 3e-4, 0., max_epochs),
#     Trainer(
#         gpus=1,
#         max_epochs=max_epochs,
#         fast_dev_run=fast_dev_run,
#     ),
#     train_dataloaders=search_train_loader,
#     val_dataloaders=search_valid_loader
# )
# config = RetiariiExeConfig(execution_engine='oneshot')
# experiment = RetiariiExperiment(model_space, evaluator=evaluator, strategy=strategy)
# experiment.run(config)
# exported_arch = experiment.export_top_models()[0]
# torch.save(exported_arch, '/home/xuedaxuan/torch_workspace/nni210/exported_arch/enas_notrick_oricls_reluconvbn_stdsep_all_5node_50_200.pth')

########################################Retrain the searched model#########################################################################
# def cutout_transform(img, length: int = 16):
#     h, w = img.size(1), img.size(2)
#     mask = np.ones((h, w), np.float32)
#     y = np.random.randint(h)
#     x = np.random.randint(w)
#
#     y1 = np.clip(y - length // 2, 0, h)
#     y2 = np.clip(y + length // 2, 0, h)
#     x1 = np.clip(x - length // 2, 0, w)
#     x2 = np.clip(x + length // 2, 0, w)
#
#     mask[y1: y2, x1: x2] = 0.
#     mask = torch.from_numpy(mask)
#     mask = mask.expand_as(img)
#     img *= mask
#     return img
# exported_arch = torch.load('/home/xuedaxuan/torch_workspace/nni210/exported_arch/darts_48channel_trick_changeAdaptcls_oricls_reluconvbn_stdsep_all_5node_50_200.pth')
# with fixed_arch(exported_arch):
#     ####这句话######3####这句话######3####这句话######3####这句话######3####这句话######3####这句话######3####这句话######3#
#     print("enter in")
#     #final_model = DARTSSpace(width=16, num_cells=5, dataset='cifar', auxiliary_loss=True, drop_path_prob=0.2)
#     final_model = DARTSSpace(width=16, num_cells=5, dataset='cifar')
#     print("enter off")
# ###这句话######3####这句话######3####这句话######3####这句话######3####这句话######3####这句话######3####这句话######3####这句话######3#
# from torchviz import make_dot
# x = torch.randn(1, 3, 32, 32)
# # 前向传播并生成计算图
# y = final_model(x)
# make_dot(y, params=dict(final_model.named_parameters())).render("cnn_model", format="png")
#
# transform_with_cutout = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
#     cutout_transform,
# ])
# train_data_cutout = nni.trace(CIFAR10)(root='/home/xuedaxuan/torch_workspace/pt.darts/data', train=True, download=True, transform=transform_with_cutout)
# train_loader_cutout = DataLoader(train_data_cutout, batch_size=96)
# transform_valid = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
# ])
# valid_data = nni.trace(CIFAR10)(root='/home/xuedaxuan/torch_workspace/pt.darts/data', train=False, download=True, transform=transform_valid)
# valid_loader = DataLoader(valid_data, batch_size=256, num_workers=1)
# #
# max_epochs = 200
# evaluator = Lightning(
#     DartsClassificationModule(0.025, 3e-4, 0.4, max_epochs),
#     trainer=Trainer(
#         gpus=1,
#         gradient_clip_val=5.,
#         max_epochs=max_epochs,
#         fast_dev_run=fast_dev_run
#     ),
#     train_dataloaders=train_loader_cutout,
#     val_dataloaders=valid_loader,
# )
# evaluator.fit(final_model)

