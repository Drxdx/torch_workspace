from nni.retiarii.hub.pytorch import ENAS as EnasSpace
import nni
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from nni.retiarii.evaluator.pytorch import DataLoader
from nni.retiarii.strategy import ENAS as EnasStrategy
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii import fixed_arch
import numpy as np
from nni.retiarii.evaluator.pytorch import Classification
from torch.utils.data import SubsetRandomSampler
from nni.retiarii.evaluator.pytorch import ClassificationModule
from nni.retiarii.evaluator.pytorch import Lightning, Trainer
from opacus.lightning import DPLightningDataModule
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

fast_dev_run = False

#单次NAS尽管计算效率很高，但也存在许多缺点。我们推荐权重共享神经结构搜索
strategy = EnasStrategy(reward_metric_name='val_acc')

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
    width=16,           # the initial filters (channel number) for the model
    num_cells=6,        # the number of stacked cells in total
    dataset='cifar'     # to give a hint about input resolution, here is 32x32
)

class DartsClassificationModule(ClassificationModule):
    def __init__(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 0.,
        auxiliary_loss_weight: float = 0.4,
        max_epochs: int = 600
    ):
        self.auxiliary_loss_weight = auxiliary_loss_weight
        # Training length will be used in LR scheduler
        self.max_epochs = max_epochs
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, export_onnx=False)

    def configure_optimizers(self):
        """Customized optimizer with momentum, as well as a scheduler."""
        optimizer = torch.optim.SGD(
            self.parameters(),
            momentum=0.9,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-3)
        }

    def training_step(self, batch, batch_idx):
        """Training step, customized with auxiliary loss."""
        x, y = batch
        if self.auxiliary_loss_weight:
            y_hat, y_aux = self(x)
            loss_main = self.criterion(y_hat, y)
            loss_aux = self.criterion(y_aux, y)
            self.log('train_loss_main', loss_main)
            self.log('train_loss_aux', loss_aux)
            loss = loss_main + self.auxiliary_loss_weight * loss_aux
        else:
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log('train_' + name, metric(y_hat, y), prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        # Set drop path probability before every epoch. This has no effect if drop path is not enabled in model.
        self.model.set_drop_path_prob(self.model.drop_path_prob * self.current_epoch / self.max_epochs)

        # Logging learning rate at the beginning of every epoch
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

# evaluator = Classification(
#     learning_rate=1e-3,
#     weight_decay=1e-4,
#     train_dataloaders=search_train_loader,
#     val_dataloaders=search_valid_loader,
#     max_epochs=1,
#     gpus=1,
#     fast_dev_run=fast_dev_run,
# )
max_epochs = 1
evaluator = Lightning(
    DartsClassificationModule(0.025, 3e-4, 0., max_epochs),
    Trainer(
        gpus=1,
        max_epochs=max_epochs,
        fast_dev_run=fast_dev_run,
    ),
    train_dataloaders=search_train_loader,
    val_dataloaders=search_valid_loader
)

config = RetiariiExeConfig(execution_engine='oneshot')
experiment = RetiariiExperiment(model_space, evaluator=evaluator, strategy=strategy)
experiment.run(config)

exported_arch = experiment.export_top_models()[0]
print("#####################arch###########################")
print(exported_arch)

