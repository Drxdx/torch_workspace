import nni
import numpy as np
from nni.retiarii.evaluator.pytorch import Classification
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10
from nni.retiarii.evaluator.pytorch import DataLoader


from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
#from nni.retiarii.strategy.rl  import PolicyBasedRL
from nni.retiarii import fixed_arch

from nni.retiarii.strategy import ENAS as ENAStrategy
from nni.retiarii.hub.pytorch import ENAS as ENASpace

from nni.retiarii.strategy import DARTS as DartsStrategy
from nni.retiarii.hub.pytorch import DARTS as DartsSpace

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

# fast_dev_run = True
fast_dev_run = False

#strategy = DartsStrategy()
strategy = ENAStrategy(reward_metric_name='val_acc')

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
    train_data, batch_size=64, num_workers=6,
    sampler=SubsetRandomSampler(indices[:split]),
)

search_valid_loader = DataLoader(
    train_data, batch_size=64, num_workers=6,
    sampler=SubsetRandomSampler(indices[split:]),
)

evaluator = Classification(
    learning_rate=1e-3,
    weight_decay=1e-4,
    train_dataloaders=search_train_loader,
    val_dataloaders=search_valid_loader,
    # max_epochs=10,
    max_epochs=2,
    gpus=1,
    fast_dev_run=fast_dev_run,
)

# model_space = DartsSpace(
#     width=16,           # the initial filters (channel number) for the model
#     num_cells=8,        # the number of stacked cells in total
#     dataset='cifar'     # to give a hint about input resolution, here is 32x32
# )

model_space = ENASpace(
    width=16,           # the initial filters (channel number) for the model
    num_cells=6,        # the number of stacked cells in total
    dataset='cifar'     # to give a hint about input resolution, here is 32x32
)



config = RetiariiExeConfig(execution_engine='oneshot')
experiment = RetiariiExperiment(model_space, evaluator=evaluator, strategy=strategy)
experiment.run(config)
# trategy.run()
exported_arch = experiment.export_top_models('dict')[0]
print(exported_arch)
# strategy.export_top_models()


#
with fixed_arch(exported_arch):
    # final_model = DartsSpace(width=16, num_cells=8, dataset='cifar')
    final_model = ENASpace(width=16, num_cells=8, dataset='cifar')
train_loader = DataLoader(train_data, batch_size=96, num_workers=6)  # Use the original training data


max_epochs = 2

evaluator = Classification(
    learning_rate=1e-3,
    weight_decay=1e-4,
    train_dataloaders=train_loader,
    val_dataloaders=search_valid_loader,
    max_epochs=max_epochs,
    gpus=1,
    export_onnx=False,          # Disable ONNX export for this experiment
    fast_dev_run=fast_dev_run   # Should be false for fully training
)

evaluator.fit(final_model)


