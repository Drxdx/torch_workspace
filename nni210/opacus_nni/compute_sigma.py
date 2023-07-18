from opacus import PrivacyEngine
import torch
from nni.retiarii import fixed_arch
from data import get_data, random_sample_loader
from nni.nas.hub.pytorch import ENAS as EnasSpace
#from nni.nas.hub.pytorch import DARTS as DartsSpace
from nni.retiarii.hub.pytorch import DARTS as DARTSSpace
exported_arch = torch.load('/home/xuedaxuan/torch_workspace/nni210/exported_arch/enas_45notrick_oricls_reluconvbn_stdsep_all_5node_50_200.pth')
with fixed_arch(exported_arch):
    final_model = DARTSSpace(width=16, num_cells=6, dataset='cifar',auxiliary_loss=True, drop_path_prob=0.2)
model = final_model


train_data, test_data = get_data('cifar10', augment=False)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=2048, shuffle=True, num_workers=1, pin_memory=True)

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.7276715037905593),
    data_loader=train_loader,
    epochs=30,
    target_epsilon=3,
    target_delta=1e-5,
    max_grad_norm=1,
)

print(f"Using sigma={optimizer.noise_multiplier}")
