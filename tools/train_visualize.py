import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data

from test_model import mymodel
import hiddenlayer as hid

# 训练可视化
# 执行命令
# tensorboard --logdir='./data/log'

mm = mymodel()

train_data = torchvision.datasets.MNIST(root='./data/MNIST', train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), download=False)
train_loader = data.DataLoader(dataset=train_data, batch_size=128, shuffle=False, num_workers=0)
test_data = torchvision.datasets.MNIST(root='./data/MNIST', train=False, download=False)

test_data_x = test_data.data.type(torch.FloatTensor)
test_data_x = torch.unsqueeze(test_data_x, dim=1)
test_data_y = test_data.targets

from tensorboardX import SummaryWriter
logger = SummaryWriter(log_dir='./data/log')

opt = torch.optim.Adam(mm.parameters(), lr=1e-4)
losf = nn.CrossEntropyLoss()

for e in range(2):
    for step, (x, y) in enumerate(train_loader):
        # a = x.convert('RGB')
        # print(a.shape)
        # print(x.shape)
        p = mm(x)
        l = losf(p, y)
        opt.zero_grad()
        l.backward()
        opt.step()
        global_step = step + e * len(train_loader) + 1
        if global_step % 10 == 0:
            logger.add_scalar('train_loss', l.item(), global_step=global_step)

