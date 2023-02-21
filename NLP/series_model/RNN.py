import  torch
import datetime
import  numpy as np
import  torch.nn as nn
import  torch.optim as optim
from    matplotlib import pyplot as plt

###########################设置全局变量##################################
num_time_steps = 16    
input_size = 3 
hidden_size = 16
output_size = 3
num_layers = 1
lr=0.01
####################定义RNN类##############################################
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Net, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        for p in self.rnn.parameters():
          nn.init.normal_(p, mean=0.0, std=0.001)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        # [b, seq, h]
        out = out.view(-1, hidden_size)
        out = self.linear(out)#[seq,h] => [seq,3]
        out = out.unsqueeze(dim=0)  # => [1,seq,3]
        return out, hidden_prev

#####################开始训练模型#################################
def tarin_RNN(data):
    model = Net(input_size, hidden_size, num_layers)
    print('model:\n',model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    #初始化h
    hidden_prev = torch.zeros(1, 1, hidden_size)
    l = []
    # 训练3000次
    for iter in range(1):
        # loss = 0
        start = np.random.randint(10, size=1)[0]
        end = start + 15
        x = torch.tensor(data[start:end]).float().view(1, num_time_steps - 1, 3)

        # 在data里面随机选择15个点作为输入，预测第16
        y = torch.tensor(data[start + 5:end + 5]).float().view(1, num_time_steps - 1, 3)
        output, hidden_prev = model(x, hidden_prev)
        hidden_prev = hidden_prev.detach()

        loss = criterion(output, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print("Iteration: {} loss {}".format(iter, loss.item()))
            l.append(loss.item())
    ##############################绘制损失函数#################################
    # plt.plot(l,'r')
    # plt.xlabel('训练次数')
    # plt.ylabel('loss')
    # plt.title('RNN损失函数下降曲线')
    return hidden_prev,model

####################初始化训练集#################################
def getdata():
    x1 = np.linspace(1,10,30).reshape(30,1)
    print(f'x1 shape: {x1.shape}')
    y1 = (np.zeros_like(x1)+2)+np.random.rand(30,1)*0.1
    z1 = (np.zeros_like(x1)+2).reshape(30,1)
    tr1 =  np.concatenate((x1,y1,z1),axis=1)
    # mm = MinMaxScaler()
    # data = mm.fit_transform(tr1)   #数据归一化
    return tr1
    
def demo():
    data = getdata()
    # print(data)
    start = datetime.datetime.now()
    hidden_pre, model = tarin_RNN(data)
    # torch.save(model, './test_rnn.pth')
    end = datetime.datetime.now()
    print('The training time: %s' % str(end - start))
    # plt.show()

if __name__ == '__main__':
    demo()