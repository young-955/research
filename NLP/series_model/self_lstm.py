import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, layer_num, input_size=1, hidden_layer_size=100, output_size=2):
        super().__init__()
        # self.embedding = torch.nn.embedding(embedding.size(0), embedding.size(1)
        # self.embedding_dim = embedding.size(1)
        
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=layer_num)

        self.classifier =nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(hidden_layer_size, output_size),
                                        nn.Sigmoid())
        
    def forward(self, inputs):
        # inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最后一个的 hidden state（我的理解是最后一个的输出对于整个文本的理解是最到位的）
        x = x[:, -1, :] 
        x = self.classifier(x)
        return x