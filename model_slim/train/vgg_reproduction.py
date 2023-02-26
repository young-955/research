# %%
import torch
import torch.nn as nn

def conv_bn_relu(input_num, output_num, kernel_size = 3, padding = 1):
    return nn.Sequential(
        nn.Conv2d(input_num, output_num, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(output_num),
        nn.ReLU(inplace=True)
    )

class VGG(nn.Module):
    def __init__(self, model_type = 'A', input_size = (224, 224), output_size = 1000) -> None:
        super(VGG, self).__init__()
        self.model_type = model_type
        self.input_size = input_size
        if self.model_type == 'A':
            self.feature = self.feature_A
        if self.model_type == 'B':
            self.feature = self.feature_B
        if self.model_type == 'C':
            self.feature = self.feature_C
        if self.model_type == 'D':
            self.feature = self.feature_D
        if self.model_type == 'E':
            self.feature = self.feature_E
        self.conv3_64_bn_relu = conv_bn_relu(3, 64)
        self.conv64_64_bn_relu = conv_bn_relu(64, 64)
        self.conv64_128_bn_relu = conv_bn_relu(64, 128)
        self.conv128_128_bn_relu = conv_bn_relu(128, 128)
        self.conv128_256_bn_relu = conv_bn_relu(128, 256)
        self.conv256_256_bn_relu = conv_bn_relu(256, 256)
        self.conv256_256_1_bn_relu = conv_bn_relu(256, 256, 1)
        self.conv256_512_bn_relu = conv_bn_relu(256, 512)
        self.conv512_512_bn_relu = conv_bn_relu(512, 512)
        self.conv512_512_1_bn_relu = conv_bn_relu(512, 512, 1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.average_pool = nn.AdaptiveAvgPool2d((7, 7))
        test_ipt = torch.autograd.Variable(torch.zeros(1, 3, input_size[0], input_size[1]))
        test_out = self.feature(test_ipt)
        print(test_out.shape)
        self.fc1 = nn.Linear(test_out.size(1), 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, output_size)
        self.dropout = nn.Dropout()

    def classifier(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

    # 11
    def feature_A(self, x):
        x = self.conv3_64_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv64_128_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv128_256_bn_relu(x)
        x = self.conv256_256_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv256_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        # x = self.average_pool(x)
        x = torch.flatten(x, 1)
        return x
        
    # 13
    def feature_B(self, x):
        x = self.conv3_64_bn_relu(x)
        x = self.conv64_64_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv64_128_bn_relu(x)
        x = self.conv128_128_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv128_256_bn_relu(x)
        x = self.conv256_256_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv256_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        # x = self.average_pool(x)
        x = torch.flatten(x, 1)
        return x
        
    # 16
    def feature_C(self, x):
        x = self.conv3_64_bn_relu(x)
        x = self.conv64_64_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv64_128_bn_relu(x)
        x = self.conv128_128_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv128_256_bn_relu(x)
        x = self.conv256_256_bn_relu(x)
        x = self.conv256_256_1_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv256_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.conv512_512_1_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.conv512_512_1_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        # x = self.average_pool(x)
        x = torch.flatten(x, 1)
        return x
        
    # 16 D is better than C
    def feature_D(self, x):
        x = self.conv3_64_bn_relu(x)
        x = self.conv64_64_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv64_128_bn_relu(x)
        x = self.conv128_128_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv128_256_bn_relu(x)
        x = self.conv256_256_bn_relu(x)
        x = self.conv256_256_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv256_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        # x = self.average_pool(x)
        x = torch.flatten(x, 1)
        return x
        
    # 19
    def feature_E(self, x):
        x = self.conv3_64_bn_relu(x)
        x = self.conv64_64_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv64_128_bn_relu(x)
        x = self.conv128_128_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv128_256_bn_relu(x)
        x = self.conv256_256_bn_relu(x)
        x = self.conv256_256_bn_relu(x)
        x = self.conv256_256_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv256_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.conv512_512_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        # x = self.average_pool(x)
        x = torch.flatten(x, 1)
        return x
        
# %%
m = VGG(model_type='D')
input = torch.rand((1, 3, 224, 224))
output = m.forward(input)

# %%
output.shape
