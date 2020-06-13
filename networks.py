# necessary garage modules
import garage.torch.utils as tu
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from garage import wrap_experiment
from garage.experiment import LocalRunner
from garage.torch.algos import VPG, PPO, TRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import value_function
from rl_racer import *


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class CNN(nn.Module):
    '''Model courtesy of ECE271C'''
    def __init__(self, block, layers, out_dim, inplanes=10):
        self.inplanes = inplanes
        super(PlainNet, self).__init__()
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(inplanes * 4, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self, env_spec, name):
        super().__init__()
        self.cat_input_dim = env_spec.observation_space.flat_dim - 3*240*320 + 10
        self.env_spec = env_spec
        self.name = name
        self.reset_hidden_state()

    @property
    def cnn(self, block=BasicBlock, layers=[1, 1, 1], out_dim=10, in_planes=10):
        return CNN(block, layers, out_dim, in_planes)

    @property
    def lstm(self, hidden_dim=50, lstm_layers=2):
        output_dim = 2*self.env_spec.action_space.flat_dim
        lstm = nn.LSTM(self.cat_input_dim, hidden_dim, lstm_layers, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        return lstm

    def forward(self, state):
        assert isinstance(state, Observation)
        img = torch.tensor(Observation.img_rgb)
        img_features = self.cnn(img).flatten()
        # vector to concatenate with cnn feature vector
        cat_vector = torch.tensor(Observation.to_numpy_array().flatten())
        in_tensor = torch.cat(img_features, cat_vector)
        out = self.lstm(in_tensor, (self.h_0, self.c_0))
        params = self.fc(out)

        return params

    def reset_hidden_state(self, lstm_layers=2, batch_size=1):
        self.h_0 = torch.randn(lstm_layers, batch_size, self.hidden_dim)
        self.c_0 = torch.randn(lstm_layers, batch_size, self.hidden_dim)
