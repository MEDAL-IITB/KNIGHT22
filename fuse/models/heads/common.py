"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

from typing import Optional, Sequence
import torch.nn as nn
import torch
import pdb

class ClassifierFCN(nn.Module):
    def __init__(self, in_ch: int, num_classes: Optional[int], layers_description: Sequence[int]=(256,), dropout_rate: float = 0.1):
        super().__init__()
        layer_list = []
        layer_list.append(nn.Conv2d(in_ch, layers_description[0], kernel_size=1, stride=1))
        layer_list.append(nn.ReLU())
        if dropout_rate is not None and dropout_rate > 0:
            layer_list.append(nn.Dropout(p=dropout_rate))
        last_layer_size = layers_description[0]
        for curr_layer_size in layers_description[1:]:
            layer_list.append(nn.Conv2d(last_layer_size, curr_layer_size, kernel_size=1, stride=1))
            layer_list.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))
            last_layer_size = curr_layer_size

        if num_classes is not None:
            layer_list.append(nn.Conv2d(last_layer_size, num_classes, kernel_size=1, stride=1))
        
        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.classifier(x)
        return x

class ClassifierMLP(nn.Module):
    def __init__(self, in_ch: int, num_classes: Optional[int], layers_description: Sequence[int]=(256,), dropout_rate: float = 0.1):
        super().__init__()
        layer_list = []
        fc_layer = []
        # conv_layer_list = []
        # conv_layer_list.append(nn.Conv1d(in_ch,4,kernel_size=3,padding='same'))
        # conv_layer_list.append(nn.ReLU())
        # conv_layer_list.append(nn.Conv1d(4,8,kernel_size=3,padding='same'))
        # conv_layer_list.append(nn.ReLU())
        # conv_layer_list.append(nn.Flatten())

        fc_layer.append(nn.Linear(in_ch,128))
        fc_layer.append(nn.ReLU())
        fc_layer.append(nn.Linear(128,256))
        fc_layer.append(nn.ReLU())
        # layer_list.append(nn.Flatten())
        layer_list.append(nn.Linear(in_ch, layers_description[0]))
        layer_list.append(nn.ReLU())
        if dropout_rate is not None and dropout_rate > 0:
            layer_list.append(nn.Dropout(p=dropout_rate))
        last_layer_size = layers_description[0]
        for curr_layer_size in layers_description[1:]:
            layer_list.append(nn.Linear(last_layer_size, curr_layer_size))
            layer_list.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))
            last_layer_size = curr_layer_size
        
        if num_classes is not None:
            layer_list.append(nn.Linear(last_layer_size, num_classes))
        # self.conv = nn.Sequential(*conv_layer_list)
        self.fc = nn.Sequential(*fc_layer)
        self.classifier = nn.Sequential(*layer_list)
        self.softmax = nn.Softmax(0)
        # self.mm = torch.mm

    def forward(self, x):
        # x = self.fc(x)
        # print("1",x.shape)
        x_orig = x
        x = torch.mm(torch.transpose(x, 0, 1),x)
        x = self.softmax(x)
        x = torch.mm(x_orig,x)
        x = x + x_orig
        # print("2",x.shape)
        x = self.classifier(x)
        # print("3",x.shape)
        return x

# m = ClassifierMLP(21,2)
# input = torch.randn(1,1,21)
# output = m(input)
