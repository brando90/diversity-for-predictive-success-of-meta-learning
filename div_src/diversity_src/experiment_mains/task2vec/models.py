# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


import torch.utils.model_zoo as model_zoo

import torchvision.models.resnet as resnet
import torch
from torch import nn
from task2vec import ProbeNetwork

_MODELS = {}


def _add_model(model_fn):
    _MODELS[model_fn.__name__] = model_fn
    return model_fn



class FNN3(ProbeNetwork):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_layers=[15, 15],
            #hidden_layer1=15,
            #hidden_layer2=15,
    ):
        super().__init__()

        assert len(hidden_layers) >= 2, "Need at least 2 hidden layers"

        # Start of our FNN: input -> hidden_layer[0]

        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(1,128)
        self.bn1= nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(128,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(128,output_size)

        self.layers = [
            self.f1,
            self.l1,
            self.bn1,
            self.relu1,
            self.l2,
            self.bn2,
            self.relu2,
            self.fc
        ]

        '''
        self.l1 = nn.Flatten()
        self.l2 = nn.Linear(input_size, hidden_layers[0])
        self.l3 =nn.BatchNorm1d(hidden_layers[0])
        self.l4 =  nn.ReLU()
        self.layers = [
            self.l1,
            self.l2,
            self.l3,
            self.l4
        ]
        layer_cnt = 5
        # Intermediate layers
        for i in range(len(hidden_layers)-1):
            ldict = {}
            ldict['linlay'] = nn.Linear(hidden_layers[i], hidden_layers[i+1])
            ldict['self'] = self
            exec("self.l" + str(layer_cnt) + " = linlay",globals(),ldict)
            linlayer = ldict['self.l' + str(layer_cnt)]
            layer_cnt+=1
            exec("self.l" + str(layer_cnt) + " = nn.BatchNorm1d(hidden_layers[i+1])",globals(),ldict)
            bnlayer = ldict['self.l' + str(layer_cnt)]
            layer_cnt+=1
            exec("self.l" + str(layer_cnt) + " = nn.ReLU()",globals(),ldict)
            relulayer = ldict['self.l' + str(layer_cnt)]
            layer_cnt+=1
            layer = [linlayer, bnlayer, relulayer]
            #exec("layer = [self.l" + str(layer_cnt-3) + ", self.l" + str(layer_cnt-2) + ", self.l" + str(layer_cnt-3) + "]")
            #layer = [
            #    nn.Linear(hidden_layers[i], hidden_layers[i+1]),
            #    nn.BatchNorm1d(hidden_layers[i+1]),
            #    nn.ReLU()
            #]
            (self.layers).extend(layer)
        self.fc = nn.Linear(hidden_layers[-1], output_size)
        '''
        '''self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.BatchNorm1d(hidden_layers[1]),
            nn.ReLU(),
        )
        self.clsfier = nn.Linear(hidden_layers[1], output_size)
        '''
    @property
    def classifier(self):
        return self.fc


    def forward(self, x,start_from=0):
        #x = x.view(-1,1).float()
        for layer in self.layers[start_from:]:
            x = layer(x)
        return x


class ResNet(resnet.ResNet, ProbeNetwork):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        # Saves the ordered list of layers. We need this to forward from an arbitrary intermediate layer.
        self.layers = [
            self.conv1, self.bn1, self.relu,
            self.maxpool, self.layer1, self.layer2,
            self.layer3, self.layer4, self.avgpool,
            lambda z: torch.flatten(z, 1), self.fc
        ]

    @property
    def classifier(self):
        return self.fc

    # @ProbeNetwork.classifier.setter
    # def classifier(self, val):
    #     self.fc = val

    # Modified forward method that allows to start feeding the cached activations from an intermediate
    # layer of the network
    def forward(self, x, start_from=0):
        """Replaces the default forward so that we can forward features starting from any intermediate layer."""
        for layer in self.layers[start_from:]:
            x = layer(x)
        return x

@_add_model
def resnet18(pretrained=False, num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model: ProbeNetwork = ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(resnet.model_urls['resnet18'])
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model

@_add_model
def gaussian_net(num_classes=5,pretrained=False,path_to_ckpt=None):
    model: ProbeNetwork = FNN3(input_size=1,output_size=num_classes, hidden_layers=[128,128,128,128,128,128,128,128])
    if pretrained:
        print("Loading pretrained!!", path_to_ckpt)
        ckpt = torch.load(path_to_ckpt)#, map_location=args.device)
        #print(ckpt['model_state_dict'].keys())
        cmsd = ckpt['model_state_dict']
        #need to remap keys
        modded_dict = {'l1.weight':cmsd['features.1.weight'],
                       'l1.bias':cmsd['features.1.bias'],
                       'bn1.weight':cmsd['features.2.weight'],
                       "bn1.bias":cmsd['features.2.bias'],
                       "bn1.running_mean":cmsd['features.2.running_mean'],
                       "bn1.running_var":cmsd['features.2.running_var'],
                       "l2.weight":cmsd['features.4.weight'],
                       "l2.bias":cmsd['features.4.bias'],
                       "bn2.weight":cmsd['features.5.weight'],
                       "bn2.bias":cmsd['features.5.bias'],
                       "bn2.running_mean":cmsd['features.5.running_mean'],
                       "bn2.running_var":cmsd['features.5.running_var'],
                       "fc.weight":cmsd['classifier.weight'],
                       "fc.bias":cmsd["classifier.bias"]}
        model.load_state_dict(modded_dict)
    return model
    #return model


@_add_model
def resnet34(pretrained=False, num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(resnet.model_urls['resnet34'])
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model


def get_model(model_name, pretrained=False, num_classes=1000):
    try:
        return _MODELS[model_name](pretrained=pretrained, num_classes=num_classes)
    except KeyError:
        raise ValueError(f"Architecture {model_name} not implemented.")
