import os.path

import torch
import torch.nn as nn
import math
import pickle
import torch.nn.functional as F
## id loss from emoca



def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, whose dimensions in the model are {} and whose ' \
                    'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, include_top=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        if not self.include_top:
            return x

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

class VGGFace2Loss(nn.Module):
    def __init__(self, pretrained_checkpoint_path=None, metric='cosine_similarity', trainable=False):
        super(VGGFace2Loss, self).__init__()
        self.reg_model = resnet50(num_classes=8631, include_top=False).eval()
        # checkpoint = pretrained_checkpoint_path or \
        #              '/ps/scratch/rdanecek/FaceRecognition/resnet50_ft_weight.pkl'
        checkpoint = os.path.join(os.path.dirname(__file__), 'resnet50_ft_weight.pkl')
        load_state_dict(self.reg_model, checkpoint)
        # this mean needs to be subtracted from the input images if using the model above
        self.register_buffer('mean_bgr', torch.tensor([91.4953, 103.8827, 131.0912]))

        self.trainable = trainable

        if metric is None:
            metric = 'cosine_similarity'

        if metric not in ["l1", "l1_loss", "l2", "mse", "mse_loss", "cosine_similarity",
                          "barlow_twins", "barlow_twins_headless"]:
            raise ValueError(f"Invalid metric for face recognition feature loss: {metric}")

        if metric == "barlow_twins_headless":
            # feature_size = self.reg_model.fc.in_features
            # self.bt_loss = BarlowTwinsLossHeadless(feature_size)
            raise ValueError
        elif metric == "barlow_twins":
            # feature_size = self.reg_model.fc.in_features
            # self.bt_loss = BarlowTwinsLoss(feature_size)
            raise ValueError
        else:
            self.bt_loss = None

        self.metric = metric

    def _get_trainable_params(self):
        params = []
        if self.trainable:
            params += list(self.reg_model.parameters())
        if self.bt_loss is not None:
            params += list(self.bt_loss.parameters())
        return params

    def train(self, b = True):
        if not self.trainable:
            ret = super().train(False)
        else:
            ret = super().train(b)
        if self.bt_loss is not None:
            self.bt_loss.train(b)
        return ret

    def requires_grad_(self, b):
        super().requires_grad_(False) # face recognition net always frozen
        if self.bt_loss is not None:
            self.bt_loss.requires_grad_(b)

    def freeze_nontrainable_layers(self):
        if not self.trainable:
            super().requires_grad_(False)
        else:
            super().requires_grad_(True)
        if self.bt_loss is not None:
            self.bt_loss.requires_grad_(True)

    def reg_features(self, x):
        # TODO: is this hard-coded margin necessary?
        margin = 10
        x = x[:, :, margin:224 - margin, margin:224 - margin]
        x = F.interpolate(x * 2. - 1., [224, 224], mode='bilinear')
        feature = self.reg_model(x)
        feature = feature.view(x.size(0), -1)
        return feature

    def transform(self, img):
        # input images in RGB in range [0-1] but the network expects them in BGR  [0-255] with subtracted mean_bgr
        img = img[:, [2, 1, 0], :, :].permute(0, 2, 3, 1) * 255 - self.mean_bgr
        img = img.permute(0, 3, 1, 2)
        return img

    def _cos_metric(self, x1, x2):
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, is_crop=True, batch_size=None, ring_size=None):
        gen = self.transform(gen)
        tar = self.transform(tar)

        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)

        if self.metric == "cosine_similarity":
            loss = self._cos_metric(gen_out, tar_out).mean()
        elif self.metric in ["l1", "l1_loss", "mae"]:
            loss = torch.nn.functional.l1_loss(gen_out, tar_out)
        elif self.metric in ["mse", "mse_loss", "l2", "l2_loss"]:
            loss = torch.nn.functional.mse_loss(gen_out, tar_out)
        elif self.metric in ["barlow_twins_headless", "barlow_twins"]:
            loss = self.bt_loss(gen_out, tar_out, batch_size=batch_size, ring_size=ring_size)
        else:
            raise ValueError(f"Invalid metric for face recognition feature loss: {self.metric}")

        return loss