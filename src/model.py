import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1) #conv2x
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=2) #conv3x
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2) #conv4x
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2) #conv5xe

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

class MultiTaskHandNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MultiTaskHandNet, self).__init__()

        # backbone
        backbone = Backbone()

        # input layer for 4 channels rgbd
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # residual layers from backbone
        self.layer1 = backbone.layer1  # 56x56, 64channel
        self.layer2 = backbone.layer2  # 28x28, 128channel
        self.layer3 = backbone.layer3  # 14x14, 256channel
        self.layer4 = backbone.layer4  # 7x7,   512channel

        # classification & bbox heads
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.05)
        # classifier
        self.fc_class = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )
        # bbox
        self.fc_bbox = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

        # segmentation decoder with skip connections
        self.upsample1 = self._expand_block(512, 256)
        self.upsample2 = self._expand_block(512, 128)  # 256 + 256 
        self.upsample3 = self._expand_block(256, 64)   # 128 + 128 
        self.upsample4 = self._expand_block(128, 32)   # 64  + 64  
        self.final_mask = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)

    def _expand_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        # encoder
        x0 = self.relu(self.bn1(self.conv1(x)))
        x0_pool = self.maxpool(x0)
        x1 = self.layer1(x0_pool)  # 56x56
        x2 = self.layer2(x1)       # 28x28
        x3 = self.layer3(x2)       # 14x14
        x4 = self.layer4(x3)       # 7x7

        # classification & bbox
        pooled = self.avgpool(x4).view(x4.size(0), -1)
        pooled = self.dropout(pooled) 
        out_class = self.fc_class(pooled)
        out_bbox = self.fc_bbox(pooled)

        # segmentation decoder with skip connections
        d1 = self.upsample1(x4)
        d2 = self.upsample2(torch.cat([d1, x3], dim=1))
        d3 = self.upsample3(torch.cat([d2, x2], dim=1))
        d4 = self.upsample4(torch.cat([d3, x1], dim=1))
        out_mask = self.final_mask(d4)

        return {'class': out_class, 'bbox': out_bbox, 'mask': out_mask}

if __name__ == "__main__":
    model = MultiTaskHandNet(num_classes=10)
    x = torch.randn(2, 4, 224, 224)
    out = model(x)
    print("class:", out['class'].shape)  
    print("bbox: ", out['bbox'].shape)   
    print("mask: ", out['mask'].shape)   