import torch
import torch.nn as nn
# import time

# Since nn.Sequential does not handle multiple inputs, create mySequential to
# handle it, inhiriting from nn.Sequential
class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class block_standard(nn.Module):
    #defining block expansion 
    expansion: int = 1
    # To devide the 'out_channels' by 2 in the standard, we create this variab.
    out_multiply: int = 2
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1, up=False):
        
        super(block_standard, self).__init__()
        
        self.up = up
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if self.up:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size=stride, stride=stride)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
    
    def forward(self, x, long_skip=None):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if (self.identity_downsample is not None):
            identity = self.identity_downsample(identity)
        # if long_skip==None: print('long skip none')
        if long_skip is not None:
            x = torch.cat((x, long_skip), dim=1)
        x += identity
        x = self.relu(x)
        
        del identity
        
        return x, long_skip


class block_bottleneck(nn.Module):
    # defining block expansion
    expansion: int = 4
    # To devide the 'out_channels' by 2 in the standard, we create this variab.
    out_multiply: int = 1
    
    def __init__(self, in_channels, out_channels, identity_scale=None, stride=1, up=False):
        
        super(block_bottleneck, self).__init__()
        
        self.up = up
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if self.up:
            self.expansion = 2
        else:
            self.expansion = 4
        
        if self.up:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size=stride, stride=stride)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        if self.up:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1)
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_scale = identity_scale
    
    def forward(self, x, long_skip=None):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_scale is not None:
            identity = self.identity_scale(identity)
        # if long_skip==None: print('long skip none')
        if long_skip is not None:
            x = torch.cat((x, long_skip), dim=1)
        x += identity
        x = self.relu(x)
        
        del identity
        
        return x, long_skip


class UResNet(nn.Module): # [3, 4, 6, 3]
    
    def __init__(self, block, layers, image_channels, num_classes):
        
        super(UResNet, self).__init__()
        self.in_channels = 64
        # First Convolutions
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7,
                               stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.long_skip = []
        
        # ResNet Layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)
        
        # ResNet Layers
        self.layer5 = self._make_layer(block, layers[3], out_channels=512, stride=2, up=True)
        self.layer6 = self._make_layer(block, layers[2], out_channels=256, stride=2, up=True)
        self.layer7 = self._make_layer(block, layers[1], out_channels=128, stride=2, up=True)
        self.layer8 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        
        # Last Convolutions
        self.conv_last1 = nn.ConvTranspose2d(self.in_channels, 64,
                                             kernel_size=2, stride=2, padding=0)
        self.conv_last2 = nn.ConvTranspose2d(64*2, num_classes,
                                             kernel_size=2, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.Softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        self.long_skip = [0, 0, 0, 0]
        self.long_skip[0] = x
        x = self.maxpool(x)
        
        x, temp = self.layer1(x, None)
        self.long_skip[1] = x
        x, temp = self.layer2(x, None)
        self.long_skip[2] = x
        x, temp = self.layer3(x, None)
        self.long_skip[3] = x
        x, temp = self.layer4(x, None)
        
        self.long_skip = self.long_skip[::-1]
        
        x, temp = self.layer5(x, self.long_skip[0])
        x, temp = self.layer6(x, self.long_skip[1])
        x, temp = self.layer7(x, self.long_skip[2])
        x, temp = self.layer8(x, None)
        x = self.conv_last1(x)
        x = self.bn1(x)
        x = torch.cat((x, self.long_skip[3]), dim=1)
        x = self.relu(x)
        x = self.conv_last2(x)
        x = self.bn2(x)
        x = self.Softmax(x)
        
        del self.long_skip, temp
        
        return x
    
    
    def _make_layer(self, block, num_residual_blocks,
                         out_channels, stride, up=False):
        identity_scale = None
        layers = []
        
        if up==False:
            if stride != 1 or self.in_channels != out_channels*block.expansion:
                identity_scale = mySequential(nn.Conv2d(self.in_channels,
                                                        out_channels*block.expansion,
                                                        kernel_size=1,
                                                        stride=stride),
                                              nn.BatchNorm2d(out_channels*block.expansion))
            
            layers.append(block(self.in_channels, out_channels,
                                identity_scale, stride))
            self.in_channels = out_channels*block.expansion
            
            for i in range(num_residual_blocks-1):
                layers.append(block(self.in_channels, out_channels))
        
        else:
            if stride != 1 or self.in_channels != out_channels*block.expansion:
                identity_scale = mySequential(nn.ConvTranspose2d(self.in_channels,
                                                                 out_channels*block.expansion,
                                                                 kernel_size=stride,
                                                                 stride=stride),
                                              nn.BatchNorm2d(out_channels*block.expansion))
            # This is to devide the 'out_channels' by 2 in the standard block
            # if you see, the output channels is really half of the value in
            # this block.
            out_channels = int(out_channels/block.out_multiply)
            layers.append(block(self.in_channels, out_channels,
                                identity_scale, stride, up=True))
            
            if stride==1 and up==True:
                self.in_channels = out_channels*2
            elif block.expansion == 1 and up==True:
                self.in_channels = out_channels*2
            else:
                self.in_channels = out_channels*4
            
            for i in range(num_residual_blocks-1):
                layers.append(block(self.in_channels, out_channels, up=True))
        
        return mySequential(*layers)
    
def UResNet18(in_channels=3, num_classes=10):
    return UResNet(block_standard, [2, 2, 2, 2], in_channels, num_classes)

def UResNet34(in_channels=3, num_classes=10):
    return UResNet(block_standard, [3, 4, 6, 3], in_channels, num_classes)

def UResNet50(in_channels=3, num_classes=1000):
    return UResNet(block_bottleneck, [3, 4, 6, 3], in_channels, num_classes)

def UResNet101(in_channels=3, num_classes=10):
    return UResNet(block_bottleneck, [3, 4, 23, 3], in_channels, num_classes)

def UResNet152(in_channels=3, num_classes=10):
    return UResNet(block_bottleneck, [3, 8, 36, 3], in_channels, num_classes)


def test():
    net = UResNet18()
    # working sizes 224, 256, 288
    x = torch.randn(2, 3, 224, 224)
    if torch.cuda.is_available():
        y = net(x).to('cuda')
    else:
        y = net(x)
    print(y.shape)

if __name__ == '__main__':
    test()

# print('- Time taken:', time.time()-start)


