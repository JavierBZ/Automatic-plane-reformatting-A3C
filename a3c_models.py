import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from a3c_utils import norm_col_init, weights_init, weights_init_mlp


"""
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch

__all__ = ['xception']

model_urls = {
    'xception':'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x



class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=3):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()


        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(9, 32, 3,2, 0, bias=False)
        #self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        #self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        #self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        #self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # dropout

        self.drop_out = nn.Dropout(p=0.2)

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------





    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        if self.training:
          x = self.drop_out(x)
        x = self.conv3(x)
        #x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        #x = self.bn4(x)
        x = self.relu(x)



        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def xception(pretrained=False,**kwargs):
    """
    Construct Xception.
    """

    model = Xception(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model



class A3C_CONV_new(torch.nn.Module):
    def __init__(self, num_channels, num_actions):
        super(A3C_CONV_new, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=32, kernel_size=(3,8,8), stride = (1,2,2))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,8,8), stride=(1,2,2))
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,4,4), stride=(1,2,2))
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1))

        #self.lstm = nn.LSTMCell(576, 128)
        #self.drop1 = nn.Drop
        self.linear = nn.Linear(576, 128)

        num_outputs = num_actions
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)
        self.actor_linear2 = nn.Linear(128, num_outputs)

        self.relu = nn.ReLU()
        # self.apply(weights_init)
        # lrelu_gain = nn.init.calculate_gain('leaky_relu')
        # self.conv1.weight.data.mul_(lrelu_gain)
        # self.conv2.weight.data.mul_(lrelu_gain)
        # self.conv3.weight.data.mul_(lrelu_gain)
        # self.conv4.weight.data.mul_(lrelu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)

        x = self.relu(self.linear(x))

        return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x)

class A3C_CONV(torch.nn.Module):
    def __init__(self, num_channels, num_actions,out_feats=128,layers=4):
        super(A3C_CONV, self).__init__()
        # self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=16, kernel_size=(5,8,8), stride = (1,2,2))
        # self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(4,8,8), stride=(1,2,2))
        # self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,4,4), stride=(1,2,2))
        # self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1))
        #
        # self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=32, kernel_size=(3,8,8), stride = (1,2,2))
        # self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,8,8), stride=(1,2,2))
        # self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,4,4), stride=(1,2,2))
        # self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1))
        self.layers = layers
        if self.layers==4:
            self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=(8,8), stride = (2,2))
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(8,8), stride=(2,2))
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4,4), stride=(2,2))
            self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1))

            self.drop1 = nn.Dropout(0.4)
            # self.drop2 = nn.Dropout(0.3)
            # self.drop3 = nn.Dropout(0.3)
            # self.drop4 = nn.Dropout(0.3)

            #self.linear = nn.Linear(5760,512)
            #self.drop4 = nn.Dropout(0.4)

            #self.lstm = nn.LSTMCell(512, 128)
            self.lstm = nn.LSTMCell(576, out_feats)
            # self.conv_model = xception(num_classes = 512)
            # self.lstm = nn.LSTMCell(512, 128)
            num_outputs = num_actions
            self.drop5 = nn.Dropout(0.4)
            self.critic_linear = nn.Linear(out_feats, 1)
            self.actor_linear = nn.Linear(out_feats, num_outputs)
            self.actor_linear2 = nn.Linear(out_feats, num_outputs)

            self.relu = nn.ReLU()

        else:
            self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=(8,8), stride = (2,2))
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(8,8), stride=(2,2))
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4,4), stride=(2,2))
            self.drop1 = nn.Dropout(0.4)
            # self.drop2 = nn.Dropout(0.3)
            # self.drop3 = nn.Dropout(0.3)
            # self.drop4 = nn.Dropout(0.3)

            #self.linear = nn.Linear(5760,512)
            #self.drop4 = nn.Dropout(0.4)

            #self.lstm = nn.LSTMCell(512, 128)
            self.lstm = nn.LSTMCell(1600, out_feats)
            # self.conv_model = xception(num_classes = 512)
            # self.lstm = nn.LSTMCell(512, 128)
            num_outputs = num_actions
            self.drop5 = nn.Dropout(0.4)
            self.critic_linear = nn.Linear(out_feats, 1)
            self.actor_linear = nn.Linear(out_feats, num_outputs)
            self.actor_linear2 = nn.Linear(out_feats, num_outputs)

            self.relu = nn.ReLU()


        # self.apply(weights_init)
        # lrelu_gain = nn.init.calculate_gain('leaky_relu')
        # self.conv1.weight.data.mul_(lrelu_gain)
        # self.conv2.weight.data.mul_(lrelu_gain)
        # self.conv3.weight.data.mul_(lrelu_gain)
        # self.conv4.weight.data.mul_(lrelu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x, (hx, cx) = inputs

        # x = inputs
        # hx = torch.zeros(2,128)
        # cx = torch.zeros(2,128)
        #
        x = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))

        if self.layers == 4:
            x = self.relu(self.conv4(x))


        if self.training:
            x = self.drop1(x)
        #x = self.drop1(x)
        #
        # x = self.conv_model(x)
        x = x.view(x.size(0), -1)

            #print("Estoy en train dropout!!")
        # x = self.linear(x)
        # if self.training:
        #     x = self.drop4(x)

        hx, cx = self.lstm(x, (hx, cx))
        if self.training:
            x = self.drop5(hx)
        else:
            x = hx


        return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx)
        #return self.critic_linear(x), F.softmax(self.actor_linear(x),dim=1), self.actor_linear2(x), (hx, cx)


# RED NACHA

# class A3C_CONV(torch.nn.Module):
#     def __init__(self, num_channels, num_actions):
#         super(A3C_CONV, self).__init__()
#         # self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=16, kernel_size=(5,8,8), stride = (1,2,2))
#         # self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(4,8,8), stride=(1,2,2))
#         # self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,4,4), stride=(1,2,2))
#         # self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1))
#
#         self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=32, kernel_size=(5,15,15), stride = (2,5,5))
#         self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,7,7), stride=(2,3,3))
#         self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(2,4,4), stride=(1,2,2))
#         self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(2,3,3), stride=(1,1,1))
#
#
#         self.drop1 = nn.Dropout(0.3)
#
#         #self.linear = nn.Linear(5760,512)
#         #self.drop4 = nn.Dropout(0.4)
#
#         #self.lstm = nn.LSTMCell(512, 128)
#         #self.lstm = nn.LSTMCell(576, 128)
#         self.lstm = nn.LSTMCell(2048, 512)
#         # self.conv_model = xception(num_classes = 512)
#         # self.lstm = nn.LSTMCell(512, 128)
#         num_outputs = num_actions
#         self.drop5 = nn.Dropout(0.5)
#         self.critic_linear = nn.Linear(512, 1)
#         self.actor_linear = nn.Linear(512, num_outputs)
#         self.actor_linear2 = nn.Linear(512, num_outputs)
#
#         self.relu = nn.ReLU()
#
#
#         self.actor_linear.weight.data = norm_col_init(
#             self.actor_linear.weight.data, 0.01)
#         self.actor_linear.bias.data.fill_(0)
#         self.actor_linear2.weight.data = norm_col_init(
#             self.actor_linear2.weight.data, 0.01)
#         self.actor_linear2.bias.data.fill_(0)
#         self.critic_linear.weight.data = norm_col_init(
#             self.critic_linear.weight.data, 1.0)
#         self.critic_linear.bias.data.fill_(0)
#
#         self.lstm.bias_ih.data.fill_(0)
#         self.lstm.bias_hh.data.fill_(0)
#
#         self.train()
#
#     def forward(self, inputs):
#         x, (hx, cx) = inputs
#
#         #print('AA',x.shape)
#         x = self.relu(self.conv1(x))
#         #print('AA',x.shape)
#         x = self.relu(self.conv2(x))
#         #print('AA',x.shape)
#         x = self.relu(self.conv3(x))
#         #print('AA',x.shape)
#         x = self.relu(self.conv4(x))
#         # print('AA',x.shape)
#
#         if self.training:
#             x = self.drop1(x)
#
#         x = x.view(x.size(0), -1)
#
#
#         hx, cx = self.lstm(x, (hx, cx))
#         if self.training:
#             x = self.drop5(hx)
#         else:
#             x = hx
#
#         return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx)
#
#
#
class A3C_CONV_2(torch.nn.Module):
    def __init__(self, num_channels, num_actions):
        super(A3C_CONV_2, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=16, kernel_size=(5,8,8), stride = (1,2,2))
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(4,8,8), stride=(1,2,2))
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,4,4), stride=(1,2,2))
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1))

        # self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=32, kernel_size=(3,8,8), stride = (1,2,2))
        # self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,8,8), stride=(1,2,2))
        # self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,4,4), stride=(1,2,2))
        # self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1))



        self.drop1 = nn.Dropout(0.3)
        # self.drop2 = nn.Dropout(0.3)
        # self.drop3 = nn.Dropout(0.3)
        # self.drop4 = nn.Dropout(0.3)

        self.linear = nn.Linear(5760,512)
        self.drop4 = nn.Dropout(0.4)

        self.lstm = nn.LSTMCell(512, 128)
        #self.lstm = nn.LSTMCell(576, 128)
        # self.conv_model = xception(num_classes = 512)
        # self.lstm = nn.LSTMCell(512, 128)
        num_outputs = num_actions
        self.drop5 = nn.Dropout(0.5)
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)
        self.actor_linear2 = nn.Linear(128, num_outputs)

        self.relu = nn.ReLU()
        # self.apply(weights_init)
        # lrelu_gain = nn.init.calculate_gain('leaky_relu')
        # self.conv1.weight.data.mul_(lrelu_gain)
        # self.conv2.weight.data.mul_(lrelu_gain)
        # self.conv3.weight.data.mul_(lrelu_gain)
        # self.conv4.weight.data.mul_(lrelu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x, (hx, cx) = inputs

        # x = inputs
        # hx = torch.zeros(2,128)
        # cx = torch.zeros(2,128)
        #
        x = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))

        x = self.relu(self.conv4(x))


        if self.training:
            x = self.drop1(x)
        #x = self.drop1(x)
        #
        # x = self.conv_model(x)
        x = x.view(x.size(0), -1)

            #print("Estoy en train dropout!!")
        x = self.linear(x)
        if self.training:
            x = self.drop4(x)

        x = self.relu(x)

        hx, cx = self.lstm(x, (hx, cx))
        if self.training:
            x = self.drop5(hx)
        else:
            x = hx


        return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx)
        #return self.critic_linear(x), F.softmax(self.actor_linear(x),dim=1), self.actor_linear2(x), (hx, cx)



# class A3C_CONV(torch.nn.Module):
#     def __init__(self, num_channels, num_actions):
#         super(A3C_CONV, self).__init__()
#         # self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=32, kernel_size=(3,8,8), stride = (1,2,2))
#         # self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,8,8), stride=(1,2,2))
#         # self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,4,4), stride=(1,2,2))
#         # self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1))
#
#         self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=16, kernel_size=(3,8,8), stride = (1,4,4))
#         self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3,3,3), stride=(1,2,2))
#         self.conv3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3,3,3), stride=(1,1,1))
#
#         self.drop1 = nn.Dropout(0.3)
#         self.linear1 = nn.Linear(2400, 512)
#
#         self.drop2 = nn.Dropout(0.5)
#         self.lstm = nn.LSTMCell(512, 128)
#         num_outputs = num_actions
#
#         self.critic_linear = nn.Linear(128, 1)
#         self.actor_linear = nn.Linear(128, num_outputs)
#         self.actor_linear2 = nn.Linear(128, num_outputs)
#
#         self.relu = nn.ReLU()
#         # self.apply(weights_init)
#         # lrelu_gain = nn.init.calculate_gain('leaky_relu')
#         # self.conv1.weight.data.mul_(lrelu_gain)
#         # self.conv2.weight.data.mul_(lrelu_gain)
#         # self.conv3.weight.data.mul_(lrelu_gain)
#         # self.conv4.weight.data.mul_(lrelu_gain)
#
#         self.actor_linear.weight.data = norm_col_init(
#             self.actor_linear.weight.data, 0.01)
#         self.actor_linear.bias.data.fill_(0)
#         self.actor_linear2.weight.data = norm_col_init(
#             self.actor_linear2.weight.data, 0.01)
#         self.actor_linear2.bias.data.fill_(0)
#         self.critic_linear.weight.data = norm_col_init(
#             self.critic_linear.weight.data, 1.0)
#         self.critic_linear.bias.data.fill_(0)
#
#         self.lstm.bias_ih.data.fill_(0)
#         self.lstm.bias_hh.data.fill_(0)
#
#         self.train()
#
#     def forward(self, inputs):
#         x, (hx, cx) = inputs
#
#         # x = inputs
#         # hx = torch.zeros(2,128)
#         # cx = torch.zeros(2,128)
#
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         #x = self.relu(self.conv4(x))
#         #print(x.shape)
#         x = x.view(x.size(0), -1)
#         #print(x.shape)
#         if self.training:
#             x = self.drop1(x)
#         x = self.relu(self.linear1(x))
#         if self.training:
#             x = self.drop2(x)
#         hx, cx = self.lstm(x, (hx, cx))
#         # if self.training:
#         #     x = self.drop2(hx)
#         # else:
#         #     x = hx
#         x = hx
#
#
#         return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx)
#         #return self.critic_linear(x), F.softmax(self.actor_linear(x),dim=1), self.actor_linear2(x), (hx, cx)
#
#
#
class A3C_LSTM(torch.nn.Module):
    def __init__(self,num_steps):
        super(A3C_CONV, self).__init__()


        self.lstm = nn.LSTM(num_steps, hidden_dim,bidirectional=False)
        #self.lstm = nn.LSTMCell(1, 1)
        self.out = nn.Sigmoid()

        self.train()

    def forward(self, inputs):
        x, (hx, cx) = inputs

        x,(hx,cx) = self.lstm(x,(hx,cx))
        print(x.shape,hx.shape,cx.shape)
        exit()
        return self.out()
        #return self.critic_linear(x), F.softmax(self.actor_linear(x),dim=1), self.actor_linear2(x), (hx, cx)

class A3C_MLP(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(A3C_MLP, self).__init__()
        # self.fc1 = nn.Linear(num_inputs, 256)
        # self.lrelu1 = nn.LeakyReLU(0.1)
        # self.fc2 = nn.Linear(256, 256)
        # self.lrelu2 = nn.LeakyReLU(0.1)
        # self.fc3 = nn.Linear(256, 128)
        # self.lrelu3 = nn.LeakyReLU(0.1)
        # self.fc4 = nn.Linear(128, 128)
        # self.lrelu4 = nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(3,8,8), stride = (1,2,2))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,8,8), stride=(1,2,2))
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,4,4), stride=(1,2,2))
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1))

        self.fc1 = nn.Linear(in_features=576, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        #self.fc3 = nn.Linear(in_features=128, out_features=num_actions)

        self.relu = nn.ReLU()

        self.lstm = nn.LSTMCell(128, 128)

        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_actions)
        self.actor_linear2 = nn.Linear(128, num_actions)

        self.apply(weights_init_mlp)
        # lrelu = nn.init.calculate_gain('leaky_relu')
        # self.fc1.weight.data.mul_(lrelu)
        # self.fc2.weight.data.mul_(lrelu)
        # self.fc3.weight.data.mul_(lrelu)
        # self.fc4.weight.data.mul_(lrelu)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x, (hx, cx) = inputs

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        x = x.view(x.size(0), -1)
        #x = x.view(1, self.m1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx)
