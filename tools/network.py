from torch import nn, cat, LongTensor
from torch.nn import functional


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_1 = self.conv_layer_(3, 16, 3)
        self.max_pool1 = nn.MaxPool2d(3, 3)
        self.conv_2 = self.conv_layer_(16, 32, 3)
        self.max_pool2 = nn.MaxPool2d(3, 3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(513, 128)
        self.linear2 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 3)

    @staticmethod
    def conv_layer_(in_c, out_c, kernel):
        conv_layer = nn.Conv2d(in_c, out_c, kernel_size=(kernel, kernel), padding=(0, 0))
        return conv_layer

    def forward(self, x, ext):
        out = functional.relu(self.conv_1(x))
        out = self.max_pool1(out)
        out = functional.relu(self.conv_2(out))
        out = self.max_pool2(out)
        out = out.view(out.size(0), -1)
        out = cat((out, ext), 1)
        out = functional.relu(self.linear1(out))
        out = functional.relu(self.linear2(out))
        return functional.softmax(self.output(out), dim=1)
