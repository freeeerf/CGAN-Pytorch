'''
添加一个推理的文件, 使用已经训练好的模型.
读取一个数字, 然后输出它的手写图片.
'''

import time
import torch
import torch.nn as nn
import torchvision.utils
import numpy as np  

batch_size = 64
latentdim = 100
n_classes = 10
imageSize = 28
img_shape = (1, imageSize, imageSize)

# 原模型的定义使用了opt参数, 无法快速拆出来, 这里只好先单独定义了
class Generator(nn.Module): 
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(n_classes, n_classes)
        self.depth=128

        def init(input, output, normalize=True): 
            layers = [nn.Linear(input, output)]
            if normalize: 
                layers.append(nn.BatchNorm1d(output, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers 

        self.generator = nn.Sequential(
            
            *init(latentdim+n_classes, self.depth), 
            *init(self.depth, self.depth*2), 
            *init(self.depth*2, self.depth*4), 
            *init(self.depth*4, self.depth*8),
            nn.Linear(self.depth * 8, int(np.prod(img_shape))),
            nn.Tanh()
           
            )

    # torchcat needs to combine tensors 
    def forward(self, noise, labels): 
        gen_input = torch.cat((self.label_embed(labels), noise), -1)
        img = self.generator(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


model = Generator()

state_dict = torch.load('outputs/generator_epoch.pth')
model.load_state_dict(state_dict)
model.eval()


# 读取一个整数, 然后显示出来
while True:
    num = int(input("Input a integer:"))

    noise = torch.randn(batch_size, latentdim)
    gen_labels = torch.ones(batch_size, dtype=torch.long) * num
    gen_imgs = model(noise, gen_labels)

    torchvision.utils.save_image(gen_imgs, 'infer_result/samples_%d_%d.png' % (num, int(time.time())), normalize=True)
