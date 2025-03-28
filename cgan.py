import numpy as np  
import os 
import time 
import argparse
import signal
import torch 
from torch.autograd import Variable
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader
from torchvision import datasets 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.utils as vutils
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist')
parser.add_argument('--dataroot', required=True, help='path to data')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--outputBatchSize', type=int, default=64, help='output batch size')
parser.add_argument('--imageSize', type=int, default=32, help='image size input')
parser.add_argument('--channels', type=int, default=1, help='number of channels')
parser.add_argument('--latentdim', type=int, default=100, help='size of latent vector')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes in data set')
parser.add_argument('--epoch', type=int, default=200, help='number of epoch')
parser.add_argument('--lrate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta', type=float, default=0.5, help='beta for adam optimizer')
parser.add_argument('--beta1', type=float, default=0.999, help='beta1 for adam optimizer')
parser.add_argument('--output', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--randomseed', type=int, help='seed')
 
opt = parser.parse_args()

img_shape = (opt.channels, opt.imageSize, opt.imageSize)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print('device:', device)

os.makedirs(opt.output, exist_ok=True)

if opt.randomseed is None: 
	opt.randomseed = random.randint(1,10000)
random.seed(opt.randomseed)
torch.manual_seed(opt.randomseed)

# preprocessing for mnist, lsun, cifar10
if opt.dataset == 'mnist': 
	dataset = datasets.MNIST(root = opt.dataroot, train=True,download=True, 
		transform=transforms.Compose([transforms.Resize(opt.imageSize), 
			transforms.ToTensor(), 
			transforms.Normalize((0.5,), (0.5,))]))

elif opt.dataset == 'lsun': 
	dataset = datasets.LSUN(root = opt.dataroot, train=True,download=True, 
		transform=transforms.Compose([transforms.Resize(opt.imageSize), 
			transforms.CenterCrop(opt.imageSize),
			transforms.ToTensor(), 
			transforms.Normalize((0.5,), (0.5,))]))

elif opt.dataset == 'cifar10':  
	dataset = datasets.CIFAR10(root = opt.dataroot, train=True,download=True, 
		transform=transforms.Compose([transforms.Resize(opt.imageSize), 
			transforms.ToTensor(), 
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))



assert dataset 
dataloader = torch.utils.data.DataLoader(dataset, batch_size = opt.batchSize, shuffle=True)

# building generator
class Generator(nn.Module): 
	def __init__(self):
		super(Generator, self).__init__()
		self.label_embed = nn.Embedding(opt.n_classes, opt.n_classes)
		self.depth=128

		def init(input, output, normalize=True): 
			layers = [nn.Linear(input, output)]
			if normalize: 
				layers.append(nn.BatchNorm1d(output, 0.8))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers 

		self.generator = nn.Sequential(
			
			*init(opt.latentdim+opt.n_classes, self.depth), 
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


class Discriminator(nn.Module): 
	def __init__(self): 
		super(Discriminator, self).__init__()
		self.label_embed1 = nn.Embedding(opt.n_classes, opt.n_classes)
		self.dropout = 0.4 
		self.depth = 512

		def init(input, output, normalize=True): 
			layers = [nn.Linear(input, output)]
			if normalize: 
				layers.append(nn.Dropout(self.dropout))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers 

		self.discriminator = nn.Sequential(
			*init(opt.n_classes+int(np.prod(img_shape)), self.depth, normalize=False),
			*init(self.depth, self.depth), 
			*init(self.depth, self.depth),
			nn.Linear(self.depth, 1),
			nn.Sigmoid()
			)

	def forward(self, img, labels): 
		imgs = img.view(img.size(0),-1)
		inpu = torch.cat((imgs, self.label_embed1(labels)), -1)
		validity = self.discriminator(inpu)
		return validity 

# weight initialization
def init_weights(m): 
	if type(m)==nn.Linear:
		torch.nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0.01)

need_quit_process = False

def handle_signal(signum, frame):
	global need_quit_process
	need_quit_process = True
	print(f'Received signal {signum}. Exiting gracefully.')

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)
	
# Building generator 
generator = Generator()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lrate, betas=(opt.beta, opt.beta1))

# Building discriminator  
discriminator = Discriminator()
discriminator.apply(init_weights)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lrate, betas=(opt.beta, opt.beta1))

# Loss functions 
a_loss = torch.nn.BCELoss()

# Labels 
real_label = 0.9
fake_label = 0.0

FT = torch.LongTensor
FT_a = torch.FloatTensor

generator.to(device)
discriminator.to(device)
a_loss.to(device)

last_time = time.time()

# training 
for epoch in range(opt.epoch): 
	last_gen_imgs = None
	last_gen_labels = None
	last_total_loss = None

	if need_quit_process:
		break

	for i, (imgs, labels) in enumerate(dataloader): 
		if need_quit_process:
			break
			
		batch_size = imgs.shape[0]

		# convert img, labels into proper form 
		imgs = imgs.to(device)
		labels = labels.to(device)
	
		# creating real and fake tensors of labels 
		reall = torch.ones(batch_size, 1, device=device) * real_label
		f_label = torch.ones(batch_size, 1, device=device) * fake_label

		# initializing gradient
		gen_optimizer.zero_grad() 
		d_optimizer.zero_grad()

		#### TRAINING GENERATOR ####
		# Feeding generator noise and labels 
		noise = torch.randn(batch_size, opt.latentdim, device=device)
		gen_labels = torch.randint(0, opt.n_classes, (batch_size,), device=device)
		
		gen_imgs = generator(noise, gen_labels)
		
		# Ability for discriminator to discern the real v generated images 
		validity = discriminator(gen_imgs, gen_labels)
		
		# Generative loss function 
		g_loss = a_loss(validity, reall)

		# Gradients 
		g_loss.backward()
		gen_optimizer.step()

		#### TRAINING DISCRIMINTOR ####

		d_optimizer.zero_grad()

		# Loss for real images and labels 
		validity_real = discriminator(imgs, labels)
		d_real_loss = a_loss(validity_real, reall)

		# Loss for fake images and labels 
		validity_fake = discriminator(gen_imgs.detach(), gen_labels)
		d_fake_loss = a_loss(validity_fake, f_label)

		# Total discriminator loss 
		d_loss = 0.5 * (d_fake_loss+d_real_loss)

		# calculates discriminator gradients
		d_loss.backward()
		d_optimizer.step()

		last_gen_imgs = gen_imgs
		last_gen_labels = gen_labels
		last_total_loss = (g_loss + d_loss).item()

	if epoch%10 == 0: 
		labels_str = ''.join(map(str, last_gen_labels[0:opt.outputBatchSize].tolist()))
		vutils.save_image(last_gen_imgs[0:opt.outputBatchSize], '%s/samples_%03d_%.3f_%s.png' % (opt.output, epoch, last_total_loss, labels_str), normalize=True)
		
	now = time.time()
	used_time = f"{(now - last_time):.2f}s"
	last_time = now
	date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	print("%s [Epoch: %d/%d]" "[D loss: %f]" "[G loss: %f] [Total loss: %f] used time: %s" % (date_str, epoch+1, opt.epoch, d_loss.item(), g_loss.item(), last_total_loss, used_time))
	
# checkpoints 
torch.save(generator.state_dict(), '%s/generator_epoch.pth' % opt.output)
torch.save(discriminator.state_dict(), '%s/discriminator_epoch.pth' % opt.output)















