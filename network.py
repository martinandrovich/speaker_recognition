import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechRecognitionNet(nn.Module):

	def __init__(self, batch_shape):
		
		super().__init__()

		self.batch_shape = batch_shape
		self.num_classes = 2

		# -------------------------------------------------------------------------

		# convolutions blocks

		# for 2D convolution, the input (batch) shape is: [b, c, h, w]
		# b = batches, c = channels, h = height, w = width

		# 1 -> 8
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2),
			nn.MaxPool2d(kernel_size=5, stride=2),
			nn.ReLU(),
			# nn.BatchNorm2d(8)
		)

		# 8 -> 16
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2),
			# nn.MaxPool2d(kernel_size=5, stride=2),
			nn.ReLU(),
			# nn.BatchNorm2d(8)
		)

		# 16 -> 32
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
			# nn.MaxPool2d(kernel_size=5, stride=2),
			nn.ReLU(),
			# nn.BatchNorm2d(8)
		)

		# -------------------------------------------------------------------------

		# fully connected layer(s)

		# compute the size of the conv layers output
		conv_output_shape = est_output_shape(batch_shape, [self.conv1, self.conv2, self.conv3])
		num_flat_layers = math.prod(conv_output_shape[1:])
		print(conv_output_shape, num_flat_layers)

		# n -> 128
		self.fc1 = nn.Sequential(
			nn.Linear(num_flat_layers, 128),
			nn.ReLU(),
		)

		# 128 -> 64
		self.fc2 = nn.Sequential(
			nn.Linear(128, 64),
			nn.ReLU(),
		)

		# 64 -> 2 (num_classes)
		self.fc3 = nn.Sequential(
			nn.Linear(64, self.num_classes)
		)

	def forward(self, x):

		print(x.size())
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)

		print(x.size())
		# x = self.adapt(x) # could use nn.AdaptiveAvgPool2d to force size
		x = torch.flatten(x, start_dim=1) # flatten all dimensions except batch
		print(x.size())

		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)

		return x

def est_output_shape(batch_shape, layers):

	seq = nn.Sequential(*layers)
	x = torch.rand(batch_shape)
	y = seq(x)
	return y.size()