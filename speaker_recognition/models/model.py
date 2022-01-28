import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_recognition.utils import est_output_shape

class SpeechRecognitionNet(nn.Module):

	def __init__(self, batch_shape):

		super().__init__()

		self.__name__ = "SpeechRecognitionNet(C3, FC3)"
		self.batch_shape = batch_shape
		self.num_classes = 2

		# -------------------------------------------------------------------------

		# convolutions blocks

		# for 2D convolution, the input (batch) shape is: [b, c, h, w]
		# b = batches, c = channels, h = height, w = width

		# 1 -> 8
		# add 8 channels = overfit BAD
		self.conv1 = nn.Sequential(
			# nn.Dropout2d(p=0.05),
			nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2),
			nn.MaxPool2d(kernel_size=5, stride=2),
			nn.ReLU(),
			nn.BatchNorm2d(8)
		) # -> [16, 39]

		# 8 -> 16
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2),
			# nn.MaxPool2d(kernel_size=5, stride=2),
			nn.ReLU(),
			nn.BatchNorm2d(16)
		) # -> [4, 10]

		# 16 -> 32
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
			# nn.MaxPool2d(kernel_size=3, stride=2),
			nn.ReLU(),
			nn.BatchNorm2d(32)
		) # -> [2, 5]

		# -------------------------------------------------------------------------

		# fully connected layer(s)

		# compute the size of the conv layers output
		conv_output_shape = est_output_shape(batch_shape, [self.conv1, self.conv2, self.conv3])
		num_flat_layers = math.prod(conv_output_shape[1:])
		print("conv_output_shape:", conv_output_shape, num_flat_layers)

		# n -> 128
		self.fc1 = nn.Sequential(
			nn.Linear(num_flat_layers, 128),
			nn.ReLU(),
			nn.BatchNorm1d(128)
		)

		# 64 -> 2 (num_classes)
		self.fc2 = nn.Sequential(
			nn.Linear(128, self.num_classes)
		)

	def forward(self, x):

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)

		# x = self.adapt(x) # could use nn.AdaptiveAvgPool2d to force size
		x = torch.flatten(x, start_dim=1) # flatten all dimensions except batch

		x = self.fc1(x)
		x = self.fc2(x)

		return x