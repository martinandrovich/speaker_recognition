import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_recognition.utils import est_output_shape

class SpeechRecognitionNet(nn.Module):

	def __init__(self, batch_shape):

		super().__init__()

		self.__name__ = "SpeechRecognitionNet"
		self.batch_shape = batch_shape
		self.num_classes = 2

		# -------------------------------------------------------------------------

		# convolutions block(s)

		# for 2D convolution, the input (batch) shape is: [b, c, h, w]
		# b = batches, c = channels, h = height, w = width

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=1, padding=2),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.ReLU(),
			# nn.BatchNorm2d(2)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.ReLU(),
			# nn.BatchNorm2d(4),
		)

		# size of the conv layers output
		conv_output_shape = est_output_shape(batch_shape,[self.conv1, self.conv2])
		num_flat_layers = math.prod(conv_output_shape[1:])

		# -------------------------------------------------------------------------

		# flatten (conv > fc)

		# self.flatten = nn.AdaptiveAvgPool2d(output_size) # force size
		self.flatten = nn.Flatten(start_dim=1) # flatten all dimensions except batch

		# -------------------------------------------------------------------------

		# fully connected layer(s)
		# print("conv_output_shape:", conv_output_shape, num_flat_layers)

		self.fc1 = nn.Sequential(
			nn.Linear(num_flat_layers, num_flat_layers//2),
			nn.ReLU(),
			# nn.BatchNorm1d(num_flat_layers//2)
		)

		self.fc2 = nn.Sequential(
			nn.Linear(num_flat_layers//2, num_flat_layers//4),
			nn.ReLU(),
			# nn.BatchNorm1d(num_flat_layers//4)
		)

		self.fc3 = nn.Sequential(
			nn.Linear(num_flat_layers//4, self.num_classes)
		)

	def forward(self, x):

		x = self.conv1(x)
		x = self.conv2(x)

		x = self.flatten(x)

		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)

		return x