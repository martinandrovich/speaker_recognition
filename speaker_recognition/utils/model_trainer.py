import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dill.source import getname
from datetime import datetime

class ModelTrainer:

	def __init__(self, model, dataset, **kwargs):

		# self.session = kwargs["session"] if "session" in kwargs else datetime.now().strftime("%Y%m%d_%H%M%S")
		self.session = kwargs["session"] if "session" in kwargs else datetime.now().strftime("demo")

		# self.dataset = dataset
		self.num_divs = kwargs["num_divs"] if "num_divs" in kwargs else 1
		self.augments = ["time_shift", "spec_augment"]
		self.mel_spec = True
		self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 64
		self.batch_shape = [16, 1, 256, 256]

		# self.model = model
		self.device = kwargs["device"] if "device" in kwargs else "cpu"
		self.lr = kwargs["lr"] if "lr" in kwargs else 0.001
		self.num_epochs = kwargs["num_epochs"] if "num_epochs" in kwargs else 10

		# create session
		os.mkdir("./sessions/" + self.session)

		# setup datasets
		self.dataset = getname(dataset)
		self.train_set = dataset(split="train", num_divs=self.num_divs, mel_spec=self.mel_spec, augments=self.augments)
		self.val_set   = dataset(split="validation", num_divs=self.num_divs, mel_spec=self.mel_spec)
		self.test_set  = dataset(split="test", num_divs=self.num_divs, mel_spec=self.mel_spec)

		# setup dataloaders
		self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
		self.val_loader = DataLoader(self.val_set, batch_size=16, shuffle=True)

		# setup model
		self.model = model(batch_shape=self.batch_shape)
		self.model = self.model.to(self.device)

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)

		# dump variables
		with open(f"./sessions/{self.session}/config.json", "w") as f:
			json.dump(vars(self), f, default=lambda obj: str(type(obj)), indent="\t", ensure_ascii=False)

	def train(self):

		for epoch in range(self.num_epochs):

			running_loss = 0.0
			correct_prediction = 0
			total_prediction = 0

			# Repeat for each batch in the training set
			for i, data in enumerate(self.train_loader):

				# Get the input features and target labels, and put them on the GPU
				inputs, labels = data[0].to(self.device), data[1].to(self.device)

				# Normalize the inputs
				inputs_m, inputs_s = inputs.mean(), inputs.std()
				inputs = (inputs - inputs_m) / inputs_s

				# Zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				# scheduler.step()

				# Keep stats for Loss and Accuracy
				running_loss += loss.item()

				# Get the predicted class with the highest score
				_, prediction = torch.max(outputs,1)
				# Count of predictions that matched the target label
				correct_prediction += (prediction == labels).sum().item()
				total_prediction += prediction.shape[0]

				#if i % 10 == 0:    # print every 10 mini-batches
				#    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

			# Print stats at the end of the epoch
			num_batches = len(train_loader)
			avg_loss = running_loss / num_batches
			acc = correct_prediction/total_prediction
			print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

	def eval(self, params=None):
		pass