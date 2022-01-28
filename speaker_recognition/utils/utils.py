import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

def est_output_shape(batch_shape, layers):

	seq = nn.Sequential(*layers)
	x = torch.rand(batch_shape)
	y = seq(x)

	return y.size()

def epoch(model, dataloder, criterion, optimizer=None):

	# set training or evaluation mode
	model.train() if optimizer else model.eval()

	num_pred = num_correct_pred = 0
	losses = []
	device = next(model.parameters()).device

	# repeat for each batch in the training set
	for i, data in enumerate(dataloder):

		# get inputs and labels; cast to device
		inputs, labels = data[0].to(device), data[1].to(device)

		# normalize inputs
		inputs = (inputs - inputs.mean()) / inputs.std()

		# forward
		outputs = model(inputs)
		loss = criterion(outputs, labels)

		if model.training:
			optimizer.zero_grad() # zero the parameter gradients
			loss.backward()
			optimizer.step()
			# scheduler.step()

		# prediction (class with highest score)
		_, prediction = torch.max(outputs, 1)

		# stats
		losses.append(loss.item())
		num_pred += prediction.shape[0] # number of predictions
		num_correct_pred += (prediction == labels).sum().item() # number of predictions that matched the target label

	return np.mean(losses), num_correct_pred/num_pred # (avg_loss, accuracy)

def train_model(model, dataloaders, num_epochs, criterion, optimizer=None, print_progress=True):

	# Stat = namedtuple("Stat", ["loss", "accuracy"])
	
	train_loader, val_loader = dataloaders
	stats_train, stats_val = [], []

	for i in range(num_epochs):

		# training iteration
		stats_train += [epoch(model, train_loader, criterion, optimizer)]

		# validation iteration (using model.eval())
		stats_val += [epoch(model, val_loader, criterion)]

		# print stats at the end of the epoch
		if print_progress:
			print(f"Epoch: {i+1:02d}/{num_epochs}, Loss (train/val): {stats_train[i][0]:.2f}/{stats_val[i][0]:.2f}, Accuracy (train/val): {stats_train[i][1]:.2f}/{stats_val[i][1]:.2f}")
		
	return stats_train, stats_val

def plot_train_history(stats):
	
	stats_train, stats_val = stats
	loss_train, acc_train = zip(*stats_train)
	loss_val, acc_val = zip(*stats_val)
	
	fig = plt.figure()
	
	plt.plot(acc_train)
	plt.plot(acc_val)
	plt.xlabel("Epoch"); plt.ylabel("Accuracy [%]")
	plt.legend(["Train", "Validation"])
	plt.grid()
	
	plt.show()
	
	pass