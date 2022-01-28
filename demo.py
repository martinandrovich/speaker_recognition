# imports

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dill.source import getname

from speaker_recognition.datasets import PodcastAudioDataset
from speaker_recognition.models import SpeechRecognitionNet
from speaker_recognition.utils import train_model, plot_train_history, epoch

# parameters

num_divs     = 1 # number of sub-samples per 5s sample
mel_spec     = True # convert waveform to MelSpectrogram
augments     = ["time_shift", "spec_augment"]
batch_size   = 8 * num_divs
lr           = 0.001
weight_decay = 1e-6
momentum     = 0.95
num_epochs   = 20

# datasets and dataloaders

train_set = PodcastAudioDataset(split="train", num_divs=num_divs, mel_spec=mel_spec, augments=augments)
val_set   = PodcastAudioDataset(split="validation", num_divs=num_divs, mel_spec=mel_spec)
test_set  = PodcastAudioDataset(split="test", num_divs=num_divs, mel_spec=mel_spec)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

batch_shape = next(iter(train_loader))[0].size() # batch shape, not size

# model (network)

model = SpeechRecognitionNet(batch_shape=batch_shape)
print(f"Using model: {getname(model)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum)

# train

stats = train_model(
	model=model,
	dataloaders=[train_loader, val_loader],
	num_epochs=num_epochs,
	criterion=criterion,
	optimizer=optimizer,
	print_progress=True
)

plot_train_history(stats)

stats = epoch(model, test_loader, criterion)
print(stats)

