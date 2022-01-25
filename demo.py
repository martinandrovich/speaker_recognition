from torch.utils.data import DataLoader
from datasets.PodcastAudioDataset import PodcastAudioDataset
from network import SpeechRecognitionNet

import sounddevice as sd
import matplotlib.pyplot as plt

# parameters
num_divs = 1 # number of sub-samples per 5s sample
mel_spec = True # MelSpectrogram

# load and process data (split sets)
train_set = PodcastAudioDataset(split="train", num_divs=num_divs, mel_spec=mel_spec, augments=["time_shift", "spec_augment"])
val_set   = PodcastAudioDataset(split="validation", num_divs=num_divs, mel_spec=mel_spec)
test_set  = PodcastAudioDataset(split="test", num_divs=num_divs, mel_spec=mel_spec)

# test data
print(f"train_set: {len(train_set)}, val_set: {len(val_set)}, test_set: {len(test_set)}")
print(f"sample_size: {train_set.sample_size()}")
# for i, sample in enumerate(val_set):
# 	print(sample)
# 	if mel_spec:
# 		plt.imshow(sample[0][0,:,:])
# 		plt.show()
# 	else:
# 		sd.play(waveform.T, self.sample_rate)
# 		sd.wait()


# create dataloader(s)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

# generate batch
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# create model
# https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html

# batch shape (not size)
batch_shape = [16] + list(train_set.sample_size())
print("batch_shape", batch_shape, train_features.size())

# test network
net = SpeechRecognitionNet(batch_shape=batch_shape)
y = net.forward(train_features)
print(y)