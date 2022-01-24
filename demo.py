from datasets.PodcastAudioDataset import PodcastAudioDataset
from torch.utils.data import DataLoader

# parameters
num_divs = 1 # number of sub-samples per 5s sample

# load and process data (split sets)
dataset = PodcastAudioDataset(num_divs=2)
train_set = PodcastAudioDataset(split="train", num_divs=num_divs, augments=["time_shift", "spec_augment"])
val_set = PodcastAudioDataset(split="validation", num_divs=num_divs)
test_set = PodcastAudioDataset(split="test", num_divs=num_divs)

print(f"train_set: {len(train_set)}, val_set: {len(val_set)}, test_set: {len(test_set)}")

for idx, waveform in enumerate(dataset):
	print(idx, waveform[1])
	continue

# create dataloader
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# generate batch
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# create model
# https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html