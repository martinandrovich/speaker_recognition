from datasets.PodcastAudioDataset import PodcastAudioDataset
from torch.utils.data import DataLoader

# parameters
num_divs = 1 # number of sub-samples per 5s sample

# load and process data (split sets)
dataset = PodcastAudioDataset(num_divs=2)
train_set = PodcastAudioDataset(split="train", num_divs=num_divs, augments=["time_shift", "spec_augment"])
val_set = PodcastAudioDataset(split="validation", num_divs=num_divs)
test_set = PodcastAudioDataset(split="test", num_divs=num_divs)