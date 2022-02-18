import os
import glob
import torch
import torchaudio
import random
import numpy as np

from math import ceil
from torch.utils.data import Dataset
from itertools import chain

class PodcastAudioDataset(Dataset):

	# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

	LABELS = { "Esben": 0, 0: "Esben", "Peter": 1, 1: "Peter" }
	SPLITS = { "train": (0.0, 0.6), "validation": (0.6, 0.8), "test": (0.8, 1.0) }

	def __init__(self, split="all", transform=None, target_transform=None, **kwargs):

		# load data
		self.split = split
		self.dir_data =  os.path.dirname(os.path.realpath(__file__)) + "/data" # absolute path to data
		self.max_dur = 5000; # [ms]
		self.files = (
			[(file, self.LABELS["Esben"]) for file in glob.glob(self.dir_data + "/1_esben/*.wav")] +
			[(file, self.LABELS["Peter"]) for file in glob.glob(self.dir_data + "/2_peter/*.wav")]
		)

		# split dataset
		if range := self.SPLITS.get(split):
			l = self.files
			l = list(chain.from_iterable(zip(l[:len(l)//2], l[len(l)//2:]))) # alternate list shuffle (0,1,0,1,...)
			self.files = l[int(len(l) * range[0]) : int(len(l) * range[1])] # splice by percent given by 'range'

		# extra transforms
		self.transform = transform
		self.target_transform = target_transform

		# optional arguments (from **kwargs)
		self.sample_rate = 16_000 # desired sample rate
		self.num_divs = kwargs["num_divs"] if "num_divs" in kwargs else 1 # number of sub-samples
		self.mel_spec = kwargs["mel_spec"] if "mel_spec" in kwargs else False # use MelSpectrogram
		self.augments = kwargs["augments"] if "augments" in kwargs else [] # ["time_shift", "spec_augment"]

		# set dataset length (based on number of divisions)
		self.len = len(self.files) * self.num_divs

		# print(f"Loaded PodcastAudioDataset ({split}) from: {self.dir_data}")

	def sample_size(self):
		return self[0][0].size()

	def __len__(self):
		return self.len

	def get_file_and_label(self, idx):

		if not idx in range(0, self.len):
			raise IndexError()

		# deduce indices (possible sub-samples)
		idx_file = idx // self.num_divs
		idx_sample = idx % self.num_divs

		return self.files[idx_file][0], self.files[idx_file][1], idx_sample # (file, label, idx_sample)

	def get_wav_sample(self, idx):

		file, label, _ = self.get_file_and_label(idx)
		return torchaudio.load(file) # (waveform, sample_rate)

	def __getitem__(self, idx):

		# load data
		file, label, idx_sample = self.get_file_and_label(idx)
		waveform, sample_rate = self.get_wav_sample(idx)
		dur = waveform.size()[1] / sample_rate * 1000

		# -------------------------------------------------------------------------

		# process waveform

		# resample
		if sample_rate != self.sample_rate:
			waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(waveform)

		# fix length of waveform
		if dur != self.max_dur:
			waveform = resize_waveform(waveform, self.max_dur, self.sample_rate)

		# sub-sample
		if self.num_divs > 1:
			chunks = torch.chunk(waveform.T, self.num_divs)
			waveform = chunks[idx_sample].T

		# -------------------------------------------------------------------------

		# augmentation (time domain)

		if "time_shift" in self.augments:
			waveform = timeshift_augment(waveform)

		# -------------------------------------------------------------------------

		# mel spectrogram

		if self.mel_spec:

			spec = torchaudio.transforms.MelSpectrogram(self.sample_rate, n_fft=1024, n_mels=64, hop_length=None)(waveform)
			spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(spec)

			# augmentation (frequency domain)

			if "spec_augment" in self.augments:
				spec = spec_augment(spec)

		# -------------------------------------------------------------------------

		# apply extra transforms

		if self.transform:
			data = self.transform(data)

		if self.target_transform:
			label = self.target_transform(label)

		# -------------------------------------------------------------------------

		return (spec if self.mel_spec else waveform, label, idx)

def resize_waveform(waveform, max_ms, sample_rate):

	# pad or truncate the signal to a fixed length 'max_ms' in milliseconds

	num_rows, sig_len = waveform.shape
	max_len = sample_rate//1000 * max_ms

	if (sig_len > max_len):
		# truncate the signal to the given length
		waveform = waveform[:,:max_len]

	elif (sig_len < max_len):
		# length of padding to add at the beginning and end of the signal
		pad_begin_len = random.randint(0, max_len - sig_len)
		pad_end_len = max_len - sig_len - pad_begin_len

		# pad with 0s
		pad_begin = torch.zeros((num_rows, pad_begin_len))
		pad_end = torch.zeros((num_rows, pad_end_len))

		waveform = torch.cat((pad_begin, waveform, pad_end), 1)

	return waveform

def timeshift_augment(waveform):

	_, sig_len = waveform.shape
	shift_amt = int(random.random() * 0.4 * sig_len)
	waveform = waveform.roll(shift_amt)

	return waveform


def spec_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):

	_, n_mels, n_steps = spec.shape
	mask_value = spec.mean()
	spec_aug = spec

	freq_mask_param = max_mask_pct * n_mels
	for _ in range(n_freq_masks):
	  spec_aug = torchaudio.transforms.FrequencyMasking(freq_mask_param)(spec_aug, mask_value)

	time_mask_param = max_mask_pct * n_steps
	for _ in range(n_time_masks):
	  spec_aug = torchaudio.transforms.TimeMasking(time_mask_param)(spec_aug, mask_value)

	return spec_aug