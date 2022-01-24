import os
import glob
import torch
import torchaudio
import random
import numpy as np

from math import ceil
from torch.utils.data import Dataset
from itertools import chain

import sounddevice as sd

class PodcastAudioDataset(Dataset):

	# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

	def __init__(self, split="all", transform=None, target_transform=None, **kwargs):

		# load data
		self.dir_data =  os.path.dirname(os.path.realpath(__file__)) + "/data" # absolute path to data
		self.max_dur = 5000; # [ms]
		self.labels = { "Esben": 0, "Peter": 1 }
		self.files = (
			[(file, self.labels["Esben"]) for file in glob.glob(self.dir_data + "/1_esben/*.wav")] +
			[(file, self.labels["Peter"]) for file in glob.glob(self.dir_data + "/2_peter/*.wav")]
		)

		# split dataset
		if range := { "train": (0.0, 0.8), "validation": (0.8, 0.9), "test": (0.9, 1.0) }.get(split):
			l = self.files
			l = list(chain.from_iterable(zip(l[:len(l)//2], l[len(l)//2:]))) # alternate list shuffle (0,1,0,1,...)
			self.files = l[int(len(l) * range[0]) : int(len(l) * range[1])] # splice by percent given by 'range'

		self.transform = transform
		self.target_transform = target_transform

		# optional arguments (from **kwargs)
		self.sample_rate = 16_000 # desired sample rate
		self.num_divs = kwargs["num_divs"] if "num_divs" in kwargs else 1 # number of sub-samples
		self.mel_spectrogram = False
		self.augments = kwargs["augments"] if "augments" in kwargs else [] # "time_shift", "spec_augment"

		# set dataset length (based on number of divisions)
		self.len = len(self.files) * self.num_divs

		print("Loaded PodcastAudioDataset from: ", self.dir_data)

	def __len__(self):
		return self.len

	def __getitem__(self, idx):

		if not idx in range(0, self.len):
			raise IndexError()

		# deduce indices (possible sub-samples)
		idx_file = idx // self.num_divs
		idx_sample = idx % self.num_divs

		# load data
		file = self.files[idx_file][0]; label = self.files[idx_file][1]
		waveform, sample_rate = torchaudio.load(file)
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

		# debug
		print(file, self.sample_rate, label)
		sd.play(waveform.T, self.sample_rate)
		sd.wait()

		# -------------------------------------------------------------------------

		# augmentation (time domain)

		if "time_shift" in self.augments:
			pass

		# -------------------------------------------------------------------------

		# mel spectrogram

		if self.mel_spectrogram:

			spectrogram = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(waveform)
			spectrogram = transforms.AmplitudeToDB(top_db=top_db)(spectrogram)

			if "spec_augment" in self.augments:
				pass

		# -------------------------------------------------------------------------

		# apply transforms

		if self.transform:
			data = self.transform(data)

		if self.target_transform:
			label = self.target_transform(label)

		# -------------------------------------------------------------------------

		return spectrogram if self.mel_spectrogram else waveform, label

def resize_waveform(waveform, max_ms, sample_rate):

	# pad or truncate the signal to a fixed length 'max_ms' in milliseconds

	num_rows, sig_len = waveform.shape
	max_len = sample_rate//1000 * max_ms

	if (sig_len > max_len):
		# Truncate the signal to the given length
		waveform = waveform[:,:max_len]

	elif (sig_len < max_len):
		# Length of padding to add at the beginning and end of the signal
		pad_begin_len = random.randint(0, max_len - sig_len)
		pad_end_len = max_len - sig_len - pad_begin_len

		# Pad with 0s
		pad_begin = torch.zeros((num_rows, pad_begin_len))
		pad_end = torch.zeros((num_rows, pad_end_len))

		waveform = torch.cat((pad_begin, waveform, pad_end), 1)

	return waveform

	def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):

		sig,sr = aud
		top_db = 80

		# spec has shape [channel, n_mels, time], where channel is mono, stereo etc
		spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

		# Convert to decibels
		spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
		return (spec)