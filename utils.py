#
import os
import torch
import numpy as np
import math
import time



def maybe_makedir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

def init_cuda(gpu=''):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used

def occumpy_mem(cuda_devices:str, part:float):
    splits = cuda_devices.split(',')
    for _cuda_device in splits:
        total, used = check_mem(_cuda_device)
        total = int(total)
        used = int(used)
        max_mem = int(total * part)
        block_mem = max_mem - used
        print('occumpy_mem', block_mem)
        with torch.no_grad():
            _nothing = torch.cuda.FloatTensor(256, 1024, block_mem).cuda(int(_cuda_device)).detach()
            print(_nothing.device)
        del _nothing


def wav_padding(wav_data_lst):
    wav_lens = [data.shape[1] for data in wav_data_lst]
    wav_max_len = max(wav_lens)
    wav_max_len = wav_max_len if wav_max_len > 300 else 300

    #wav_lens = np.array([leng//8 for leng in wav_lens])
    new_wav_data_lst = np.zeros((len(wav_data_lst), 200, wav_max_len, 1), dtype=np.float32)
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :, :wav_data_lst[i].shape[1], 0] = wav_data_lst[i]
    return new_wav_data_lst #, wav_lens


class WavFeatureCollate():
    def __init__(self):
        pass

    def collate_pad(self, batch):
        wavs_data = []
        labels = []
        paths = []
        for _, sample in enumerate(batch):
            wav_data, label, fs = None, None, None
            for _, tup in enumerate(sample):
                if isinstance(tup, int):
                    # label
                    labels.append(tup)
                elif isinstance(tup, np.ndarray):
                    # wav_data
                    wav_data = tup
                    wavs_data.append(wav_data)
                elif isinstance(tup, str):
                    paths.append(tup)
        wavs_data = wav_padding(wavs_data)

        try:
            wavs_data = wavs_data.transpose(0, 3, 1, 2)
            wavs_data = torch.from_numpy(wavs_data)
            # labels = torch.from_numpy(np.array(labels))
        except Exception as e:
            print(e)
            for wav_data, label, path in zip(wavs_data, labels, paths):
                print(path, label)
        labels = torch.from_numpy(np.array(labels, dtype=np.int))
        return wavs_data, labels, paths

    def __call__(self, batch):
        return self.collate_pad(batch)

class WavTestFeatureCollate():
    def __init__(self):
        pass

    def collate_pad(self, batch):
        wavs_data = []
        labels = []
        paths = []
        for _, sample in enumerate(batch):
            wav_data, label, fs = None, None, None
            for _, tup in enumerate(sample):
                if isinstance(tup, int):
                    # label
                    labels.append(tup)
                elif isinstance(tup, np.ndarray):
                    # wav_data
                    wav_data = tup
                    wavs_data.append(wav_data)
                elif isinstance(tup, str):
                    paths.append(tup)
        wavs_data = wav_padding(wavs_data)

        try:
            wavs_data = wavs_data.transpose(0, 3, 1, 2)
            wavs_data = torch.from_numpy(wavs_data)
            # labels = torch.from_numpy(np.array(labels))
        except Exception as e:
            print(e)
            for wav_data, label, path in zip(wavs_data, labels, paths):
                print(path, label)
        labels = torch.from_numpy(np.array(labels, dtype=np.int))
        return wavs_data, labels, paths

    def __call__(self, batch):
        return self.collate_pad(batch)

def decode(t, length, raw=False):
    """Decode encoded texts back into strs.

    Args:
        torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
        torch.IntTensor [n]: length of each text.

    Raises:
        AssertionError: when the texts and its length does not match.

    Returns:
        text (str or list of str): texts to convert.
    """
    alphabet = '-0123456789'
    if length.numel() == 1:
        length = length[0]
        assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
        if raw:
            return [''.join([alphabet[i] for i in t])]
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(alphabet[t[i]])
            return [''.join(char_list)]
    else:
        # batch mode
        assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
        texts = []
        index = 0
        for i in range(length.numel()):
            l = length[i]
            texts.extend(
                decode(
                    t[index:index + l], torch.IntTensor([l]), raw=raw))
            index += l
        return texts

