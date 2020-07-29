import os
import glob
import random
import wave
import numpy as np

from tqdm import tqdm
from scipy.fftpack import fft
from specaugment import spec_augment
from torch.utils.data import Dataset



X = np.linspace(0, 400 - 1, 400, dtype = np.int64)
W = 0.54 - 0.46 * np.cos(2 * np.pi * (X) / (400 - 1) ) # 汉明窗

def getFrequencyFeature(wavsignal, fs):
    if(16000 != fs):
        raise ValueError('[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(fs) + ' Hz. ')
    
    # wav波形 加时间窗以及时移10ms
    time_window = 25 # 单位ms
    window_length = int(fs / 1000 * time_window) # 计算窗长度的公式，目前全部为400固定值
    
    wav_arr = np.array(wavsignal)
    #wav_length = len(wavsignal[0])
    wav_length = wav_arr.shape[1]
    
    range0_end = int(len(wavsignal[0])/fs*1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, window_length // 2), dtype=np.float32) # 用于存放最终的频率特征数据
    data_line = np.zeros((1, window_length), dtype=np.float32)
    
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        
        data_line = wav_arr[0, p_start:p_end]
        
        data_line = data_line * W # 加窗
        
        data_line = np.abs(fft(data_line)) / wav_length
        
        
        data_input[i]=data_line[0: window_length // 2] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
        
    #print(data_input.shape)
    data_input = np.log(data_input + 1)
    data_input = data_input.T
    return data_input


def wav_loader(pth):
    with wave.open(pth, 'rb') as wav:
        try:
            num_frame = wav.getnframes() # 获取帧数
            num_channel = wav.getnchannels() # 获取声道数
            framerate = wav.getframerate() # 获取帧速率
            num_sample_width = wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
            str_data = wav.readframes(num_frame) # 读取全部的帧
            wave_data = np.fromstring(str_data, dtype=np.short) # 将声音文件数据转换为数组矩阵形式
            wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
            wave_data = wave_data.T # 将矩阵转置
            ffimg = getFrequencyFeature(wave_data, framerate)
            return ffimg
        except Exception as e:
            print(e)
            print(pth)

class WavDataset(Dataset):
    def __init__(self, is_train, augment=False):
        self.augment = augment
        self.is_train = is_train
        data_root = './datasets/one16000/train'
        cry_types = os.listdir(data_root)
        self.train_labels = []
        self.val_labels = []
        self.train_wavs = []
        self.val_wavs = []
        self.samples = []
        self.type2label = {'awake': 0, 'diaper': 1, 'hug': 2, 'hungry': 3, 'sleepy': 4, 'uncomfortable': 5}
        print('-------------------------------')
        for _, cry_type in enumerate(cry_types):
            _label = self.type2label[cry_type]
            wavs = glob.glob(os.path.join(data_root, cry_type, '*.wav'))
            random.seed(2020)
            random.shuffle(wavs)
            _train_wavs = wavs[:-10]
            _val_wavs = wavs[-10:]
            self.train_wavs.extend(_train_wavs)
            self.train_labels.extend([_label for _ in range(len(_train_wavs))])
            self.val_wavs.extend(_val_wavs)
            self.val_labels.extend([_label for _ in range(len(_val_wavs))])
            print(cry_type, len(_train_wavs), len(_val_wavs))
        print('all train', len(self.train_wavs), len(self.train_labels))
        print('all val', len(self.val_wavs), len(self.val_labels))
        print('-------------------------------')
        
    def __len__(self):
        assert len(self.train_wavs) == len(self.train_labels)
        assert len(self.val_wavs) == len(self.val_labels)
        if self.is_train:
            return len(self.train_wavs)
        else:
            return len(self.val_wavs)

    def __getitem__(self, idx):
        if self.is_train:
            sample = self.train_wavs[idx]
            label = self.train_labels[idx]
        else:
            sample = self.val_wavs[idx]
            label = self.val_labels[idx]
        
        ffimg = wav_loader(sample)
        _path = sample
        if self.augment:
            if np.random.random() > 0.4:
                ffimg = ffimg[None, :, :]
                ffimg = spec_augment(mel_spectrogram=ffimg, time_warping_para=30, frequency_masking_para=20,
                                     time_masking_para=30, frequency_mask_num=1, time_mask_num=1)
                ffimg = ffimg[0]
        return ffimg, label, _path

class WavTestDataset(Dataset):
    def __init__(self):
        data_root = './datasets/one16000/test'
        self.labels = []
        self.wavs = []
        self.label2name = {0: 'awake', 1: 'diaper', 2: 'hug', 3: 'hungry', 4: 'sleepy', 5: 'uncomfortable'}
        print('-------------------------------')
        self.wavs = glob.glob(os.path.join(data_root, '*.wav'))
        
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        sample = self.wavs[idx]
        ffimg = wav_loader(sample)
        _path = sample
        return ffimg, _path

if __name__ == '__main__':
    import cv2
    import utils
    wavDataset = WavDataset(is_train=True)
    for _ff, _label, _path in wavDataset:
        print(_ff.shape, _label, _path)
        pa_wav = utils.wav_padding([_ff, _ff])
        print(pa_wav.shape)
    #     cv2.imshow('', pa_wav[0])
    #     cv2.waitKey()
