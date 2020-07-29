
import os
import wave
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_wav_time(filename):
    with wave.open(filename, 'rb') as f:
        f = wave.open(filename)
        return f.getparams().nframes/f.getparams().framerate

def statistic_time():
    times = []
    wavs = glob.glob('../datasets/one16000/train/*/*.wav')
    for wav_path in tqdm(wavs):
        t = get_wav_time(wav_path)
        times.append(t)
    with open('time.pickle', 'wb') as f:
        pickle.dump(np.array(times), f)

def plot_():
    with open('time.pickle', 'rb') as f:
        data = pickle.load(f)
        plt.hist(data, bins=100)
        plt.grid()
        plt.show()

if __name__=='__main__':
    statistic_time()
    plot_()