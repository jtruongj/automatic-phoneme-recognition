import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
from itertools import groupby
import torch

from transformers import pipeline

import os

##############
# This code was found and adapted from here:
# https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft
##############


# Add the path to espeak library to the environment variables
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = "/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib"

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

data = [(0.6204948593073594, 26),
 (0.6405108225108226, 33),
 (0.680542748917749, 15),
 (0.7405906385281386, 36),
 (0.8406704545454546, 23),
 (0.9007183441558441, 6),
 (0.9407502705627706, 43),
 (1.1609258658008659, 53),
 (1.2810216450216452, 8),
 (1.3010376082251083, 8),
 (1.3811014610389611, 22),
 (1.4211333874458874, 36),
 (1.5011972402597402, 6),
 (1.581261093073593, 11),
 (1.6413089826839826, 7),
 (1.7213728354978355, 13),
 (1.781420725108225, 49),
 (1.9415484307359308, 38),
 (2.0015963203463203, 7),
 (2.081660173160173, 4),
 (2.1016761363636367, 4),
 (2.882298701298702, 32),
 (2.902314664502165, 32),
 (2.942346590909091, 33),
 (3.042426406926407, 21),
 (3.102474296536797, 17),
 (3.142506222943723, 15),
 (3.222570075757576, 53),
 (3.3226498917748923, 8),
 (3.422729707792208, 32),
 (3.4827775974025976, 43),
 (3.6028733766233763, 52),
 (3.722969155844156, 32),
 (3.7830170454545455, 37),
 (3.983176677489178, 8),
 (4.003192640692641, 8),
 (5.524405844155845, 36),
 (5.704549512987013, 26),
 (5.80462932900433, 5),
 (5.864677218614719, 7),
 (5.984772997835498, 8),
 (6.044820887445888, 46),
 (6.144900703463204, 6),
 (6.164916666666667, 6),
 (6.204948593073594, 8),
 (6.24498051948052, 10),
 (6.365076298701299, 24),
 (6.385092261904762, 24),
 (6.445140151515152, 14),
 (6.7053476731601735, 5),
 (7.485970238095238, 22),
 (7.526002164502165, 7),
 (7.686129870129871, 5),
 (7.7461777597402595, 14),
 (7.846257575757576, 6),
 (7.906305465367967, 12),
 (7.946337391774892, 17),
 (8.046417207792208, 21),
 (8.106465097402598, 37),
 (8.326640692640693, 4),
 (8.346656655844157, 4),
 (8.4867683982684, 32),
 (8.506784361471862, 32),
 (8.526800324675325, 33),
 (8.586848214285714, 21),
 (8.606864177489179, 21),
 (8.726959956709957, 26),
 (8.746975919913421, 26),
 (8.827039772727273, 27),
 (8.8670716991342, 14),
 (8.987167478354978, 52),
 (9.06723133116883, 6),
 (9.127279220779222, 44),
 (9.267390963203464, 11),
 (9.327438852813852, 17),
 (9.42751866883117, 42),
 (9.80782196969697, 22),
 (9.827837932900433, 17),
 (9.967949675324675, 36),
 (10.10806141774892, 11),
 (10.148093344155845, 6),
 (10.208141233766234, 43),
 (10.328237012987014, 21),
 (10.448332792207793, 32),
 (10.508380681818183, 43),
 (10.80862012987013, 17),
 (10.86866801948052, 4),
 (10.968747835497837, 11),
 (11.108859577922079, 27),
 (11.148891504329006, 14),
 (11.228955357142858, 12),
 (11.248971320346321, 7),
 (11.268987283549784, 7),
 (11.349051136363638, 26),
 (11.409099025974026, 61),
 (12.109657738095239, 36),
 (12.189721590909091, 4),
 (12.209737554112555, 12),
 (12.24976948051948, 22),
 (12.289801406926408, 7),
 (12.409897186147187, 5),
 (12.42991314935065, 5),
 (12.48996103896104, 53),
 (12.670104707792209, 42),
 (12.770184523809524, 21),
 (12.830232413419914, 32),
 (12.87026433982684, 43),
 (13.030392045454548, 13),
 (13.070423971861473, 14),
 (13.170503787878788, 13),
 (13.230551677489178, 27),
 (13.25056764069264, 27),
 (13.270583603896105, 7),
 (13.370663419913422, 26),
 (13.410695346320347, 61)]

def plot_phonemes(start_times, end_times, phonemes, merge=False):
    plt.figure(figsize=(15, 6))
    
    previous_phoneme = None
    merged_start = start_times[0]
    for i, (start, end, phoneme) in enumerate(zip(start_times, end_times, phonemes)):
        color = "skyblue"
        
        # If the phoneme is the same as the previous one, color it red
        if phoneme == previous_phoneme:
            color = "red"
            if merge:
                continue
        
        # If we're merging and the current phoneme is different than the previous one,
        # we plot the previous phoneme before continuing
        if merge and phoneme != previous_phoneme and i != 0:
            plt.barh(i-1, end - merged_start, left=merged_start, height=0.9, color=color)
            plt.text(merged_start + (end - merged_start) / 2, i-1, previous_phoneme, ha="center", va="bottom")
            merged_start = start

        # If not merging or at the last phoneme, just plot
        if not merge or i == len(phonemes) - 1:
            plt.barh(i, end - start, left=start, height=0.9, color=color)
            plt.text(start + (end - start) / 2, i, phoneme, ha="center", va="bottom")
        
        previous_phoneme = phoneme
    
    # Making the plot look nice
    plt.yticks([])
    plt.xlabel("Time (s)")
    plt.title("Phoneme Durations (Merged)" if merge else "Phoneme Durations")
    plt.grid(axis="x")
    blue_patch = mpatches.Patch(color='skyblue', label='Phoneme Duration')
    red_patch = mpatches.Patch(color='red', label='Repeated Phoneme Duration')
    plt.legend(handles=[blue_patch, red_patch])
    plt.tight_layout()
    plt.show()

# Splitting data into start times and phoneme IDs
start_times, phoneme_ids = zip(*data)
# Compute end times as start times of the next phoneme
end_times = list(start_times[1:]) + [start_times[-1] + 0.2]  # Adding a dummy end time for the last phoneme
# Convert phoneme IDs to their symbols
phonemes = [processor.decode(pid) for pid in phoneme_ids]

# Plotting without merging
plot_phonemes(start_times, end_times, phonemes)
# Plotting with merging
plot_phonemes(start_times, end_times, phonemes, merge=True)