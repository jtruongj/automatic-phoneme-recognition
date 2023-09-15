
from typing import List, Tuple

import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
from itertools import groupby
import torch

from transformers import pipeline

import os

def get_resampled_audio(audio_filepath):
    # Resample function using torchaudio
    def resample(waveform, sample_rate, target_sample_rate=16000):
        """Resample the WAV audio to the target sample rate."""
        if sample_rate == target_sample_rate:
            return waveform

        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        return resampler(waveform)

    # Read and resample
    waveform, current_sample_rate = torchaudio.load(audio_filepath)
    resampled_waveform = resample(waveform, current_sample_rate)
    return resampled_waveform

def merge_consecutive_phonemes(phoneme_data: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    if not phoneme_data:
        return []

    merged_data = []
    current_phoneme, current_start, current_end = phoneme_data[0]

    for phoneme, start, end in phoneme_data[1:]:
        # If current phoneme is the same as next phoneme and end time matches the start time
        if phoneme == current_phoneme and current_end == start:
            current_end = end
        else:
            merged_data.append((current_phoneme, current_start, current_end))
            current_phoneme, current_start, current_end = phoneme, start, end

    merged_data.append((current_phoneme, current_start, current_end))
    return merged_data


##############
# This code was found and adapted from here:
# https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft
##############


# Add the path to espeak library to the environment variables
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = "/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib"

model_name = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"

# load model and processor
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

print(model)

# # Prepare inputs with dataset
# # load dummy dataset and read soundfiles
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values
# sample_rate = ds[0]["audio"]["sampling_rate"]

audio_filepath = os.path.join(os.path.dirname(__file__), '../data/assessment_9.wav')
text = "But after all that commotion, was it all worthwhile? Absolutely yes! The set design was breathtaking, the actors were incredible, and the songs were memorable."

# Convert to numpy for feeding into processor
speech = get_resampled_audio(audio_filepath).numpy()[0]
sample_rate = processor.feature_extractor.sampling_rate
input_values = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_values


# retrieve logits
with torch.no_grad():
    res = model(input_values)
    logits = res.logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
# => should give ['m ɪ s t ɚ k w ɪ l t ɚ ɪ z ð ɪ ɐ p ɑː s əl l ʌ v ð ə m ɪ d əl k l æ s ɪ z æ n d w iː aʊ ɡ l æ d t ə w ɛ l k ə m h ɪ z ɡ ɑː s p ə']

print(transcription)

##############
# this is where the logic starts to get the start and end timestamp for each word
##############
phonemes = [w for w in transcription[0].split(' ') if len(w) > 0]
predicted_ids = predicted_ids[0].tolist()
duration_sec = input_values.shape[1] / sample_rate


ids_w_time = [(i / len(predicted_ids) * duration_sec, _id) for i, _id in enumerate(predicted_ids)]
# remove entries which are just "padding" (i.e. no characers are recognized)
ids_w_time = [i for i in ids_w_time if i[1] != processor.tokenizer.pad_token_id]
# now split the ids into groups of ids where each group represents a word
split_ids_w_time = [list(group) for k, group
                    in groupby(ids_w_time, lambda x: x[1] == processor.tokenizer.word_delimiter_token_id)
                    ]
print("split_ids_w_time: ", split_ids_w_time)
phonemes_w_time_and_sep = []
for word in split_ids_w_time:
    _time, _ix = word[0]
    if _ix == 0:
        phonemes_w_time_and_sep.append((_time, "SEP"))  
        continue 
        
    for _time, _ix in word:
        phonemes_w_time_and_sep.append((_time, processor.decode(_ix)))   

# phonemes_w_time_and_sep = phonemes_w_time_and_sep[1:]
for t, p in phonemes_w_time_and_sep:
    print(t, p)

non_empty_phonemes_w_begin_and_end = []
for ix, (t, p) in enumerate(phonemes_w_time_and_sep):
    if p == "" or p == "SEP":
        continue
    
    if ix == len(phonemes_w_time_and_sep) -1:
        non_empty_phonemes_w_begin_and_end.append((p, t, t+0.02))
    else:
        non_empty_phonemes_w_begin_and_end.append((p, t, phonemes_w_time_and_sep[ix+1][0]))

 
non_empty_phonemes_w_begin_and_end_deduplicated = merge_consecutive_phonemes(non_empty_phonemes_w_begin_and_end)


print("List of phonemes with start and end times:")
for p, beg, end in non_empty_phonemes_w_begin_and_end_deduplicated:
    print(f"{p} {beg} {end}")

print("non_empty_phonemes_w_begin_and_end_deduplicated: ", non_empty_phonemes_w_begin_and_end_deduplicated)
    