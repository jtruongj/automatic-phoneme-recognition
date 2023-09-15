
from typing import List, Tuple

import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
from itertools import groupby
import torch


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
    # Merge together phonemes that are the same and consecutive
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


# Define model and processor
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


## Section on running model on our audio file.
# Load an input audio file
audio_filepath = os.path.join(os.path.dirname(__file__), '../data/assessment_9.wav')

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
# Code to extract phonemes and timestamps together. 
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

# get start time of all phonemes, including the separator phoneme SEP
phonemes_w_time_and_sep = []
for word in split_ids_w_time:
    _time, _ix = word[0]
    if _ix == 0:
        phonemes_w_time_and_sep.append((_time, "SEP"))  
        continue 
        
    for _time, _ix in word:
        phonemes_w_time_and_sep.append((_time, processor.decode(_ix)))   

# Print phonemes and start timestamps
for t, p in phonemes_w_time_and_sep:
    print(t, p)

# Add end timestamps to phonemes
non_empty_phonemes_w_begin_and_end = []
for ix, (t, p) in enumerate(phonemes_w_time_and_sep):
    if p == "" or p == "SEP":
        continue
    
    if ix == len(phonemes_w_time_and_sep) -1:
        non_empty_phonemes_w_begin_and_end.append((p, t, t+0.02))
    else:
        non_empty_phonemes_w_begin_and_end.append((p, t, phonemes_w_time_and_sep[ix+1][0]))

 
# Merge identical consecutive phonemes
phonemes_from_audio = merge_consecutive_phonemes(non_empty_phonemes_w_begin_and_end)
# phonemes_from_audio = [p for p, _, _ in non_empty_phonemes_w_begin_and_end_deduplicated]

# Print list of phonemes with their start time.
print("List of phonemes with start and end times:")
for p, beg, end in phonemes_from_audio:
    print(f"{p} {beg} {end}")

print("phonemes_from_audio: ", phonemes_from_audio)
    
    
#### Perform grapheme to phoneme conversion
import string
from g2p_en import G2p

from globals import CMU_TO_IPA

def text_to_phonemes(text):
    g2p = G2p()
    phonemes = g2p(text)
    phonemes = [CMU_TO_IPA[p] for p in phonemes]
    return phonemes

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


text = "But after all that commotion, was it all worthwhile? Absolutely yes! The set design was breathtaking, the actors were incredible, and the songs were memorable."
normalized_text = remove_punctuation(text)
phonemes_from_transcription = text_to_phonemes(normalized_text)


#### Match predicted phonemes with phonemes from transcription, and add timestamp matching
from itertools import zip_longest
from globals import IPA_TO_CMU_40

def pred_to_CMU(pred):
    
    predictions = []
    skip_next_char = False
    for i in range(len(pred)):
        if skip_next_char:
            skip_next_char = False
            continue
        
        if (i!= len(pred) - 1) and (pred[i:i+2] in IPA_TO_CMU_40):
            predictions.append(IPA_TO_CMU_40[pred[i:i+2]])
            skip_next_char = True
        else:
            predictions.append(IPA_TO_CMU_40[pred[i]])        
        
    return predictions

from typing import List, Tuple, Union

def compare_phonemes(transcript: List[str], predictions: List[str]) -> List[Tuple[str, int, str]]:
    """
    Compare a list of phonemes from a transcript with a list of phoneme predictions.
    
    For each phoneme in the transcript:
    - Return the phoneme string.
    - Return the index of the matching phoneme in the predictions (or -1 for a missing phoneme).
    - Return a string indicating the scenario for the phoneme (correct, substitution, insertion, or deletion).
    
    Parameters:
    - transcript (List[str]): List of phonemes from the transcript.
    - predictions (List[str]): List of phoneme predictions.
    
    Returns:
    - List[Tuple[str, int, str]]: List of tuples with phoneme, index in predictions, and scenario.
    """
    
    m, n = len(transcript), len(predictions)
    
    # Initialize a matrix to store the operations and their cost
    dp = [[(0, None) for _ in range(n+1)] for _ in range(m+1)]
    
    # Base cases: filling the first row and column
    for j in range(1, n+1):
        dp[0][j] = (j, 'insertion')
        
    for i in range(1, m+1):
        dp[i][0] = (i, 'deletion')
    
    # Dynamic programming step
    for i in range(1, m+1):
        for j in range(1, n+1):
            if transcript[i-1] == predictions[j-1]:
                dp[i][j] = (dp[i-1][j-1][0], 'correct')
            else:
                insertion = dp[i][j-1][0] + 1
                deletion = dp[i-1][j][0] + 1
                substitution = dp[i-1][j-1][0] + 1
                
                # Choose the operation with the least cost
                if insertion <= deletion and insertion <= substitution:
                    dp[i][j] = (insertion, 'insertion')
                elif deletion <= insertion and deletion <= substitution:
                    dp[i][j] = (deletion, 'deletion')
                else:
                    dp[i][j] = (substitution, 'substitution')
    
    # Backtrack to reconstruct the sequence of operations
    i, j = m, n
    operations = []
    while i > 0 or j > 0:
        if dp[i][j][1] == 'correct':
            operations.append((transcript[i-1], j-1, 'correct'))
            i, j = i-1, j-1
        elif dp[i][j][1] == 'insertion':
            operations.append(("", j-1, 'insertion'))
            j -= 1
        elif dp[i][j][1] == 'deletion':
            operations.append((transcript[i-1], -1, 'deletion'))
            i -= 1
        else:  # substitution
            operations.append((transcript[i-1], j-1, 'substitution'))
            i, j = i-1, j-1
            
    operations.reverse()
    
    return operations



transcript_cmu = [IPA_TO_CMU_40[p] for p in phonemes_from_transcription if p != " "] # convert to CMU phonemes

prediction_times = [(t_begin, t_end) for _, t_begin, t_end in phonemes_from_audio]
predictions_cmu = [IPA_TO_CMU_40[p] for p, _, _ in phonemes_from_audio if p != " "]



levenshtein_matching = compare_phonemes(transcript_cmu, predictions_cmu)
for res in levenshtein_matching:
    print(res)

print("Percent correct",  sum([1 for x in filter(lambda x: x[2] == "correct", levenshtein_matching)]) / len(levenshtein_matching))
print("Percent substitution",  sum([1 for x in filter(lambda x: x[2] == "substitution", levenshtein_matching)]) / len(levenshtein_matching))
print("Percent insertion",  sum([1 for x in filter(lambda x: x[2] == "insertion", levenshtein_matching)]) / len(levenshtein_matching))
print("Percent deletion",  sum([1 for x in filter(lambda x: x[2] == "deletion", levenshtein_matching)]) / len(levenshtein_matching))


import pandas as pd

# Step 1: Convert the results to a pandas DataFrame

data = []
for transcribed_ph, audio_ph_idx, result in levenshtein_matching:
    predicted_phoneme = predictions_cmu[audio_ph_idx] if audio_ph_idx != -1 else ""
    if result != "deletion":
        begin_time = prediction_times[audio_ph_idx][0]
        end_time = prediction_times[audio_ph_idx][1]
    else:
        begin_time = None
        end_time = None
        
    data.append([transcribed_ph, predicted_phoneme, result, begin_time, end_time])

df = pd.DataFrame(data, columns=["Transcribed Phoneme", "Predicted Phoneme", "Result", "Begin Time", "End Time"])


# Replace NaN values with "-"
df.fillna("-", inplace=True)

# Step 2: Display the DataFrame in markdown-ready format
print(df.to_markdown(index=False))
