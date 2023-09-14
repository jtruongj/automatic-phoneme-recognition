from itertools import groupby
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import torchaudio
from datasets import load_dataset

##############
# load model & audio and run audio through model
##############
model_name = 'facebook/wav2vec2-large-960h-lv60-self'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

audio_filepath = 'data/assessment_9.wav'

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

# Convert to numpy for feeding into processor
speech = resampled_waveform.squeeze().numpy()
sample_rate = processor.feature_extractor.sampling_rate
input_values = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_values

# # load dummy dataset and read soundfiles
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
# sample_rate = ds[0]["audio"]["sampling_rate"]
# input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values

with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0]).lower()

##############
# this is where the logic starts to get the start and end timestamp for each word
##############
words = [w for w in transcription.split(' ') if len(w) > 0]
predicted_ids = predicted_ids[0].tolist()
duration_sec = input_values.shape[1] / sample_rate


ids_w_time = [(i / len(predicted_ids) * duration_sec, _id) for i, _id in enumerate(predicted_ids)]
# remove entries which are just "padding" (i.e. no characers are recognized)
ids_w_time = [i for i in ids_w_time if i[1] != processor.tokenizer.pad_token_id]
# now split the ids into groups of ids where each group represents a word
split_ids_w_time = [list(group) for k, group
                    in groupby(ids_w_time, lambda x: x[1] == processor.tokenizer.word_delimiter_token_id)
                    if not k]

assert len(split_ids_w_time) == len(words)  # make sure that there are the same number of id-groups as words. Otherwise something is wrong

word_start_times = []
word_end_times = []
for cur_ids_w_time, cur_word in zip(split_ids_w_time, words):
    _times = [_time for _time, _id in cur_ids_w_time]
    word_start_times.append(min(_times))
    word_end_times.append(max(_times))
    
words, word_start_times, word_end_times