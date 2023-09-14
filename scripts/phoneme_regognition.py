from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
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
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

print(model)

# load dummy dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# tokenize
input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values

# retrieve logits
with torch.no_grad():
    res = model(input_values)
    logits = res.logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
# => should give ['m ɪ s t ɚ k w ɪ l t ɚ ɪ z ð ɪ ɐ p ɑː s əl l ʌ v ð ə m ɪ d əl k l æ s ɪ z æ n d w iː aʊ ɡ l æ d t ə w ɛ l k ə m h ɪ z ɡ ɑː s p ə']

print(transcription)
