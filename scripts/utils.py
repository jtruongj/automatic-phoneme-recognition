import os
import string
import torchaudio
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from typing import List, Tuple
from g2p_en import G2p
from globals import IPA_TO_CMU_40, CMU_TO_IPA


def pred_to_CMU(pred: str) -> List[str]:
    """
    Convert a predicted string of IPA phonemes to their CMU equivalents.
    """
    predictions = []
    skip_next_char = False
    for i in range(len(pred)):
        if skip_next_char:
            skip_next_char = False
            continue

        if (i != len(pred) - 1) and (pred[i:i + 2] in IPA_TO_CMU_40):
            predictions.append(IPA_TO_CMU_40[pred[i:i + 2]])
            skip_next_char = True
        else:
            predictions.append(IPA_TO_CMU_40[pred[i]])

    return predictions


def text_to_phonemes(text: str) -> List[str]:
    """
    Convert text to phonemes using g2p_en and then map CMU phonemes to IPA format.
    """
    g2p = G2p()
    phonemes = g2p(text)
    return [CMU_TO_IPA[p] for p in phonemes]


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from the provided text.
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def get_resampled_audio(audio_filepath: str):
    """
    Load and resample the audio to the target sample rate using torchaudio.
    """

    def resample(waveform, sample_rate, target_sample_rate=16000):
        if sample_rate == target_sample_rate:
            return waveform
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        return resampler(waveform)

    waveform, current_sample_rate = torchaudio.load(audio_filepath)
    return resample(waveform, current_sample_rate)


def merge_consecutive_phonemes(phoneme_data: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """
    Merge consecutive phonemes that are identical to produce merged intervals.
    """
    if not phoneme_data:
        return []

    merged_data = []
    current_phoneme, current_start, current_end = phoneme_data[0]

    for phoneme, start, end in phoneme_data[1:]:
        if phoneme == current_phoneme and current_end == start:
            current_end = end
        else:
            merged_data.append((current_phoneme, current_start, current_end))
            current_phoneme, current_start, current_end = phoneme, start, end

    merged_data.append((current_phoneme, current_start, current_end))
    return merged_data


def print_table_in_markdown(levenshtein_matching, phonemes_from_audio_cmu, phonemes_from_audio_times):
    """
    Convert the matching results to a markdown table format and print it.
    """
    data = []
    for transcribed_ph, audio_ph_idx, result in levenshtein_matching:
        predicted_phoneme = phonemes_from_audio_cmu[audio_ph_idx] if audio_ph_idx != -1 else ""
        if result != "deletion":
            begin_time, end_time = phonemes_from_audio_times[audio_ph_idx]
        else:
            begin_time, end_time = None, None

        data.append([transcribed_ph, predicted_phoneme, result, begin_time, end_time])

    df = pd.DataFrame(data, columns=["Transcribed Phoneme", "Predicted Phoneme", "Result", "Begin Time", "End Time"])
    df.fillna("-", inplace=True)
    print(df.to_markdown(index=False))


def print_table_in_markdown(levenshtein_matching, phonemes_from_audio_cmu, phonemes_from_audio_times):
    # Convert the results to a pandas DataFrame
    data = []
    for transcribed_ph, audio_ph_idx, result in levenshtein_matching:
        predicted_phoneme = phonemes_from_audio_cmu[audio_ph_idx] if audio_ph_idx != -1 else ""
        if result != "deletion":
            begin_time = phonemes_from_audio_times[audio_ph_idx][0]
            end_time = phonemes_from_audio_times[audio_ph_idx][1]
        else:
            begin_time = None
            end_time = None

        data.append([transcribed_ph, predicted_phoneme, result, begin_time, end_time])

    df = pd.DataFrame(data, columns=["Transcribed Phoneme", "Predicted Phoneme", "Result", "Begin Time", "End Time"])

    # Replace NaN values with "-"
    df.fillna("-", inplace=True)

    # Display the DataFrame in markdown-ready format
    print(df.to_markdown(index=False))


def setup_espeak_library_path():
    """
    Set up the path to espeak library in the environment variables.
    """
    # Define path to espeak library
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = "/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib"


def initialize_model(model_name):
    """
    Initialize the processor and model from a pretrained model name.
    """
    # Load processor and model using the provided model name
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return processor, model


def compute_levenshtein_matching_metrics(levenshtein_matching):
    """
    Compute and print the Levenshtein distance metrics.
    """
    # Calculate the total number of phonemes
    total_num_phonemes = len(levenshtein_matching)

    def calculate_percent(condition):
        return sum(1 for x in filter(lambda x: x[2] == condition, levenshtein_matching)) / total_num_phonemes

    # Print various Levenshtein distance metrics
    print("")  # For formatting
    print("Percent correct:", calculate_percent("correct"))
    print("Percent substitution:", calculate_percent("substitution"))
    print("Percent insertion:", calculate_percent("insertion"))
    print("Percent deletion:", calculate_percent("deletion"))
    print("")  # For formatting
