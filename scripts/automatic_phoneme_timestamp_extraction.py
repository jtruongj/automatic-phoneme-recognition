import os
from typing import List, Tuple
import torch
from itertools import groupby
from globals import IPA_TO_CMU_40
from utils import (get_resampled_audio, merge_consecutive_phonemes, print_table_in_markdown, remove_punctuation,
                   text_to_phonemes, setup_espeak_library_path, initialize_model, compute_levenshtein_matching_metrics)


def match_transcription_to_predictions(transcript: List[str], predictions: List[str]) -> List[Tuple[str, int, str]]:
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
    dp = [[(0, None) for _ in range(n + 1)] for _ in range(m + 1)]

    # Base cases: filling the first row and column
    for j in range(1, n + 1):
        dp[0][j] = (j, 'insertion')

    for i in range(1, m + 1):
        dp[i][0] = (i, 'deletion')

    # Dynamic programming step
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if transcript[i - 1] == predictions[j - 1]:
                dp[i][j] = (dp[i - 1][j - 1][0], 'correct')
            else:
                insertion = dp[i][j - 1][0] + 1
                deletion = dp[i - 1][j][0] + 1
                substitution = dp[i - 1][j - 1][0] + 1

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
            operations.append((transcript[i - 1], j - 1, 'correct'))
            i, j = i - 1, j - 1
        elif dp[i][j][1] == 'insertion':
            operations.append(("", j - 1, 'insertion'))
            j -= 1
        elif dp[i][j][1] == 'deletion':
            operations.append((transcript[i - 1], -1, 'deletion'))
            i -= 1
        else:  # substitution
            operations.append((transcript[i - 1], j - 1, 'substitution'))
            i, j = i - 1, j - 1

    operations.reverse()

    return operations


def get_predicted_phoneme_ids(audio_filepath: str, processor, model):

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

    print("predicted phonemes (IPA)", transcription)
    duration_sec = input_values.shape[1] / sample_rate

    return predicted_ids, duration_sec


def get_predicted_phonemes_and_timestamps(predicted_ids: List[int], duration_sec: float, processor):
    # Convert the tensor of predicted IDs to a list.
    predicted_ids_list = predicted_ids[0].tolist()

    # Calculate the timestamp for each predicted ID based on its position in the sequence.
    timestamped_ids = [(i / len(predicted_ids_list) * duration_sec, _id) for i, _id in enumerate(predicted_ids_list)]

    # Filter out padding tokens, which don't represent recognized characters.
    non_padded_timestamped_ids = [item for item in timestamped_ids if item[1] != processor.tokenizer.pad_token_id]

    # Group IDs by word. The word delimiter is used to split the sequence into words.
    word_groups = [
        list(group) for k, group in groupby(non_padded_timestamped_ids,
                                            lambda x: x[1] == processor.tokenizer.word_delimiter_token_id)
    ]

    # Convert the predicted IDs to phonemes and extract their timestamps.
    timestamped_phonemes = []
    for word in word_groups:
        start_time, token_id = word[0]
        if token_id == 0:
            timestamped_phonemes.append((start_time, "SEP"))
            continue

        for start_time, token_id in word:
            timestamped_phonemes.append((start_time, processor.decode(token_id)))

    # Calculate end timestamps for each phoneme.
    phonemes_with_start_and_end_times = []
    for ix, (start_time, phoneme) in enumerate(timestamped_phonemes):
        if phoneme == "" or phoneme == "SEP":
            continue

        if ix == len(timestamped_phonemes) - 1:
            phonemes_with_start_and_end_times.append((phoneme, start_time, start_time + 0.02))
        else:
            phonemes_with_start_and_end_times.append((phoneme, start_time, timestamped_phonemes[ix + 1][0]))

    # Merge consecutive phonemes that are identical.
    merged_phonemes = merge_consecutive_phonemes(phonemes_with_start_and_end_times)

    return merged_phonemes


if __name__ == '__main__':
    # Set up espeak library for phonemization
    setup_espeak_library_path()

    # Initialize the model and processor
    model_name = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
    processor, model = initialize_model(model_name)

    # Predict phoneme IDs from an audio file
    audio_filepath = os.path.join(os.path.dirname(__file__), '../data/assessment_9.wav')
    predicted_ids, duration_sec = get_predicted_phoneme_ids(audio_filepath, processor, model)

    # Extract phonemes and their timestamps from the predictions
    phonemes_from_audio = get_predicted_phonemes_and_timestamps(predicted_ids, duration_sec, processor)

    # Extract phonemes from a given text transcription
    text = ("But after all that commotion, was it all worthwhile? Absolutely yes! "
            "The set design was breathtaking, the actors were incredible, and the songs were memorable.")
    normalized_text = remove_punctuation(text)
    phonemes_from_transcription = text_to_phonemes(normalized_text)

    # Convert the phonemes from IPA to CMU format
    phonemes_from_transcription_cmu = [IPA_TO_CMU_40[p] for p in phonemes_from_transcription if p != " "]
    phonemes_from_audio_times = [(t_begin, t_end) for _, t_begin, t_end in phonemes_from_audio]
    phonemes_from_audio_cmu = [IPA_TO_CMU_40[p] for p, _, _ in phonemes_from_audio if p != " "]

    # Compute the Levenshtein distance between phonemes of the transcription and audio
    levenshtein_matching = match_transcription_to_predictions(phonemes_from_transcription_cmu, phonemes_from_audio_cmu)

    # Compute and print metrics related to the Levenshtein matching
    compute_levenshtein_matching_metrics(levenshtein_matching)

    # Display the matching results in a markdown table format
    print_table_in_markdown(levenshtein_matching, phonemes_from_audio_cmu, phonemes_from_audio_times)
