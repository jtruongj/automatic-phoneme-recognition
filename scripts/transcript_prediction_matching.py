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

transcript = ['b', 'ˈʌ', 't', ' ', 'ˈæ', 'f', 't', 'ɚ', ' ', 'ˈɔː', 'l', ' ', 'ð', 'ˈæ', 't', ' ', 'k', 'ə', 'm', 'ˈoʊ', 'ʃ', 'ə', 'n', ' ', 'w', 'ˈɑː', 'z', ' ', 'ˈɪ', 't', ' ', 'ˈɔː', 'l', ' ', 'w', 'ˈɚ', 'θ', 'w', 'ˈaɪ', 'l', ' ', 'ˌæ', 'b', 's', 'ə', 'l', 'ˈuː', 't', 'l', 'i', ' ', 'j', 'ˈɛ', 's', ' ', 'ð', 'ə', ' ', 's', 'ˈɛ', 't', ' ', 'd', 'ɪ', 'z', 'ˈaɪ', 'n', ' ', 'w', 'ˈɑː', 'z', ' ', 'b', 'ɹ', 'ˈɛ', 'θ', 't', 'ˌeɪ', 'k', 'ɪ', 'ŋ', ' ', 'ð', 'ə', ' ', 'ˈæ', 'k', 't', 'ɚ', 'z', ' ', 'w', 'ɚ', ' ', 'ɪ', 'n', 'k', 'ɹ', 'ˈɛ', 'd', 'ə', 'b', 'ə', 'l', ' ', 'ə', 'n', 'd', ' ', 'ð', 'ə', ' ', 's', 'ˈɔː', 'ŋ', 'z', ' ', 'w', 'ɚ', ' ', 'm', 'ˈɛ', 'm', 'ɚ', 'ə', 'b', 'ə', 'l']


predictions = 'bəɾæf tɝ ɑlðæ kəmoʊʃɪnwəzɪɾɑl wɝθwaɪlæ b səlu lijɛsðɪsɛ dɪzaɪn wəz b ɹɛθ teɪ kɪŋ ði æ tɝz wɝ ɪn k ɹɛɾə b l ænðəsoʊŋ z wɝmɛmə b l'


text = "But after all that commotion, was it all worthwhile? Absolutely yes! The set design was breathtaking, the actors were incredible, and the songs were memorable."

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

# transcript_cmu = ['B', 'AH', 'T', 'AE', 'F', 'T', 'ER', 'AO', 'L', 'DH', 'AE', 'T', 'K', 'AH', 'M', 'OW', 'SH', 'AH', 'N', 'W', 'AA', 'Z', 'IH', 'T', 'AO', 'L', 'W', 'ER', 'TH', 'W', 'AY', 'L', 'AE', 'B', 'S', 'AH', 'L', 'UW', 'T', 'L', 'IY', 'Y', 'EH', 'S', 'DH', 'AH', 'S', 'EH', 'T', 'D', 'IH', 'Z', 'AY', 'N', 'W', 'AA', 'Z', 'B', 'R', 'EH', 'TH', 'T', 'EY', 'K', 'IH', 'NG', 'DH', 'AH', 'AE', 'K', 'T', 'ER', 'Z', 'W', 'ER', 'IH', 'N', 'K', 'R', 'EH', 'D', 'AH', 'B', 'AH', 'L', 'AH', 'N', 'D', 'DH', 'AH', 'S', 'AO', 'NG', 'Z', 'W', 'ER', 'M', 'EH', 'M', 'ER', 'AH', 'B', 'AH', 'L']
# predictions_cmu = ['B', 'AH', 'T', 'AE', 'F', 'T', 'ER', 'AA', 'L', 'DH', 'AE', 'K', 'AH', 'M', 'OW', 'SH', 'IH', 'N', 'W', 'AH', 'Z', 'IH', 'T', 'AA', 'L', 'W', 'ER', 'TH', 'W', 'AY', 'L', 'AE', 'B', 'S', 'AH', 'L', 'UW', 'L', 'IY', 'Y', 'EH', 'S', 'DH', 'IH', 'S', 'EH', 'D', 'IH', 'Z', 'AY', 'N', 'W', 'AH', 'Z', 'B', 'R', 'EH', 'TH', 'T', 'EY', 'K', 'IH', 'NG', 'DH', 'IY', 'AE', 'T', 'ER', 'Z', 'W', 'ER', 'IH', 'N', 'K', 'R', 'EH', 'T', 'AH', 'B', 'L', 'AE', 'N', 'DH', 'AH', 'S', 'OW', 'NG', 'Z', 'W', 'ER', 'M', 'EH', 'M', 'AH', 'B', 'L']


if __name__ == "__main__":
    print("Raw transcript: ", transcript)
    print("Raw predictions: ", predictions)
    
    transcript_cmu = [IPA_TO_CMU_40[p] for p in transcript if p != " "] # convert to CMU phonemes

    predictions_cmu = [p for p in pred_to_CMU(predictions) if p != " "]
    
    print("Transcript:  ", transcript_cmu)
    print("Predictions: ", predictions_cmu)

    
    # go through each pair of phonemes in transcript_cmu and predictions_cmu but keep printing if one runs out:
    print("Pairs of phonemes:")
    for t, p in zip_longest(transcript_cmu, predictions_cmu, fillvalue=""):
        print(f"{t} {p}")



    result = compare_phonemes(transcript_cmu, predictions_cmu)
    for res in result:
        print(res)
