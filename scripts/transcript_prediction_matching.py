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


predictions_old = 'bəɾæf tɝ ɑlðæ kəmoʊʃɪnwəzɪɾɑl wɝθwaɪlæ b səlu lijɛsðɪsɛ dɪzaɪn wəz b ɹɛθ teɪ kɪŋ ði æ tɝz wɝ ɪn k ɹɛɾə b l ænðəsoʊŋ z wɝmɛmə b l'

predictions = [('b', 0.6004788961038962, 0.6405108225108226), ('ə', 0.6405108225108226, 0.680542748917749), ('ɾ', 0.680542748917749, 0.780622564935065), ('æ', 0.780622564935065, 0.8406704545454546), ('f', 0.8406704545454546, 0.8606864177489177), ('t', 0.9007183441558441, 0.9407502705627706), ('ɝ', 0.9407502705627706, 1.0007981601731601), ('ɑ', 1.2209737554112554, 1.2810216450216452), ('l', 1.2810216450216452, 1.361085497835498), ('ð', 1.361085497835498, 1.461165313852814), ('æ', 1.461165313852814, 1.5212132034632035), ('k', 1.581261093073593, 1.6613249458874462), ('ə', 1.6613249458874462, 1.7013568722943724), ('m', 1.7013568722943724, 1.821452651515152), ('oʊ', 1.821452651515152, 1.9215324675324676), ('ʃ', 1.9215324675324676, 2.041628246753247), ('ɪ', 2.041628246753247, 2.081660173160173), ('n', 2.081660173160173, 2.882298701298702), ('w', 2.882298701298702, 2.9823785173160173), ('ə', 2.9823785173160173, 3.042426406926407), ('z', 3.042426406926407, 3.102474296536797), ('ɪ', 3.102474296536797, 3.142506222943723), ('ɾ', 3.142506222943723, 3.2826179653679652), ('ɑ', 3.2826179653679652, 3.3226498917748923), ('l', 3.3226498917748923, 3.3626818181818185), ('w', 3.4027137445887448, 3.502793560606061), ('ɝ', 3.502793560606061, 3.6028733766233763), ('θ', 3.6028733766233763, 3.7029531926406927), ('w', 3.7029531926406927, 3.863080898268399), ('aɪ', 3.863080898268399, 3.9631607142857144), ('l', 3.9631607142857144, 5.564437770562771), ('æ', 5.564437770562771, 5.644501623376623), ('b', 5.704549512987013, 5.7245654761904765), ('s', 5.784613365800866, 5.884693181818182), ('ə', 5.884693181818182, 5.964757034632035), ('l', 5.964757034632035, 6.0648368506493515), ('u', 6.0648368506493515, 6.12488474025974), ('l', 6.204948593073594, 6.264996482683983), ('i', 6.264996482683983, 6.365076298701299), ('j', 6.365076298701299, 6.485172077922078), ('ɛ', 6.485172077922078, 6.665315746753247), ('s', 6.665315746753247, 7.485970238095238), ('ð', 7.485970238095238, 7.546018127705628), ('ɪ', 7.546018127705628, 7.6460979437229435), ('s', 7.6460979437229435, 7.786209686147187), ('ɛ', 7.786209686147187, 7.826241612554113), ('d', 7.886289502164503, 7.946337391774892), ('ɪ', 7.946337391774892, 8.026401244588746), ('z', 8.026401244588746, 8.206544913419913), ('aɪ', 8.206544913419913, 8.326640692640693), ('n', 8.326640692640693, 8.406704545454547), ('w', 8.4867683982684, 8.546816287878787), ('ə', 8.546816287878787, 8.586848214285714), ('z', 8.586848214285714, 8.646896103896104), ('b', 8.726959956709957, 8.746975919913421), ('ɹ', 8.807023809523809, 8.907103625541126), ('ɛ', 8.907103625541126, 8.987167478354978), ('θ', 8.987167478354978, 9.027199404761905), ('t', 9.06723133116883, 9.167311147186147), ('eɪ', 9.167311147186147, 9.207343073593075), ('k', 9.267390963203464, 9.36747077922078), ('ɪ', 9.36747077922078, 9.407502705627707), ('ŋ', 9.407502705627707, 9.507582521645022), ('ð', 9.787806006493506, 9.847853896103897), ('i', 9.847853896103897, 9.907901785714287), ('æ', 10.007981601731602, 10.068029491341992), ('t', 10.148093344155845, 10.228157196969697), ('ɝ', 10.228157196969697, 10.328237012987014), ('z', 10.328237012987014, 10.368268939393941), ('w', 10.428316829004329, 10.568428571428571), ('ɝ', 10.568428571428571, 10.68852435064935), ('ɪ', 10.828636093073593, 10.86866801948052), ('n', 10.86866801948052, 10.908699945887447), ('k', 10.948731872294372, 10.9887637987013), ('ɹ', 11.068827651515152, 11.168907467532469), ('ɛ', 11.168907467532469, 11.228955357142858), ('ɾ', 11.228955357142858, 11.268987283549784), ('ə', 11.268987283549784, 11.30901920995671), ('b', 11.349051136363638, 11.409099025974026), ('l', 11.449130952380953, 11.569226731601733), ('æ', 12.129673701298703, 12.189721590909091), ('n', 12.189721590909091, 12.24976948051948), ('ð', 12.24976948051948, 12.289801406926408), ('ə', 12.289801406926408, 12.389881222943725), ('s', 12.389881222943725, 12.55000892857143), ('oʊ', 12.55000892857143, 12.670104707792209), ('ŋ', 12.670104707792209, 12.710136634199134), ('z', 12.770184523809524, 12.790200487012987), ('w', 12.830232413419914, 12.910296266233766), ('ɝ', 12.910296266233766, 12.99036011904762), ('m', 12.99036011904762, 13.1104558982684), ('ɛ', 13.1104558982684, 13.170503787878788), ('m', 13.170503787878788, 13.29059956709957), ('ə', 13.29059956709957, 13.31061553030303), ('b', 13.370663419913422, 13.410695346320347), ('l', 13.450727272727274, 13.831030573593074)]

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

    prediction_times = [(t_begin, t_end) for _, t_begin, t_end in predictions]
    predictions_cmu = [IPA_TO_CMU_40[p] for p, _, _ in predictions if p != " "]
    
    print("Transcript:  ", transcript_cmu)
    print("Predictions: ", predictions_cmu)

    
    # go through each pair of phonemes in transcript_cmu and predictions_cmu but keep printing if one runs out:
    print("Pairs of phonemes:")
    for t, p in zip_longest(transcript_cmu, predictions_cmu, fillvalue=""):
        print(f"{t} {p}")



    result = compare_phonemes(transcript_cmu, predictions_cmu)
    for res in result:
        print(res)

    print("Percent correct",  sum([1 for x in filter(lambda x: x[2] == "correct", result)]) / len(result))
    print("Percent substitution",  sum([1 for x in filter(lambda x: x[2] == "substitution", result)]) / len(result))
    print("Percent insertion",  sum([1 for x in filter(lambda x: x[2] == "insertion", result)]) / len(result))
    print("Percent deletion",  sum([1 for x in filter(lambda x: x[2] == "deletion", result)]) / len(result))