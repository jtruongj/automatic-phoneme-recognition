# Dictionary found here:
# https://github.com/Kyubyong/g2p/issues/29#issuecomment-1366493750
CMU_TO_IPA = {
    '<pad>': '<pad>',
    '<unk>': '<unk>',
    '<s>': '<s>',
    '</s>': '</s>',
    'AA0': 'ɑ',
    'AA1': 'ˈɑː',
    'AA2': 'ˌɑ',
    'AE0': 'æ',
    'AE1': 'ˈæ',
    'AE2': 'ˌæ',
    'AH0': 'ə',
    'AH1': 'ˈʌ',
    'AH2': 'ˌʌ',
    'AO0': 'ɔ',
    'AO1': 'ˈɔː',
    'AO2': 'ˌɔ',
    'AW0': 'aʊ',
    'AW1': 'ˈaʊ',
    'AW2': 'ˌaʊ',
    'AY0': 'aɪ',
    'AY1': 'ˈaɪ',
    'AY2': 'ˌaɪ',
    'B': 'b',
    'CH': 'tʃ',
    'D': 'd',
    'DH': 'ð',
    'EH0': 'ɛ',
    'EH1': 'ˈɛ',
    'EH2': 'ˌɛ',
    'ER0': 'ɚ',
    'ER1': 'ˈɚ',
    'ER2': 'ˌɚ',
    'EY0': 'eɪ',
    'EY1': 'ˈeɪ',
    'EY2': 'ˌeɪ',
    'F': 'f',
    'G': 'g',
    'HH': 'h',
    'IH0': 'ɪ',
    'IH1': 'ˈɪ',
    'IH2': 'ˌɪ',
    'IY0': 'i',
    'IY1': 'ˈi:',
    'IY2': 'ˌi',
    'JH': 'dʒ',
    'K': 'k',
    'L': 'l',
    'M': 'm',
    'N': 'n',
    'NG': 'ŋ',
    'OW0': 'oʊ',
    'OW1': 'ˈoʊ',
    'OW2': 'ˌoʊ',
    'OY0': 'ɔɪ',
    'OY1': 'ˈɔɪ',
    'OY2': 'ˌɔɪ',
    'P': 'p',
    'R': 'ɹ',
    'S': 's',
    'SH': 'ʃ',
    'T': 't',
    'TH': 'θ',
    'UH0': 'ʊ',
    'UH1': 'ˈʊ',
    'UH2': 'ˌʊ',
    # 'UW': 'u:',
    'UW0': 'u',
    'UW1': 'ˈuː',
    'UW2': 'ˌu',
    'V': 'v',
    'W': 'w',
    'Y': 'j',
    'Z': 'z',
    'ZH': 'ʒ',
    ' ': ' '
}

# Create inverse mapping
IPA_TO_CMU_40 = {}

for k, v in CMU_TO_IPA.items():
    # Remove stress markers
    if k[-1] in ['0', '1', '2']:
        k = k[:-1]
    IPA_TO_CMU_40[v] = k

# Add special cases for incompatible between IPA and the model's vocabulary
IPA_TO_CMU_40["ɾ"] = "T"
IPA_TO_CMU_40["ɝ"] = "ER"

# Delete special tokens
del IPA_TO_CMU_40['<pad>']
del IPA_TO_CMU_40['<unk>']
del IPA_TO_CMU_40['<s>']
del IPA_TO_CMU_40['</s>']
# Note that the phoneme " " is encoded SIL in the CMU dictionary but left as a space here.

assert len(set(IPA_TO_CMU_40.values())) == 40
