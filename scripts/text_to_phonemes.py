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

if __name__ == '__main__':
    
    text = "But after all that commotion, was it all worthwhile? Absolutely yes! The set design was breathtaking, the actors were incredible, and the songs were memorable."

    normalized_text = remove_punctuation(text)
    print(normalized_text)

    print(text_to_phonemes(normalized_text))

