def merge_consecutive(phoneme_data):
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

data = data = [
    ('s', 12.389881222943725, 12.55000892857143),
    ('oʊ', 12.55000892857143, 12.670104707792209),
    ('ŋ', 12.670104707792209, 12.710136634199134),
    ('z', 12.770184523809524, 12.790200487012987),
    ('w', 12.830232413419914, 12.910296266233766),
    ('ɝ', 12.910296266233766, 12.99036011904762),
    ('m', 12.99036011904762, 13.010376082251083),
    ('m', 13.010376082251083, 13.1104558982684),
    ('ɛ', 13.1104558982684, 13.170503787878788),
    ('m', 13.170503787878788, 13.29059956709957),
    ('ə', 13.29059956709957, 13.31061553030303),
]

print("data: ", data)
print(merge_consecutive(data))