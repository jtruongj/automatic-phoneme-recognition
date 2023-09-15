# Phoneme Recognition and Speech Timestamping Project

Problem: Given an audio recording and its transcription, determine the beginning and end timestamps of each expected phoneme as heard in the recording.

## Results
Below is a table showing all the phonemes from the transcription, as converted using the CMU dictionary. Each phoneme is associated with:
- the phoneme predicted by the model
- the type of matching between the transcribed phoneme and the predicted one (correct, substitution, deletion, or addition)
- the begining and end of the timestamp for this phoneme

Additionally here are a few additional metrics on how the predicted and transcribed phonemes match.

Percent correct: 82.6%
Percent substitution: 9.7%
Percent insertion: 0.0%
Percent deletion: 7.7%


| Transcribed Phoneme   | Predicted Phoneme   | Result       | Begin Time         | End Time           |
|:----------------------|:--------------------|:-------------|:-------------------|:-------------------|
| B                     | B                   | correct      | 0.6004788961038962 | 0.6405108225108226 |
| AH                    | AH                  | correct      | 0.6405108225108226 | 0.680542748917749  |
| T                     | T                   | correct      | 0.680542748917749  | 0.780622564935065  |
| AE                    | AE                  | correct      | 0.780622564935065  | 0.8406704545454546 |
| F                     | F                   | correct      | 0.8406704545454546 | 0.8606864177489177 |
| T                     | T                   | correct      | 0.9007183441558441 | 0.9407502705627706 |
| ER                    | ER                  | correct      | 0.9407502705627706 | 1.0007981601731601 |
| AO                    | AA                  | substitution | 1.2209737554112554 | 1.2810216450216452 |
| L                     | L                   | correct      | 1.2810216450216452 | 1.361085497835498  |
| DH                    | DH                  | correct      | 1.361085497835498  | 1.461165313852814  |
| AE                    | AE                  | correct      | 1.461165313852814  | 1.5212132034632035 |
| T                     |                     | deletion     | -                  | -                  |
| K                     | K                   | correct      | 1.581261093073593  | 1.6613249458874462 |
| AH                    | AH                  | correct      | 1.6613249458874462 | 1.7013568722943724 |
| M                     | M                   | correct      | 1.7013568722943724 | 1.821452651515152  |
| OW                    | OW                  | correct      | 1.821452651515152  | 1.9215324675324676 |
| SH                    | SH                  | correct      | 1.9215324675324676 | 2.041628246753247  |
| AH                    | IH                  | substitution | 2.041628246753247  | 2.081660173160173  |
| N                     | N                   | correct      | 2.081660173160173  | 2.882298701298702  |
| W                     | W                   | correct      | 2.882298701298702  | 2.9823785173160173 |
| AA                    | AH                  | substitution | 2.9823785173160173 | 3.042426406926407  |
| Z                     | Z                   | correct      | 3.042426406926407  | 3.102474296536797  |
| IH                    | IH                  | correct      | 3.102474296536797  | 3.142506222943723  |
| T                     | T                   | correct      | 3.142506222943723  | 3.2826179653679652 |
| AO                    | AA                  | substitution | 3.2826179653679652 | 3.3226498917748923 |
| L                     | L                   | correct      | 3.3226498917748923 | 3.3626818181818185 |
| W                     | W                   | correct      | 3.4027137445887448 | 3.502793560606061  |
| ER                    | ER                  | correct      | 3.502793560606061  | 3.6028733766233763 |
| TH                    | TH                  | correct      | 3.6028733766233763 | 3.7029531926406927 |
| W                     | W                   | correct      | 3.7029531926406927 | 3.863080898268399  |
| AY                    | AY                  | correct      | 3.863080898268399  | 3.9631607142857144 |
| L                     | L                   | correct      | 3.9631607142857144 | 5.564437770562771  |
| AE                    | AE                  | correct      | 5.564437770562771  | 5.644501623376623  |
| B                     | B                   | correct      | 5.704549512987013  | 5.7245654761904765 |
| S                     | S                   | correct      | 5.784613365800866  | 5.884693181818182  |
| AH                    | AH                  | correct      | 5.884693181818182  | 5.964757034632035  |
| L                     | L                   | correct      | 5.964757034632035  | 6.0648368506493515 |
| UW                    | UW                  | correct      | 6.0648368506493515 | 6.12488474025974   |
| T                     |                     | deletion     | -                  | -                  |
| L                     | L                   | correct      | 6.204948593073594  | 6.264996482683983  |
| IY                    | IY                  | correct      | 6.264996482683983  | 6.365076298701299  |
| Y                     | Y                   | correct      | 6.365076298701299  | 6.485172077922078  |
| EH                    | EH                  | correct      | 6.485172077922078  | 6.665315746753247  |
| S                     | S                   | correct      | 6.665315746753247  | 7.485970238095238  |
| DH                    | DH                  | correct      | 7.485970238095238  | 7.546018127705628  |
| AH                    | IH                  | substitution | 7.546018127705628  | 7.6460979437229435 |
| S                     | S                   | correct      | 7.6460979437229435 | 7.786209686147187  |
| EH                    | EH                  | correct      | 7.786209686147187  | 7.826241612554113  |
| T                     |                     | deletion     | -                  | -                  |
| D                     | D                   | correct      | 7.886289502164503  | 7.946337391774892  |
| IH                    | IH                  | correct      | 7.946337391774892  | 8.026401244588746  |
| Z                     | Z                   | correct      | 8.026401244588746  | 8.206544913419913  |
| AY                    | AY                  | correct      | 8.206544913419913  | 8.326640692640693  |
| N                     | N                   | correct      | 8.326640692640693  | 8.406704545454547  |
| W                     | W                   | correct      | 8.4867683982684    | 8.546816287878787  |
| AA                    | AH                  | substitution | 8.546816287878787  | 8.586848214285714  |
| Z                     | Z                   | correct      | 8.586848214285714  | 8.646896103896104  |
| B                     | B                   | correct      | 8.726959956709957  | 8.746975919913421  |
| R                     | R                   | correct      | 8.807023809523809  | 8.907103625541126  |
| EH                    | EH                  | correct      | 8.907103625541126  | 8.987167478354978  |
| TH                    | TH                  | correct      | 8.987167478354978  | 9.027199404761905  |
| T                     | T                   | correct      | 9.06723133116883   | 9.167311147186147  |
| EY                    | EY                  | correct      | 9.167311147186147  | 9.207343073593075  |
| K                     | K                   | correct      | 9.267390963203464  | 9.36747077922078   |
| IH                    | IH                  | correct      | 9.36747077922078   | 9.407502705627707  |
| NG                    | NG                  | correct      | 9.407502705627707  | 9.507582521645022  |
| DH                    | DH                  | correct      | 9.787806006493506  | 9.847853896103897  |
| AH                    | IY                  | substitution | 9.847853896103897  | 9.907901785714287  |
| AE                    | AE                  | correct      | 10.007981601731602 | 10.068029491341992 |
| K                     |                     | deletion     | -                  | -                  |
| T                     | T                   | correct      | 10.148093344155845 | 10.228157196969697 |
| ER                    | ER                  | correct      | 10.228157196969697 | 10.328237012987014 |
| Z                     | Z                   | correct      | 10.328237012987014 | 10.368268939393941 |
| W                     | W                   | correct      | 10.428316829004329 | 10.568428571428571 |
| ER                    | ER                  | correct      | 10.568428571428571 | 10.68852435064935  |
| IH                    | IH                  | correct      | 10.828636093073593 | 10.86866801948052  |
| N                     | N                   | correct      | 10.86866801948052  | 10.908699945887447 |
| K                     | K                   | correct      | 10.948731872294372 | 10.9887637987013   |
| R                     | R                   | correct      | 11.068827651515152 | 11.168907467532469 |
| EH                    | EH                  | correct      | 11.168907467532469 | 11.228955357142858 |
| D                     | T                   | substitution | 11.228955357142858 | 11.268987283549784 |
| AH                    | AH                  | correct      | 11.268987283549784 | 11.30901920995671  |
| B                     | B                   | correct      | 11.349051136363638 | 11.409099025974026 |
| AH                    |                     | deletion     | -                  | -                  |
| L                     | L                   | correct      | 11.449130952380953 | 11.569226731601733 |
| AH                    | AE                  | substitution | 12.129673701298703 | 12.189721590909091 |
| N                     | N                   | correct      | 12.189721590909091 | 12.24976948051948  |
| D                     |                     | deletion     | -                  | -                  |
| DH                    | DH                  | correct      | 12.24976948051948  | 12.289801406926408 |
| AH                    | AH                  | correct      | 12.289801406926408 | 12.389881222943725 |
| S                     | S                   | correct      | 12.389881222943725 | 12.55000892857143  |
| AO                    | OW                  | substitution | 12.55000892857143  | 12.670104707792209 |
| NG                    | NG                  | correct      | 12.670104707792209 | 12.710136634199134 |
| Z                     | Z                   | correct      | 12.770184523809524 | 12.790200487012987 |
| W                     | W                   | correct      | 12.830232413419914 | 12.910296266233766 |
| ER                    | ER                  | correct      | 12.910296266233766 | 12.99036011904762  |
| M                     | M                   | correct      | 12.99036011904762  | 13.1104558982684   |
| EH                    | EH                  | correct      | 13.1104558982684   | 13.170503787878788 |
| M                     | M                   | correct      | 13.170503787878788 | 13.29059956709957  |
| ER                    |                     | deletion     | -                  | -                  |
| AH                    | AH                  | correct      | 13.29059956709957  | 13.31061553030303  |
| B                     | B                   | correct      | 13.370663419913422 | 13.410695346320347 |
| AH                    |                     | deletion     | -                  | -                  |
| L                     | L                   | correct      | 13.450727272727274 | 13.831030573593074 |

## Approach description
To solve this problem, I used a pre-trained Wav2Vec2 with CTC model to predict phonemes. The default vocabulary used the IPA phoneme notation, I used a mapping table to map them back to the phonemes used in the CMU classification.

Since the model splits the audio into pieces of about 20ms this is the maximum resolution that I can get to predict the timestamps. I used the timestamps of each phoneme, as well as those of each empty string tokens ('') and word separators to determine the end timestamp of each phoneme.

In addition to this, I used the CMU dictionary to extract phonemes from the provided transcription.

Lastly, I implemented a Levenshtein distance-measuring algorithm to match the transcribed phonemes to the predicted ones. This finally results in the table above.

Note: while this code works on the audio recordings that I have tested, it is possible that the code breaks on some other recordings. The reason is that the phoneme vocabulary used by the underlying model is different than that I found online. Therefore, it is possible that the IPA -> CMU phoneme translation break. This can be resolved by adding an entry in the dictionary, mapping the IPA phoneme that yields an IndexError to the dictionary in `scripts/globals.py`


## Installation
### Prerequisites

Before running the installation steps, ensure that you have Python 3.x installed on your machine.

### Steps

1. **Clone the repository**:
   
   ```bash
   git clone git@github.com:jtruongj/automatic-phoneme-recognition.git
   cd automatic-phoneme-recognition
   ```

2. **Create a virtual environment**:
   
   Using `virtualenv`:
   
   ```bash
   python3 -m venv env
   ```

   Using `conda` (if you're using Anaconda or Miniconda):
   
   ```bash
   conda create --name env python=3.x
   ```

3. **Activate the virtual environment**:

   For `virtualenv`:
   
   ```bash
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

   For `conda`:

   ```bash
   conda activate env
   ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Install `espeak`**:

   - On macOS:
     
     ```bash
     brew install espeak
     ```

   - On Linux:

     ```bash
     sudo apt-get install espeak
     ```

## Usage
1. **Convert MP3 file to WAV**:

   ```bash
   python scripts/convert_mp3_to_wav.py
   ```

2. **Phoneme Recognition**:

   ```bash
   python scripts/phoneme_regognition.py
   ```

3. **Word Extraction with Timestamps**:

   ```bash
   python scripts/speech_recognition_with_timestamps.py
   ```

