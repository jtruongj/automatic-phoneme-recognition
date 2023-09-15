# Phoneme Recognition and Speech Timestamping Project

**Problem**: Given an audio recording and its transcription, determine the beginning and end timestamps of each expected phoneme as heard in the recording.

## Results
The table below represents phonemes from the transcription, converted using the CMU dictionary. For each phoneme, the table displays:
- CMU dictionary-based transcribed phoneme
- Model's predicted phoneme
- Match type with the transcribed phoneme (correct, substitution, deletion, or addition)
- Start and end timestamps

**Performance Metrics**:
- Correct: 82.6%
- Substitution: 9.7%
- Insertion: 0.0%
- Deletion: 7.7%


| Transcribed Phoneme   | Predicted Phoneme   | Result       | Begin Time   | End Time   |
|:----------------------|:--------------------|:-------------|:-------------|:-----------|
| B                     | B                   | correct      | 0.6          | 0.641      |
| AH                    | AH                  | correct      | 0.641        | 0.681      |
| T                     | T                   | correct      | 0.681        | 0.781      |
| AE                    | AE                  | correct      | 0.781        | 0.841      |
| F                     | F                   | correct      | 0.841        | 0.861      |
| T                     | T                   | correct      | 0.901        | 0.941      |
| ER                    | ER                  | correct      | 0.941        | 1.001      |
| AO                    | AA                  | substitution | 1.221        | 1.281      |
| L                     | L                   | correct      | 1.281        | 1.361      |
| DH                    | DH                  | correct      | 1.361        | 1.461      |
| AE                    | AE                  | correct      | 1.461        | 1.521      |
| T                     |                     | deletion     | -            | -          |
| K                     | K                   | correct      | 1.581        | 1.661      |
| AH                    | AH                  | correct      | 1.661        | 1.701      |
| M                     | M                   | correct      | 1.701        | 1.821      |
| OW                    | OW                  | correct      | 1.821        | 1.922      |
| SH                    | SH                  | correct      | 1.922        | 2.042      |
| AH                    | IH                  | substitution | 2.042        | 2.082      |
| N                     | N                   | correct      | 2.082        | 2.882      |
| W                     | W                   | correct      | 2.882        | 2.982      |
| AA                    | AH                  | substitution | 2.982        | 3.042      |
| Z                     | Z                   | correct      | 3.042        | 3.102      |
| IH                    | IH                  | correct      | 3.102        | 3.143      |
| T                     | T                   | correct      | 3.143        | 3.283      |
| AO                    | AA                  | substitution | 3.283        | 3.323      |
| L                     | L                   | correct      | 3.323        | 3.363      |
| W                     | W                   | correct      | 3.403        | 3.503      |
| ER                    | ER                  | correct      | 3.503        | 3.603      |
| TH                    | TH                  | correct      | 3.603        | 3.703      |
| W                     | W                   | correct      | 3.703        | 3.863      |
| AY                    | AY                  | correct      | 3.863        | 3.963      |
| L                     | L                   | correct      | 3.963        | 5.564      |
| AE                    | AE                  | correct      | 5.564        | 5.645      |
| B                     | B                   | correct      | 5.705        | 5.725      |
| S                     | S                   | correct      | 5.785        | 5.885      |
| AH                    | AH                  | correct      | 5.885        | 5.965      |
| L                     | L                   | correct      | 5.965        | 6.065      |
| UW                    | UW                  | correct      | 6.065        | 6.125      |
| T                     |                     | deletion     | -            | -          |
| L                     | L                   | correct      | 6.205        | 6.265      |
| IY                    | IY                  | correct      | 6.265        | 6.365      |
| Y                     | Y                   | correct      | 6.365        | 6.485      |
| EH                    | EH                  | correct      | 6.485        | 6.665      |
| S                     | S                   | correct      | 6.665        | 7.486      |
| DH                    | DH                  | correct      | 7.486        | 7.546      |
| AH                    | IH                  | substitution | 7.546        | 7.646      |
| S                     | S                   | correct      | 7.646        | 7.786      |
| EH                    | EH                  | correct      | 7.786        | 7.826      |
| T                     |                     | deletion     | -            | -          |
| D                     | D                   | correct      | 7.886        | 7.946      |
| IH                    | IH                  | correct      | 7.946        | 8.026      |
| Z                     | Z                   | correct      | 8.026        | 8.207      |
| AY                    | AY                  | correct      | 8.207        | 8.327      |
| N                     | N                   | correct      | 8.327        | 8.407      |
| W                     | W                   | correct      | 8.487        | 8.547      |
| AA                    | AH                  | substitution | 8.547        | 8.587      |
| Z                     | Z                   | correct      | 8.587        | 8.647      |
| B                     | B                   | correct      | 8.727        | 8.747      |
| R                     | R                   | correct      | 8.807        | 8.907      |
| EH                    | EH                  | correct      | 8.907        | 8.987      |
| TH                    | TH                  | correct      | 8.987        | 9.027      |
| T                     | T                   | correct      | 9.067        | 9.167      |
| EY                    | EY                  | correct      | 9.167        | 9.207      |
| K                     | K                   | correct      | 9.267        | 9.367      |
| IH                    | IH                  | correct      | 9.367        | 9.408      |
| NG                    | NG                  | correct      | 9.408        | 9.508      |
| DH                    | DH                  | correct      | 9.788        | 9.848      |
| AH                    | IY                  | substitution | 9.848        | 9.908      |
| AE                    | AE                  | correct      | 10.008       | 10.068     |
| K                     |                     | deletion     | -            | -          |
| T                     | T                   | correct      | 10.148       | 10.228     |
| ER                    | ER                  | correct      | 10.228       | 10.328     |
| Z                     | Z                   | correct      | 10.328       | 10.368     |
| W                     | W                   | correct      | 10.428       | 10.568     |
| ER                    | ER                  | correct      | 10.568       | 10.689     |
| IH                    | IH                  | correct      | 10.829       | 10.869     |
| N                     | N                   | correct      | 10.869       | 10.909     |
| K                     | K                   | correct      | 10.949       | 10.989     |
| R                     | R                   | correct      | 11.069       | 11.169     |
| EH                    | EH                  | correct      | 11.169       | 11.229     |
| D                     | T                   | substitution | 11.229       | 11.269     |
| AH                    | AH                  | correct      | 11.269       | 11.309     |
| B                     | B                   | correct      | 11.349       | 11.409     |
| AH                    |                     | deletion     | -            | -          |
| L                     | L                   | correct      | 11.449       | 11.569     |
| AH                    | AE                  | substitution | 12.13        | 12.19      |
| N                     | N                   | correct      | 12.19        | 12.25      |
| D                     |                     | deletion     | -            | -          |
| DH                    | DH                  | correct      | 12.25        | 12.29      |
| AH                    | AH                  | correct      | 12.29        | 12.39      |
| S                     | S                   | correct      | 12.39        | 12.55      |
| AO                    | OW                  | substitution | 12.55        | 12.67      |
| NG                    | NG                  | correct      | 12.67        | 12.71      |
| Z                     | Z                   | correct      | 12.77        | 12.79      |
| W                     | W                   | correct      | 12.83        | 12.91      |
| ER                    | ER                  | correct      | 12.91        | 12.99      |
| M                     | M                   | correct      | 12.99        | 13.11      |
| EH                    | EH                  | correct      | 13.11        | 13.171     |
| M                     | M                   | correct      | 13.171       | 13.291     |
| ER                    |                     | deletion     | -            | -          |
| AH                    | AH                  | correct      | 13.291       | 13.311     |
| B                     | B                   | correct      | 13.371       | 13.411     |
| AH                    |                     | deletion     | -            | -          |
| L                     | L                   | correct      | 13.451       | 13.831     |

#### Evaluation of Results
The model's performance is commendable, achieving 82.6% accuracy. While some phoneme substitutions are understandable due to phonetic similarities, they highlight potential areas for refinement. A comprehensive evaluation would provide a clearer picture.

It's worth noting that certain phonemes, notably the initial `N`, have an extended duration of approximately 800 ms. This anomaly could be attributed to the model's infrequent predictions of the "word delimiter token." Further model fine-tuning might address this.

## Methodology
I employed the pre-trained Wav2Vec2 with CTC model to predict phonemes. By default, this model utilizes the IPA phoneme notation. To align it with our needs, I used a mapping table to convert them to the CMU classification.

Considering the model's segmentation of audio into ~20ms units, this granularity dictated our timestamp resolution. I utilized the timestamps of each phoneme, empty string tokens (''), and word separators to determine the precise ending timestamp for each phoneme.

Further, the CMU dictionary served as a valuable resource for extracting phonemes from the provided transcription. To ensure an accurate correlation between transcribed and predicted phonemes, I integrated a Levenshtein distance algorithm. This algorithm helps find the best match between transcription and prediciton, and it lead to the aforementioned results table.

**Note**: The code functions well with most tested audio recordings. However, potential discrepancies might arise due to variations in phoneme vocabulary between different sources. In cases where the IPA -> CMU phoneme mapping is absent, you can remedy this by updating the dictionary in `scripts/globals.py`. Phonemes such as `ʧ` or `ʤ` are currently absent.

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

2. **Phoneme Recognition with Timestamps**:

   ```bash
   python scripts/automatic_phoneme_timestamp_extraction.py
   ```


