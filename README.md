# Phoneme Recognition and Speech Timestamping Project

This project provides tools for phoneme recognition, word extraction with timestamps, and MP3 to WAV file conversion. It uses deep learning models and tools to achieve this.

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

