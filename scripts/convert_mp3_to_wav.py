import torchaudio
from pydub import AudioSegment


def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 file to WAV format."""
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")


def load_audio(wav_path):
    """Load audio data from WAV file."""
    waveform, sample_rate = torchaudio.load(wav_path, num_frames=-1, offset=0, normalize=True)
    return waveform, sample_rate


def main():
    # File paths
    mp3_path = "data/assessment_9.mp3"
    wav_path = "data/assessment_9.wav"

    # Convert MP3 to WAV
    convert_mp3_to_wav(mp3_path, wav_path)


if __name__ == "__main__":
    main()
