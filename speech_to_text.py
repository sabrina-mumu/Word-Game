import speech_recognition as sr
from pydub import AudioSegment, silence
import os
import requests
from datetime import datetime
import mimetypes
import validators


class CustomException(Exception):
    """A custom exception class for handling specific error types in the transcription process."""
    
    def __init__(self, message, error_type):
        """Initialize the CustomException with a message and an error type.

        Args:
            message (str): Error message describing the issue.
            error_type (str): Type of error, e.g., "audio_error", "network_error".
        """
        super().__init__(message)
        self.error_type = error_type

    def __str__(self):
        """Return only the message part when the exception is printed."""
        return self.args[0]


class Transcriber:
    """Handles downloading, processing, and transcribing audio files from a given URL."""
    
    def __init__(self, url, silence_thresh=-50, min_silence_len=500, sample_rate=44100):
        """Initialize the Transcriber with the audio URL and optional audio processing parameters.

        Args:
            url (str): URL of the audio file to download and transcribe.
            silence_thresh (int, optional): Silence threshold (in dB) for detecting silence. Defaults to -50.
            min_silence_len (int, optional): Minimum length of silence (in ms) to split audio chunks. Defaults to 500.
            sample_rate (int, optional): Sample rate for audio processing. Defaults to 44100.
        """
        self.url = url
        self.audio_folder = "audio"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_extension = self.get_file_extension()
        self.file_path = os.path.join(self.audio_folder, f"downloaded_audio_{self.timestamp}.{self.file_extension}")
        self.temp_wav_file = f"temp_audio_{self.timestamp}.wav"
        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len
        self.sample_rate = sample_rate

    def get_file_extension(self):
        """Retrieve the file extension from the URL or infer from the content type.

        Returns:
            str: The file extension of the audio file.

        Raises:
            CustomException: If unable to determine the file extension.
        """
        try:
            ext = os.path.splitext(self.url)[-1].replace(".", "")
            if not ext:
                ext = mimetypes.guess_extension(requests.head(self.url).headers["content-type"]).replace(".", "")
            return ext
        except Exception as e:
            raise CustomException(f"Error determining file extension: {str(e)}", "audio_error")

    def is_valid_url(self):
        """Checks if the provided URL is valid and supported for audio transcription.

        Returns:
            bool: True if URL is valid and audio format is supported; False otherwise.
        """
        return self.file_extension in ['mp3', 'avi', 'ogg'] and validators.url(self.url)

    def download_audio(self):
        """Download audio file from the URL and save it locally.

        Raises:
            CustomException: If the URL is invalid or download fails.
        """
        try:
            if not self.is_valid_url():
                raise CustomException("Invalid URL. Supported formats are .mp3, .avi, and .ogg.", "audio_error")

            if not os.path.exists(self.audio_folder):
                os.makedirs(self.audio_folder)

            response = requests.get(self.url)
            if response.status_code == 200:
                with open(self.file_path, "wb") as audio_file:
                    audio_file.write(response.content)
            else:
                raise CustomException(f"Failed to download audio file. Status code: {response.status_code}", "audio_error")

        except requests.RequestException as e:
            raise CustomException(f"Network error occurred: {str(e)}", "network_error")

        except Exception as e:
            raise CustomException(f"Unexpected error occurred: {str(e)}", "audio_error")

    def convert_to_wav(self):
        """Convert the downloaded audio file to WAV format with specified sample rate.

        Raises:
            CustomException: If the conversion process fails.
        """
        try:
            sound = AudioSegment.from_file(self.file_path)
            sound = sound.set_frame_rate(self.sample_rate)
            sound.export(self.temp_wav_file, format="wav")
        except Exception as e:
            raise CustomException(f"Error converting audio to WAV: {str(e)}", "audio_error")

    def remove_silence(self, audio):
        """Remove silent portions of the audio using defined thresholds.

        Args:
            audio (AudioSegment): The loaded audio segment from which silence will be removed.

        Returns:
            list: List of non-silent audio chunks.

        Raises:
            CustomException: If silence removal fails.
        """
        try:
            non_silent_chunks = silence.split_on_silence(
                audio,
                min_silence_len=self.min_silence_len,
                silence_thresh=self.silence_thresh
            )
            return non_silent_chunks
        except Exception as e:
            raise CustomException(f"Error removing silence: {str(e)}", "audio_error")

    def transcribe_audio_segment(self, audio_segment, chunk_number):
        """Transcribe a single audio segment using Google Speech Recognition API.

        Args:
            audio_segment (AudioSegment): An audio segment to be transcribed.
            chunk_number (int): The sequence number of the audio chunk being transcribed.

        Returns:
            str: Transcribed text for the audio segment.

        Raises:
            CustomException: If API request for transcription fails.
        """
        recognizer = sr.Recognizer()
        temp_segment = f"temp_segment_{self.timestamp}_{chunk_number}.wav"
        audio_segment.export(temp_segment, format="wav")

        try:
            with sr.AudioFile(temp_segment) as source:
                audio_data = recognizer.record(source)
                return recognizer.recognize_google(audio_data, language="en-IN")
        except sr.UnknownValueError:
            return "[Unintelligible]"
        except sr.RequestError as e:
            raise CustomException(f"API request error: {e}", "audio_error")
        finally:
            os.remove(temp_segment)

    def transcribe(self):
        """Full transcription process including downloading, processing, and transcribing the audio file.

        Returns:
            str: Final transcribed text from the audio file.

        Raises:
            CustomException: If no intelligible speech is detected or if multiple words are found in a single segment.
        """
        transcript = []
        final_transcript = ""

        try:
            # Download and prepare audio
            self.download_audio()
            self.convert_to_wav()

            # Load the WAV file and remove silence
            sound = AudioSegment.from_wav(self.temp_wav_file)
            non_silent_chunks = self.remove_silence(sound)

            # Transcribe each non-silent chunk
            for i, chunk in enumerate(non_silent_chunks):
                try:
                    transcribed_text = self.transcribe_audio_segment(chunk, i+1)
                    if transcribed_text != "[Unintelligible]":
                        transcript.append(transcribed_text)
                except CustomException as e:
                    raise e

            # Join all transcribed text into a single string
            final_transcript = " ".join(transcript).strip()
            if not final_transcript:
                raise CustomException("No intelligible speech detected.", "audio_error")

            # Check if the final transcript contains more than one word
            if len(final_transcript.split()) > 1:
                raise CustomException("The audio contains multiple words.", "invalid_input")

        except CustomException as e:
            raise e

        finally:
            # Clean up temporary audio files
            if os.path.exists(self.temp_wav_file):
                os.remove(self.temp_wav_file)
            if os.path.exists(self.file_path):
                os.remove(self.file_path)

        return final_transcript
