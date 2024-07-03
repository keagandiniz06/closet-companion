import whisper
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import io


class SpeechProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.whisper_model = whisper.load_model("large-v3")

    def is_speech_present(audio_file_path, silence_thresh=-50, min_silence_len=500):
        try:
            # Load the audio file
            audio = AudioSegment.from_file(audio_file_path)
            
            # Detect non-silent chunks in the audio file
            non_silent_chunks = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
            
            # Check if there are any non-silent chunks
            if len(non_silent_chunks) == 0:
                return False
            
            # Use speech recognition to confirm if there is speech in the non-silent chunks
            # recognizer = sr.Recognizer()
            for chunk in non_silent_chunks:
                start, end = chunk
                audio_chunk = audio[start:end]
                
                # Export audio chunk to memory buffer
                audio_chunk_buffer = io.BytesIO()
                audio_chunk.export(audio_chunk_buffer, format="wav")
                audio_chunk_buffer.seek(0)
                
                with sr.AudioFile(audio_chunk_buffer) as source:
                    audio_data = self.recognizer.record(source)
                    try:
                        self.recognizer.recognize_google(audio_data)
                        return True
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as e:
                        print(f"Could not request results from Google Speech Recognition service; {e}")
                        return False
            
            return False
        
        except Exception as e:
            print(f"Error in checking speech presence: {e}")
            return False

    def is_speech_quality_good(self, audio_file_path):
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)
                energy_threshold = 300  # Adjust this threshold as needed
                self.recognizer.energy_threshold = energy_threshold
                
                # Check if recognizer can recognize speech
                try:
                    self.recognizer.recognize_google(audio_data)
                    return True
                except sr.UnknownValueError:
                    return False
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")
                    return False
        except Exception as e:
            print(f"Error in checking speech quality: {e}")
            return False

    def convert_speech_to_text_whisper(self, audio_file_path):
        try:
            result = self.whisper_model.transcribe(audio_file_path)
            return result["text"]
        except Exception as e:
            print(f"Error in converting speech to text using Whisper: {e}")
            return ""
