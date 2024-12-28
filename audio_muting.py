import os
import re
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
import whisper
from pydub import AudioSegment
from better_profanity import profanity

def extract_audio(video_path):
    """
    Extract audio from video file.

    :param video_path: Path to the input video file
    :return: Path to the extracted audio file
    """
    video = VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video.audio.write_audiofile(audio_path)
    video.close()
    return audio_path

def transcribe_audio_whisper(audio_path):
    """
    Transcribe audio using Whisper model.

    :param audio_path: Path to the audio file
    :return: Transcribed text from the audio
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def convert_audio_to_text_with_timestamps_whisper(audio_path, transcribed_text):
    """
    Use the Whisper-transcribed text and assign timestamps in 2-second intervals.

    :param audio_path: Path to the audio file
    :param transcribed_text: The text transcribed by the Whisper model
    :return: List of tuples containing (timestamp, transcribed_text)
    """
    audio = AudioSegment.from_wav(audio_path)
    chunk_length_ms = 2000
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    transcription = []
    words = transcribed_text.split()
    chunk_size = len(words) // len(chunks) if len(chunks) > 0 else len(words)

    for i, chunk in enumerate(chunks):
        start_time = i * 2
        chunk_text = " ".join(words[i * chunk_size:(i + 1) * chunk_size])
        transcription.append((start_time, chunk_text))

    return transcription

def timestamp_to_seconds(timestamp):
    minutes, seconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds

# Function to detect harmful words and return the timestamps
def find_harmful_word_timestamps(transcript_lines, timestamps):
    mute_times = []
    for i, line in enumerate(transcript_lines):
        if profanity.contains_profanity(line):
            mute_times.append(timestamps[i])
    return mute_times

# Function to mute sections of the video
def mute_video_at_timestamps(video_file, mute_times):
    video = VideoFileClip(video_file)
    clips = []
    last_end = 0

    for timestamp in mute_times:
        start_time = timestamp_to_seconds(timestamp)
        end_time = start_time + 1

        if start_time > last_end:
            clips.append(video.subclip(last_end, start_time))

        muted_clip = video.subclip(start_time, end_time).volumex(0)
        clips.append(muted_clip)
        last_end = end_time

    if last_end < video.duration:
        clips.append(video.subclip(last_end, video.duration))

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile("output_muted_video.mp4", codec="libx264", audio_codec="aac")

def process_video(video_file):
    """
    Main function to process the video by extracting audio, transcribing it using Whisper, detecting harmful words,
    and muting the sections with harmful words in the final output video.
    """
    audio_path = extract_audio(video_file)
    print("Audio extracted successfully.")

    transcribed_text = transcribe_audio_whisper(audio_path)
    print("Audio transcribed using Whisper successfully.")

    transcription = convert_audio_to_text_with_timestamps_whisper(audio_path, transcribed_text)
    os.remove(audio_path)

    timestamps = [f"{int(ts // 60):02d}:{int(ts % 60):02d}" for ts, _ in transcription]
    transcript_lines = [text for _, text in transcription]

    mute_times = find_harmful_word_timestamps(transcript_lines, timestamps)

    mute_video_at_timestamps(video_file, mute_times)
    print("Video processed and harmful words muted successfully.")

if _name_ == "_main_":
    video_file = "/content/New Project - Made with Clipchamp (3).mp4"
    process_video(video_file)