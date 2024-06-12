import numpy as np

from typing import Dict, List, Union, Tuple

import whisperx

import os

import pydub
from pydub import AudioSegment


def pydub_to_np(audio: pydub.AudioSegment) -> Tuple[np.ndarray, int]:
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    Returns tuple (audio_np_array, sample_rate).
    """

    return (
        np.array(audio.get_array_of_samples(), dtype=np.float32).reshape(
            (-1, audio.channels)
        )
        / (1 << (8 * audio.sample_width - 1)),
        audio.frame_rate,
    )

def transcribe_audio(
    audio_path: str,
    device: str = "cpu",
    batch_size: int = 2,
    compute_type: str = "float32",
    language: str = "en",
    model_name_whisper: str = "large-v2",
) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
    """
    Transcribe audio using whisperx,
    and return the text (transcription) and the words with their start and end times.
    """

    ## Load whisperx model
    model_whisperx = whisperx.load_model(
        model_name_whisper,
        device,
        compute_type=compute_type,
        language=language,
    )

    ## Transcribe audio
    audio = whisperx.load_audio(audio_path)
    result = model_whisperx.transcribe(audio, batch_size=batch_size)
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )

    ## Align timestamps
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if result is None or "segments" not in result or len(result["segments"]) == 0:
        return "", []

    if len(result["segments"]) == 1:
        text = result["segments"][0]["text"]
        words = result["segments"][0]["words"]
    else:
        text = " ".join(
            result["segments"][i]["text"] for i in range(len(result["segments"]))
        )
        words = [word for segment in result["segments"] for word in segment["words"]]

    # Remove words that are not properly transcribed
    words = [word for word in words if "start" in word]
    return text, words

def print_log(*args):
    # This is just a wrapper to easily spot the print :) - I use it to debug
    print(args)

def remove_word(audio, word, removal_type: str = "nothing"):
    """
    Remove a word from audio using pydub, by replacing it with:
    - nothing
    - silence
    - white noise
    - pink noise

    Args:
        audio (pydub.AudioSegment): audio
        word: word to remove with its start and end times
        removal_type (str, optional): type of removal. Defaults to "nothing".
    """

    a, b = 100, 40

    before_word_audio = audio[: word["start"] * 1000 - a]
    after_word_audio = audio[word["end"] * 1000 + b :]
    word_duration = (word["end"] * 1000 - word["start"] * 1000) + a + b

    if removal_type == "nothing":
        replace_word_audio = AudioSegment.empty()
    elif removal_type == "silence":
        replace_word_audio = AudioSegment.silent(duration=word_duration)

    elif removal_type == "white noise":
        sound_path = ("./explanation_sounds/white_noise.mp3",)
        replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]

        # display(audio_removed)
    elif removal_type == "pink noise":
        sound_path = ("./explanation_sounds/pink_noise.mp3",)
        replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]

    audio_removed = before_word_audio + replace_word_audio + after_word_audio
    return audio_removed


#PATH_DIR = "../fluent_speech_commands_dataset/wavs/speakers/2ojo7YRL7Gck83Z3"
#AUDIO_PATH = "../fluent_speech_commands_dataset/wavs/speakers/2ojo7YRL7Gck83Z3/0a36b8a0-45e0-11e9-81ce-69b74fd7e64e.wav"
AUDIO_PATH = "./0a36b8a0-45e0-11e9-81ce-69b74fd7e64e.wav"

#text, words_transcript =  transcribe_audio(
#    audio_path=AUDIO_PATH,
#    language="en"
#)

#print(text)
#print()
#print(words_transcript)




def remove_specified_words(audio, words, removal_type: str = "nothing"):
    """
    Remove a word from audio using pydub, by replacing it with:
    - nothing
    - silence
    - white noise
    - pink noise

    Args:
        audio (pydub.AudioSegment): audio
        word: word to remove with its start and end times
        removal_type (str, optional): type of removal. Defaults to "nothing".
    """

    from copy import deepcopy

    audio_removed = deepcopy(audio)

    a, b = 100, 40

    from IPython.display import display

    for word in words:
        start = int(word["start"] * 1000)
        end = int(word["end"] * 1000)

        before_word_audio = audio_removed[: start - a]
        after_word_audio = audio_removed[end + b :]

        word_duration = (end - start) + a + b

        if removal_type == "nothing":
            replace_word_audio = AudioSegment.empty()
        elif removal_type == "silence":
            replace_word_audio = AudioSegment.silent(duration=word_duration)
        elif removal_type == "white noise":
            sound_path = ("./explanation_sounds/white_noise.mp3"),
            replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]
        elif removal_type == "pink noise":
            sound_path = ("./explanation_sounds/pink_noise.mp3"),
            replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]

        audio_removed = before_word_audio + replace_word_audio + after_word_audio
    return audio_removed