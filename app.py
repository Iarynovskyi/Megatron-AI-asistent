from http.client import responses

from torch.utils.tensorboard.summary import audio
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import os
import time
import pandas as pd
import re
import numpy as np
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline
import evaluate
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from huggingface_hub import login
#from kaggle_secrets import UserSecretsClient

#secret_label = "HF Hub"
#secret_value = UserSecretsClient().get_secret(secret_label)
#login(token=secret_value)


import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

import speech_recognition as sr
import sounddevice as sd
import torch
import pyttsx3
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Підключення пристрою (GPU/CPU)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Завантаження моделі та процесора Whisper
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Створення pipeline для автоматичного розпізнавання мовлення
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)
def record_audio(duration=5, fs=16000):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()  # Очікуємо завершення запису
    return audio


def convert_to_mono(audio_data):
    if len(audio_data.shape) > 1:  # Якщо є більше одного каналу
        audio_data = np.mean(audio_data, axis=1)  # Усереднюємо по всіх каналах
    return audio_data
def Record_Text():
    while True:
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                print("Recording...")
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                return MyText
        except sr.RequestError as e:
            print("Could not request audio")
        except sr.UnknownValueError:
            print("UnknownValueError")

def listen_for_keyword(keyword):
    print("Слухаю... (Скажіть 'стоп', щоб почати запис)")
    while True:
        # Запис короткого аудіо для виявлення ключового слова
        audio_data = record_audio(duration=3)  # Записуємо 3 секунди
        audio_data = convert_to_mono(audio_data)

        # Перетворення аудіо в текст
        result = pipe(audio_data, return_timestamps=False)
        recognized_text = result['text']

        print(f"Виявлено: {recognized_text}")

        # Перевірка на ключове слово
        if keyword in recognized_text.lower():
            print("Ключове слово виявлено! Починаю запис...")
            break

    # Тепер записуємо основне аудіо після виявлення ключового слова
    main_audio = record_audio(duration=10)  # Можна змінити тривалість запису
    main_audio = convert_to_mono(main_audio)

    # Перетворення основного аудіо в текст
    result = pipe(main_audio, return_timestamps=True)
    transcribed_text = result['text']

    # Виведення результату
    print("Текст запису:", transcribed_text)
    return main_audio

audio_file = "/Users/mac/Downloads/Your_First_Lesson.mp3"

from huggingface_hub import InferenceClient
from dotenv import load_dotenv
api_key = os.getenv("API-KEY")
client = InferenceClient(api_key=api_key)

def Massage_to_chat(massage):
    response = ""
    for message in client.chat_completion(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": massage}],
        max_tokens=500,
        stream=False,
    ):
        print(message.choices[0].delta.content, end="")
    print("\n-" * 3)

def Massage_to_chat_q(massage):
    for message in client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=500,
            stream=True,
    ):
        print(message.choices[0].delta.content, end="")


def Speak_Text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 100)
    engine.say(text)
    engine.runAndWait()
r = sr.Recognizer()


#pipe = pipeline("text-to-speech", model="myshell-ai/MeloTTS-English")
'''
while(True):
    audio_data = listen_for_keyword("мегатрон")
    result = pipe(audio_data, return_timestamps=True)
    transcribed_text = result['text']
    base_text = "Your name is Megatron so you must talk like him. You can talk in english. You must tell me a not very big texts. Be kind, speak without insertions, like '(In a deep, robotic voice)'. My question is: "
    print(transcribed_text)
    all_text = base_text + transcribed_text

    print("if you wont to stop print s: ")
    input_text = input("Print massage: ").lower()
    if input_text == "s":
        break
    Massage_to_chat(all_text)
'''

#--------------------------------------------------------------------------------------







import datetime
import json
import os
import queue
import random
import struct
import subprocess
import sys
import time
from ctypes import POINTER, cast
import torch
import requests
import pvporcupine
import simpleaudio as sa
import vosk
from vosk import Model, KaldiRecognizer
import yaml
from fuzzywuzzy import fuzz
from pvrecorder import PvRecorder
from rich import print
import tts
import wave
import config
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

api_key = os.getenv("API-KEY")
client = InferenceClient(api_key=api_key)

# some consts
CDIR = os.getcwd()
VA_CMD_LIST = yaml.safe_load(open('commands.yaml', 'rt', encoding='utf8'))

# ChatGPT vars
system_message = {"role": "system", "content": "You are Megatron. Speak without insertions, like '(In a deep, robotic voice)'."}
message_log = [system_message]

Megatron = "/Users/mac/Downloads/Megatron_en_mac_v3_0_0/Megatron_en_mac_v3_0_0.ppn"
# PORCUPINE
porcupine = pvporcupine.create(
    access_key=os.getenv("PICOVOICE_TOKEN"),
    keywords=["jarvis"],
    sensitivities=[1]
)

from deep_translator import GoogleTranslator

def translate_text(text, dest_language='uk'):
    try:
        translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return None

# VOSK
model = vosk.Model("vosk-model-uk-v3-lgraph")
#model = vosk.Model("vosk-model-ru-0.22")

samplerate = 16000
device = config.MICROPHONE_INDEX
kaldi_rec = vosk.KaldiRecognizer(model, samplerate)
q = queue.Queue()


def llama_answer():
    global message_log

    model_engine = "meta-llama/Meta-Llama-3-8B-Instruct"
    max_tokens = 500  # стандартно 1024
    accumulated_response = ""  # Змінна для накопичення відповіді

    try:
        # Виклик для потокового отримання відповідей
        for response in client.chat_completion(
                model=model_engine,
                messages=message_log,
                max_tokens=max_tokens,
                stream=True
        ):
            # Додаємо частину відповіді до накопичувальної змінної
            accumulated_response += response.choices[0].delta.content

    except Exception as ex:
        return "Сталася помилка при зверненні до Llama."

    return accumulated_response


import pyaudio
pa = pyaudio.PyAudio()
def open_audio_stream():
    stream = pa.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=samplerate,
                        input=True,
                        frames_per_buffer=512)
    return stream
stream = open_audio_stream()

# Play audio response (macOS compatible)
import pygame
pygame.mixer.init()


def play(phrase, wait_done=True):
    global recorder
    filename = f"{CDIR}/sound/"

    # Вибір звуку на основі фрази
    if phrase == "greet":
        filename += f"sound_greet{random.choice([1, 2, 3])}.wav"
    elif phrase == "ok":
        filename += f"sound_ok{random.choice([1, 2, 3])}.wav"
    elif phrase == "not_found":
        filename += "sound_not_found.wav"
    elif phrase == "thanks":
        filename += "sound_thanks.wav"
    elif phrase == "run":
        filename += "sound_run.wav"
    elif phrase == "stupid":
        filename += "sound_stupid.wav"
    elif phrase == "ready":
        filename += "sound_ready.wav"
    elif phrase == "off":
        filename += "sound_off.wav"

    # Перевірка, чи існує файл
    if not os.path.isfile(filename):
        print(f"File does not exist: {filename}")
        return

    print(f"Playing sound from: {filename}")

    # Відтворення звуку
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    if wait_done:
        # Чекати, поки звук грає
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(5)  # Додаємо затримку для зменшення навантаження на ЦП
    print("Sound playback finished.")

def q_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def va_respond(voice: str):
    global recorder, message_log
    print(f"Recognized: {voice}")
    cmd = recognize_cmd(filter_cmd(voice))
    print(cmd)
    if len(cmd['cmd'].strip()) <= 0:
        return False
    elif cmd['percent'] < 60 or cmd['cmd'] not in VA_CMD_LIST.keys():
        if fuzz.ratio(voice.join(voice.split()[:1]).strip(), "скажи") > 75:
            play("stupid")
            message_log.append({"role": "user", "content": voice})
            response = llama_answer()
            ukr_response = translate_text(response)
            message_log.append({"role": "assistant", "content": ukr_response})
            print(f"Text to speak: {ukr_response}")
            stream.stop_stream()
            tts.va_speak(ukr_response)
            time.sleep(0.5)
            stream.start_stream()
            return False
        else:
            play("not_found")
            time.sleep(1)

        return False
    else:
        execute_cmd(cmd['cmd'], voice)
        return True


from wakeonlan import send_magic_packet

# Введіть MAC-адресу вашої PS5

MAC_ADDRESS = os.getenv("MAC_ADDRESS")
def turn_on_playstation():
    try:
        send_magic_packet(MAC_ADDRESS)
        print("PlayStation вмикається...")
    except Exception as e:
        print(f"Виникла помилка: {e}")
turn_on_playstation()

import socket
import struct


def wake_on_lan(mac_address):
    # Форматування MAC-адреси
    mac_address = mac_address.replace(":", "").lower()
    if len(mac_address) != 12:
        raise ValueError("Неправильна MAC-адреса")

    # Створення WOL пакета
    magic_packet = bytes.fromhex('FF' * 6) + (bytes.fromhex(mac_address) * 16)

    # Надсилання пакета на широкомовну адресу
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Увімкнення широкомовлення
        sock.sendto(magic_packet, ("255.255.255.255", 9))  # Надсилання на широкомовну адресу та порт 9


# Виклик функції
wake_on_lan(MAC_ADDRESS)  # Ваша MAC-адреса


def get_audio_devices():
    """Get a list of audio output devices."""
    output = subprocess.check_output(["SwitchAudioSource", "-a"]).decode("utf-8")
    devices = output.splitlines()
    return devices

def switch_audio_output(target_device):
    """Switch audio output to the specified device."""
    subprocess.run(["SwitchAudioSource", "-s", target_device])

def switch_to_headphones():
    """Switch audio output to headphones."""
    devices = get_audio_devices()
    headphones = [device for device in devices if "headphones" in device.lower()]

    if headphones:
        print(f"Switching to headphones: {headphones[0]}")
        switch_audio_output(headphones[0])
    else:
        print("No headphones found.")

def switch_to_speakers():
    """Switch audio output to speakers."""
    devices = get_audio_devices()
    speakers = [device for device in devices if "speakers" in device.lower()]

    if speakers:
        print(f"Switching to speakers: {speakers[0]}")
        switch_audio_output(speakers[0])
    else:
        print("No speakers found.")

def mute_sound():
    """Mute the system sound."""
    os.system("osascript -e 'set volume output muted true'")

def unmute_sound():
    """Unmute the system sound."""
    os.system("osascript -e 'set volume output muted false'")

def shutdown_system():
    try:
        # Очищення ресурсу porcupine
        if porcupine is not None:
            porcupine.delete()

        # Відключення аудіо-системи, якщо активна
        if sd.get_stream() is not None:
            sd.stop()
            sd.close()

        # Виведення повідомлення про вимкнення
        print("Система вимкнена коректно.")

        # Коректне завершення програми
        sys.exit(0)

    except Exception as e:
        print(f"Помилка при вимкненні системи: {e}")
        sys.exit(1)

def filter_cmd(raw_voice: str):
    cmd = raw_voice

    for x in config.VA_ALIAS:
        cmd = cmd.replace(x, "").strip()

    for x in config.VA_TBR:
        cmd = cmd.replace(x, "").strip()

    return cmd


def recognize_cmd(cmd: str):
    rc = {'cmd': '', 'percent': 0}
    for c, v in VA_CMD_LIST.items():
        for x in v:
            vrt = fuzz.ratio(cmd, x)
            if vrt > rc['percent']:
                rc['cmd'] = c
                rc['percent'] = vrt

    return rc


# Use macOS specific commands in place of Windows-specific ones
def execute_cmd(cmd: str, voice: str):
    if cmd == 'open_browser':
        subprocess.Popen(["open", "-a", "Safari"])
        play("ok")
    elif cmd == 'open_youtube':
        subprocess.Popen(["open", "https://www.youtube.com"])
        play("ok")
    elif cmd == 'open_google':
        subprocess.Popen(["open", "https://www.google.com"])
        play("ok")
    elif cmd == 'thanks':
        play("thanks")
    elif cmd == 'stupid':
        play("stupid")
    elif cmd == 'music':
        subprocess.Popen(["open", "-a", "Spotify"])
        print("Spotify відкрито!")
    elif cmd == 'sound_off':
        mute_sound()
        play("ok")
    elif cmd == 'sound_on':
        unmute_sound()
        play("ok")
    elif cmd == 'switch_to_dynamics':
        switch_to_speakers()
        play("ok")
    elif cmd == 'switch_to_headphones':
        switch_to_headphones()
        play("ok")
    elif cmd == 'gaming_mode_on':

        play("ok")
    elif cmd == 'off':
        play("off", True)
        shutdown_system()



#print(f'Using device: {stream.selected_device}')
print(f"Jarvis (v3.0) started...")
play("run")
print("run")
time.sleep(0.5)
ltc = time.time()-1000
try:
    while True:
        pcm = stream.read(512, exception_on_overflow=False)  # Зчитуємо 512 байтів
        pcm = struct.unpack_from("h" * (len(pcm) // 2), pcm)  # Визначення кількості семплів
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            print(f"Detected keyword: {keyword_index}")
            stream.stop_stream()
            print("Ключове слово виявлено!")
            play("greet", True)
            print("Так, сер.")
            stream.start_stream()
            ltc = time.time()
            k = True
        while time.time() - ltc <= 10:
            data = stream.read(512, exception_on_overflow=False)
            if kaldi_rec.AcceptWaveform(data):
                recognized_text = json.loads(kaldi_rec.Result())["text"]
                print("Recognized text:", recognized_text)
                if va_respond(recognized_text):
                    print("12345")
                    ltc = time.time()
                    break
            else:
                partial_result = kaldi_rec.PartialResult()
                print("Partial result:", json.loads(partial_result)["partial"])
            time.sleep(0.01)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    if stream is not None and stream.is_active():
        stream.stop_stream()
    stream.close()
    pa.terminate()
    porcupine.delete()