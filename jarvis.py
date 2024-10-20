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
system_message = {"role": "system", "content": "Ты голосовой ассистент из железного человека."}
message_log = [system_message]

# PORCUPINE
porcupine = pvporcupine.create(
    access_key=os.getenv("PICOVOICE_TOKEN"),
    keywords=['jarvis'],
    sensitivities=[1]
)

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
    max_tokens = 256  # стандартно 1024
    try:
        response = client.chat_completion(
            model=model_engine,
            messages=message_log,
            max_tokens=max_tokens,
            stream=False  # Disable streaming to get a complete answer
        )
    except Exception as ex:
        return "Сталася помилка при зверненні до Llama."

    # Return the text of the answer
    return response.choices[0].message.content


# Play audio response (macOS compatible)

def play(phrase, wait_done=True):
    global recorder
    filename = f"{CDIR}/sound/"

    if phrase == "greet":  # for py 3.8
        filename += f"sound_greet{random.choice([1, 2, 3])}.wav"
    elif phrase == "ok":
        filename += f"sound_ok{random.choice([1, 2, 3])}.wav"
    elif phrase == "found":
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

    if wait_done:
        print("1")
        stream.stop_stream()
        print("2")
        #stream.close()
        print("3")
        #recorder.stop()
    print("4")
    wave_obj = sa.WaveObject.from_wave_file(filename)
    print("5")
    play_obj = wave_obj.play()
    print("6")


    if wait_done:
        print("7")
        #play_obj.wait_done()
        print("8")
        #recorder.start()
        stream.start_stream()
        print("9")


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
    elif cmd['percent'] < 70 or cmd['cmd'] not in VA_CMD_LIST.keys():
        if fuzz.ratio(voice.join(voice.split()[:1]).strip(), "скажи") > 75:

            message_log.append({"role": "user", "content": voice})
            response = llama_answer()
            message_log.append({"role": "assistant", "content": response})

            recorder.stop()
            tts.va_speak(response)
            time.sleep(0.5)
            recorder.start()
            return False
        else:
            play("not_found")
            time.sleep(1)

        return False
    else:
        execute_cmd(cmd['cmd'], voice)
        return True


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
    elif cmd == 'off':
        play("off", True)
        porcupine.delete()
        exit(0)



rec = KaldiRecognizer(model, 16000)

with wave.open('test_audio.wav', 'rb') as wf:
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            print(json.loads(result)["text"])
        else:
            partial_result = rec.PartialResult()
            print(json.loads(partial_result)["partial"])
with wave.open('test_audio.wav', 'rb') as wf:
    n_frames = wf.getnframes()
    frames_per_read = 1600  # Наприклад, 0.1 секунди при 16000 Гц

    # Читання частинами
    while True:
        data = wf.readframes(frames_per_read)
        if len(data) == 0:
            break  # Вихід з циклу, якщо дані закінчилися

        if kaldi_rec.AcceptWaveform(data):
            print("Recognized text:", json.loads(kaldi_rec.Result())["text"])
        else:
            partial_result = kaldi_rec.PartialResult()
            print("Partial result:", json.loads(partial_result)["partial"])

    # Після завершення читання даних, отримайте останній результат
    final_result = kaldi_rec.FinalResult()
    print("Final result:", json.loads(final_result)["text"])

import pyaudio
pa = pyaudio.PyAudio()
stream = pa.open(rate=samplerate,
                  channels=1,
                  format=pyaudio.paInt16,
                  input=True,
                  frames_per_buffer=512)

'''
print("Starting audio stream...")

try:
    while True:
        data = stream.read(1600)  # Читання 1600 кадрів
        if kaldi_rec.AcceptWaveform(data):
            # Якщо кадр прийнято, виводимо текст
            recognized_text = json.loads(kaldi_rec.Result())["text"]
            print("Recognized text:", recognized_text)
        else:
            # Отримуємо частковий результат
            partial_result = kaldi_rec.PartialResult()
            print("Partial result:", json.loads(partial_result)["partial"])

except KeyboardInterrupt:
    print("Stopping...")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()

'''
k=False
ltc = time.time() - 1000
try:
    while True:

        pcm = stream.read(512, exception_on_overflow=False)  # Зчитуємо 512 байтів
        pcm = struct.unpack_from("h" * (len(pcm) // 2), pcm)  # Визначення кількості семплів
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            print(f"Detected keyword: {keyword_index}")
            stream.stop_stream()  # Зупиняємо потік
            print("Ключове слово виявлено!")
            play("greet", True)  # Відтворюємо відповідь
            print("Так, сер.")
            print("10")
            stream.start_stream()  # Запускаємо потік
            print("11")
            ltc = time.time()  # Оновлюємо таймер
            print("12")
        print("xui")
        '''

        while time.time() - ltc >= 10:
            pcm = stream.read(512, exception_on_overflow=False)
            #sp = struct.unpack("h" * (len(pcm) // 2), pcm)  # Створюємо пакет
            #byte_data = struct.pack("h" * len(sp), *sp)
            byte_data = struct.pack("h" * len(pcm), *pcm)
            #data = stream.read(1600)
            if kaldi_rec.AcceptWaveform(byte_data ):
                if va_respond(json.loads(kaldi_rec.Result())["text"]):
                    ltc = time.time()
                break
            if kaldi_rec.AcceptWaveform(byte_data):
                response_text = json.loads(kaldi_rec.Result())["text"]
                print("Recognized text:", response_text)
            else:
                partial_result = kaldi_rec.PartialResult()
                print(f"Partial result: {json.loads(partial_result)['partial']}")
            time.sleep(0.1)  # Затримка 100 мс
        '''


except KeyboardInterrupt:
    print("Stopping...")
finally:
    if stream is not None and stream.is_active():
        stream.stop_stream()
    stream.close()
    pa.terminate()
    porcupine.delete()


# Set up the microphone and start listening
recorder = PvRecorder(device_index=config.MICROPHONE_INDEX, frame_length=porcupine.frame_length)
recorder.start()
print(f'Using device: {recorder.selected_device}')

print(f"Jarvis (v3.0) started...")
play("run")
print("run")
time.sleep(0.5)
print("sleeping")

ltc = time.time() - 1000
print("time.time")
'''
try:
    while True:
        try:
            pcm = recorder.read(512)
            #keyword_index = porcupine.process(pcm)
            pcm = struct.unpack_from("h" * 512, pcm)

            keyword_index = porcupine.process(pcm)

            # Якщо виявлено ключове слово
            if keyword_index >= 0:
                recorder.stop()  # Зупиняємо запис для розпізнавання
                print("Ключове слово виявлено!")

                # Відповідь на ключове слово
                play("greet", True)  # Ваша функція відтворення звуку
                print("Так, сер.")

                recorder.start()  # Запускаємо запис знову
                ltc = time.time()  # Оновлюємо таймер

                # Цикл для розпізнавання мови
                while time.time() - ltc <= 10:  # Продовжуємо розпізнавати протягом 10 секунд
                    pcm = recorder.read()
                    sp = struct.pack("h" * len(pcm), *pcm)

                    if kaldi_rec.AcceptWaveform(sp):
                        # Отримуємо результат розпізнавання
                        result = json.loads(kaldi_rec.Result())
                        text = result.get("text", "")
                        print("Розпізнано:", text)

                        # Якщо потрібно, реагуємо на розпізнану фразу
                        if va_respond(text):  # Ваша функція для відповіді на команду
                            ltc = time.time()  # Оновлюємо таймер, якщо була команда

                # Відновлюємо стан запису
                recorder.start()  # Запускаємо запис знову

        except Exception as err:
            print(f"Несподівана помилка: {err}")

except KeyboardInterrupt:
    print("Запис зупинено.")
finally:
    # Завершення роботи
    recorder.stop()
    recorder.close()
    porcupine.delete()


def play(phrase, wait_done=True):
    global recorder
    filename = f"{CDIR}/sound/"

    if phrase == "greet":
        filename += f"sound_greet{random.choice([1, 2, 3])}.wav"
    elif phrase == "ok":
        filename += f"sound_ok{random.choice([1, 2, 3])}.wav"
    elif phrase == "found":
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

    if wait_done:
        recorder.stop()

    try:
        wave_obj = sa.WaveObject.from_wave_file(filename)
        play_obj = wave_obj.play()
    except Exception as e:
        print(f"Error playing sound '{filename}': {e}")


# Основний цикл програми
    recorder = PvRecorder(device_index=config.MICROPHONE_INDEX, frame_length=porcupine.frame_length)
    recorder.start()
    print(f'Using device: {recorder.selected_device}')

    print(f"Jarvis (v3.0) started...")
    play("run")
    time.sleep(0.5)

    ltc = time.time() - 1000
    max_retries = 5  # Максимальна кількість спроб
    retry_count = 0

    while retry_count < max_retries:
        try:
            pcm = recorder.read()
            # Ваш код обробки тут
            retry_count = 0  # Скинути лічильник після успішного читання
        except Exception as e:
            print("Error reading from recorder:", e)
            retry_count += 1
            time.sleep(1)
'''
