import os
import queue
import random
import subprocess
import sys
import time
import pvporcupine
import vosk
import yaml
from fuzzywuzzy import fuzz
from rich import print
import tts
import config
from huggingface_hub import InferenceClient
import pyautogui
import pygame
from deep_translator import GoogleTranslator
import pyaudio


api_key = os.getenv("API-KEY")
megatron = os.getenv("MEGATRON")
client = InferenceClient(api_key=api_key)
# some consts
CDIR = os.getcwd()
VA_CMD_LIST = yaml.safe_load(open('commands.yaml', 'rt', encoding='utf8'))
# LLama vars
system_message = {"role": "system", "content": "You are Megatron voice assistant. Speak in english. Without coments like 'dramatic pause'."}
message_log = [system_message]
# PORCUPINE Megatron
Megatron = "/Users/mac/Downloads/Megatron_en_mac_v3_0_0/Megatron_en_mac_v3_0_0.ppn"
porcupine = pvporcupine.create(
    access_key=os.getenv("PICO"),
    keyword_paths=[megatron],
    sensitivities=[1]
)
# VOSK
model = vosk.Model("vosk-model-uk-v3-lgraph")
#model = vosk.Model("vosk-model-ru-0.22")

samplerate = 16000
device = config.MICROPHONE_INDEX
kaldi_rec = vosk.KaldiRecognizer(model, samplerate)
q = queue.Queue()




def translate_text(text, dest_language='uk'):
    try:
        translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return None

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

def open_audio_stream():
    stream = pa.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=samplerate,
                        input=True,
                        frames_per_buffer=512)
    return stream

pa = pyaudio.PyAudio()
stream = open_audio_stream()

pygame.mixer.init()

def play(phrase, wait_done=True):
    global recorder
    filename = f"{CDIR}/sound/"
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
    elif phrase == "who":
        filename += "megatron-yard-leader-101soundboards.mp3"

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

'''
def lab_discr(app_path, file_path):
    subprocess.Popen(["open", "-a", app_path, file_path])
    time.sleep(4)
    pyautogui.click(863, 526)
    time.sleep(3)
    pyautogui.hotkey('command', 'a')
    time.sleep(0.2)
    pyautogui.hotkey('command', 'c')
    time.sleep(0.2)
    pyautogui.hotkey('ctrl', 'right')
    time.sleep(4)
    button_x, button_y = pyautogui.position()
    print(button_x, button_y)
    pyautogui.click(button_x, button_y)
    time.sleep(0.1)
    pyautogui.hotkey('command', 'v')
    voice = поясни код детально дуже простою мовою без використання його вставок: 
    print(voice)
    time.sleep(4)
    button_x, button_y = pyautogui.position()
    print(button_x, button_y)
    pyautogui.click(button_x, button_y)
    time.sleep(4)
    button_x, button_y = pyautogui.position()
    print(button_x, button_y)
    pyautogui.click(button_x, button_y)
    time.sleep(4)
    button_x, button_y = pyautogui.position()
    print(button_x, button_y)
    pyautogui.click(button_x, button_y)
    pyautogui.click(button_x, button_y)
    pyautogui.click(button_x, button_y)
    pyautogui.press('delete')
    message_log.append({"role": "user", "content": voice})
    response = llama_answer()
    ukr_response = translate_text(response)
    message_log.append({"role": "assistant", "content": ukr_response})
    print(f"Text to speak: {ukr_response}")
    stream.stop_stream()
    tts.va_speak(ukr_response)
    time.sleep(0.5)
    stream.start_stream()
lab_discr("/Applications/Rider.app", "/Users/mac/Desktop/Unik/Labs/discr_2lab3/discr_2lab3/Program.cs" )
'''
def start_ps_remote_play():
    remote_play_path = "/Applications/RemotePlay.app"
    try:
        subprocess.Popen(["open", "-a", remote_play_path])
        print("PlayStation Remote Play запущено...")
        time.sleep(2)  # Дайте час додатку для запуску
        # Переключіть фокус на вікно Remote Play
        pyautogui.hotkey('command', 'tab')  # Перейти до наступного вікна
        time.sleep(3)  # Зачекайте, щоб переконатися, що вікно активне
        button_x, button_y = 644, 555  # Встановіть правильні координати кнопки
        pyautogui.click(button_x, button_y)  # Натисніть на кнопку
        print("Натиснуто кнопку для запуску...")
        time.sleep(15)
        print(pyautogui.position())
    except FileNotFoundError:
        print("PS Remote Play не знайдено за вказаним шляхом.")

def close_ps_remote_play():
    try:
        pyautogui.hotkey('ctrl', 'left')
        pyautogui.hotkey('ctrl', 'left')
        pyautogui.hotkey('ctrl', 'left')
        pyautogui.hotkey('ctrl', 'left')
        pyautogui.hotkey('ctrl', 'left')
        time.sleep(3)  # Зачекайте, щоб переконатися, що вікно активне
        button_x, button_y = 219, 191  # Встановіть правильні координати кнопки
        pyautogui.click(button_x, button_y)  # Натисніть на кнопку
        print("Натиснуто кнопку для вимкнення...")
        time.sleep(2)
        button_x, button_y = 593, 317  # Встановіть правильні координати кнопки
        pyautogui.click(button_x, button_y)  # Натисніть на кнопку
        print("Натиснуто кнопку для підтвердження вимкнення...")
    except FileNotFoundError:
        print("PS Remote Play не знайдено за вказаним шляхом.")

def get_audio_devices():
    """Get a list of audio output devices."""
    output = subprocess.check_output(["SwitchAudioSource", "-a"]).decode("utf-8")
    devices = output.splitlines()
    return devices

def switch_audio_output(target_device):
    """Switch audio output to the specified device."""
    subprocess.run(["SwitchAudioSource", "-s", target_device])

def switch_to_headphones(device_name):
    """Switch audio output to headphones."""
    command = f'SwitchAudioSource -s "{device_name}"'
    os.system(command)
    print(f'Аудіо перемкнуто на: {device_name}')


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
        switch_to_headphones("YARIN🖤")
        play("ok")
    elif cmd == 'gaming_mode_on':
        start_ps_remote_play()
        play("ok")
    elif cmd == 'gaming_mode_off':
        close_ps_remote_play()
        play("ok")
    elif cmd == 'off':
        play("off", True)
        shutdown_system()
    elif cmd == 'who':
        play("who")
    elif cmd == 'youtube_play':
        pyautogui.press('space')

