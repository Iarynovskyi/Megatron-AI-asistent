
import json
import struct
from commands import play, stream, kaldi_rec, va_respond, porcupine, pa, open_audio_stream
import time

import tkinter as tk
#from tkinter import messagebox
import threading
import pyaudio


import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle
# Ваш код функції Megatron_on
def Megatron_on():

    pa = pyaudio.PyAudio()
    stream = open_audio_stream()
    print(f"Jarvis (v3.0) started...")
    play("run")
    print("run")
    time.sleep(0.5)
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
                while True:
                    data = stream.read(512, exception_on_overflow=False)
                    if kaldi_rec.AcceptWaveform(data):
                        recognized_text = json.loads(kaldi_rec.Result())["text"]
                        print("Recognized text:", recognized_text)
                        if va_respond(recognized_text):
                            print("12345")
                            break
                    else:
                        partial_result = kaldi_rec.PartialResult()
                        print("Partial result:", json.loads(partial_result)["partial"])
                    time.sleep(0.01)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()

'''
# Функція для запуску Megatron_on в окремому потоці
def start_megatron():
    threading.Thread(target=Megatron_on, daemon=True).start()

# Створюємо головне вікно
root = tk.Tk()
root.title("Megatron AI assistant")
root.geometry("400x300")

# Додаємо віджет (кнопку)
button = tk.Button(root, text="Натисни мене", command=start_megatron)
button.pack(pady=20)

# Запускаємо головний цикл програми
root.mainloop()
'''
class CustomWindow(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(0.5, 0.7, 0.8, 1)  # Встановлюємо колір фону
            self.rect = Rectangle(size=(500, 300), pos=(50, 50))  # Розмір і позиція вікна


class MyApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.button = Button(text='Запустити віртуального асистента')
        self.button.bind(on_press=self.start_assistant)
        layout.add_widget(self.button)
        return layout

    def start_assistant(self, instance):
        # Знеактивуємо кнопку, поки асистент працює
        self.button.disabled = True
        print("Асистент запущений. Кнопка деактивована.")

        # Запускаємо модель у новому потоці
        assistant_thread = threading.Thread(target=self.run_assistant, daemon=True)
        assistant_thread.start()

    def run_assistant(self):
        Megatron_on()  # Запускаємо вашу модель

        # Коли асистент закінчує роботу, активуємо кнопку знову
        self.button.disabled = False
        print("Асистент закінчив роботу. Кнопка активована.")


MyApp().run()