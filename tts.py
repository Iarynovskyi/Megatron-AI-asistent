import time

import sounddevice as sd
import torch
#from pygments.lexers.sql import language_re

language = 'ua'
model_id = 'v4_ua'
#language = 'ru'
#model_id = 'ru_v3'
sample_rate = 48000  # 48000
speaker = 'mykyta'  # aidar, baya, kseniya, xenia, random
put_accent = True
put_yo = True
device = torch.device('cpu')  # cpu или gpu
#text = "Хауди Хо, друзья!!!"

model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=language,
                          speaker=model_id)
model.to(device)

def split_text(text, max_length=800):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    # Додати останній фрагмент, якщо він не порожній
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
def va_speak(what: str):
    if not what:
        print("Помилка: Немає тексту для озвучення.")
        return

    text_chunks = split_text(what)

    for i, chunk in enumerate(text_chunks):
        try:
            print(f"Обробка фрагмента {i+1}/{len(text_chunks)}: {chunk[:100]}...")  # Логування
            audio = model.apply_tts(text=chunk + "..",
                                    speaker=speaker,
                                    sample_rate=sample_rate,
                                    put_accent=put_accent,
                                    put_yo=put_yo)

            # Відтворити аудіо
            sd.play(audio, sample_rate * 1.05)
            time.sleep((len(audio) / sample_rate) + 0.5)
            sd.stop()

        except Exception as e:
            print(f"Виникла помилка: {e} при обробці фрагмента {i+1}/{len(text_chunks)}")
            continue

# sd.play(audio, sample_rate)
# time.sleep(len(audio) / sample_rate)
# sd.stop()