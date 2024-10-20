import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Налаштування авторизації
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='YOUR_CLIENT_ID',
                                               client_secret='YOUR_CLIENT_SECRET',
                                               redirect_uri='YOUR_REDIRECT_URI',
                                               scope='user-modify-playback-state user-read-playback-state'))

def play_song(song_uri):
    try:
        # Отримати доступні пристрої
        devices = sp.devices()
        if devices['devices']:
            device_id = devices['devices'][0]['id']  # Отримати перший доступний пристрій
            sp.start_playback(device_id=device_id, uris=[song_uri])  # Запустити відтворення пісні
            print("Пісня запущена!")
        else:
            print("Немає доступних пристроїв для відтворення.")
    except Exception as e:
        print(f"Помилка при відтворенні пісні: {e}")


def stop_music():
    try:
        sp.pause_playback()  # Зупинити відтворення
        print("Музика зупинена!")
    except Exception as e:
        print(f"Помилка при зупинці музики: {e}")

