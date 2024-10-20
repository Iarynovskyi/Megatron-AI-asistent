from dotenv import load_dotenv
import os

# Find .env file with os variables
load_dotenv()

# Конфигурация
VA_NAME = 'Megatron'
VA_VER = "3.0"
VA_ALIAS = ('Мегатрон',)
VA_TBR = ('скажи', 'покажи', 'відповіси', 'розкажи', 'скільки', 'слухай')

# ID микрофона (можете просто менять ID пока при запуске не отобразится нужный)
# -1 это стандартное записывающее устройство
MICROPHONE_INDEX = -1

# Путь к браузеру Google Chrome
CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome %s'

# Токен Picovoice
PICOVOICE_TOKEN = os.getenv('PICOVOICE_TOKEN')