import sys
import time
import sounddevice as sd
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
from pyqtgraph import PlotWidget
from PyQt5.QtGui import QCursor


class VoiceAssistantWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Voice Assistant')
        self.setGeometry(100, 100, 800, 300)

        # Вікно завжди зверху і без кнопок керування
        self.setWindowFlags(
            self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint)

        # Фон
        self.setStyleSheet("background-color: black;")

        # Полотно для хвильової анімації
        self.graphWidget = PlotWidget(self)
        self.graphWidget.setGeometry(50, 50, 700, 200)
        self.graphWidget.setBackground('k')
        self.graphWidget.showGrid(x=True, y=True)

        # Лінія для хвиль
        self.wave_line = self.graphWidget.plot(pen=QtGui.QPen(QtGui.QColor("cyan"), 2))

        self.audio_stream = None

    def start_animation(self):
        # Запуск запису аудіо
        self.audio_stream = sd.InputStream(callback=self.audio_callback)
        self.audio_stream.start()

    def stop_animation(self):
        if self.audio_stream is not None:
            self.audio_stream.stop()

    def audio_callback(self, indata, frames, time, status):
        volume_norm = np.linalg.norm(indata) * 10
        x = np.arange(0, len(indata))
        y = indata[:, 0] * 100 + volume_norm

        self.wave_line.setData(x, y)


class VoiceAssistant:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = VoiceAssistantWindow()

    def show_window(self):
        self.window.show()

        # Визначення екрану, на якому знаходиться курсор
        cursor_pos = QCursor.pos()
        screen = QApplication.screenAt(cursor_pos)

        if screen:
            # Центруємо вікно на екрані, де знаходиться курсор
            screen_geometry = screen.availableGeometry()
            self.window.setGeometry(screen_geometry.center().x() - self.window.width() // 2,
                                    screen_geometry.center().y() - self.window.height() // 2,
                                    self.window.width(), self.window.height())

        # Анімація хвиль
        self.window.start_animation()
        self.window.raise_()  # Піднімаємо вікно поверх інших
        self.window.activateWindow()  # Фокусуємо на вікні
        self.app.exec_()

    def hide_window(self):
        self.window.stop_animation()
        self.window.hide()


def wake_word_detected():
    print("Wake word detected!")
    assistant.show_window()


def simulate_wake_word_detection():
    print("Waiting for wake word...")
    time.sleep(3)  # Місце для реального wake word detection
    wake_word_detected()


# Створення об'єкта
assistant = VoiceAssistant()

simulate_wake_word_detection()
