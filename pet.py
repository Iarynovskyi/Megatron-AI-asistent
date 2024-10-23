import pyautogui
import random
import tkinter as tk
from PIL import Image, ImageTk

x = 400  # Поточна координата X
y = 550  # Поточна координата Y
cycle = 0
check = 1
idle_num = [1, 2, 3, 4]
sleep_num = [10, 11, 12, 13, 15]
walk_left = [6, 7]
walk_right = [8, 9]
event_number = random.randrange(1, 3, 1)
impath = "/Users/mac/Desktop/Unik/My_Projects/Megatron-AI-asistent/gifs/"

drag_data = {"x": 0, "y": 0, "item": None}  # Змінна для зберігання даних про перетягування

def process_gif_frame(image_path):
    image = Image.open(image_path)
    image = image.convert("RGBA")

    data = image.getdata()
    new_data = []
    for item in data:
        if item[:3] == (0, 0, 0):  # Чорний піксель
            new_data.append((255, 255, 255, 0))  # Прозорий
        else:
            new_data.append(item)
    image.putdata(new_data)

    return ImageTk.PhotoImage(image)

def on_drag_start(event):
    drag_data["x"] = event.x
    drag_data["y"] = event.y

def on_drag_motion(event):
    global x, y
    new_x = window.winfo_x() + event.x - drag_data["x"]  # Оновлення по осі X
    new_y = window.winfo_y() + event.y - drag_data["y"]  # Оновлення по осі Y
    window.geometry(f"+{new_x}+{new_y}")
    x, y = new_x, new_y  # Фіксація нових координат

def on_drag_end(event):
    drag_data["x"] = 0
    drag_data["y"] = 0

def event(cycle, check, event_number, x):
    if event_number in idle_num:
        check = 0
        window.after(400, update, cycle, check, event_number, x)

    elif event_number == 5:
        check = 1
        window.after(100, update, cycle, check, event_number, x)
    elif event_number in walk_left:
        check = 4
        window.after(100, update, cycle, check, event_number, x)
    elif event_number in walk_right:
        check = 5
        window.after(100, update, cycle, check, event_number, x)
    elif event_number in sleep_num:
        check = 2
        window.after(1000, update, cycle, check, event_number, x)
    elif event_number == 14:
        check = 3
        window.after(100, update, cycle, check, event_number, x)

def gif_work(cycle, frames, event_number, first_num, last_num):
    if cycle < len(frames) - 1:
        cycle += 1
    else:
        cycle = 0
        event_number = random.randrange(first_num, last_num + 1, 1)
    return cycle, event_number

def update(cycle, check, event_number, x):
    global y  # Додаємо глобальну змінну для Y
    if check == 0:
        frame = idle[cycle]
        cycle, event_number = gif_work(cycle, idle, event_number, 1, 9)
    elif check == 1:
        frame = idle_to_sleep[cycle]
        cycle, event_number = gif_work(cycle, idle_to_sleep, event_number, 10, 10)
    elif check == 2:
        frame = sleep[cycle]
        cycle, event_number = gif_work(cycle, sleep, event_number, 10, 15)
    elif check == 3:
        frame = sleep_to_idle[cycle]
        cycle, event_number = gif_work(cycle, sleep_to_idle, event_number, 1, 1)
    elif check == 4:
        frame = walk_positive[cycle]
        cycle, event_number = gif_work(cycle, walk_positive, event_number, 1, 9)
        x -= 3
    elif check == 5:
        frame = walk_negative[cycle]
        cycle, event_number = gif_work(cycle, walk_negative, event_number, 1, 9)
        x -= -3

    window.geometry(f'100x100+{x}+{y}')
    label.configure(image=frame)
    label.image = frame  # Щоб уникнути garbage collection
    window.after(1, event, cycle, check, event_number, x)

window = tk.Tk()

# Виклик дій
idle = [process_gif_frame(impath + 'output-onlinegiftools (2).gif') for i in range(5)]  # idle gif
idle_to_sleep = [tk.PhotoImage(file=impath + 'change-to-sleep.gif', format='gif -index %i' % (i)) for i in range(8)]
sleep = [tk.PhotoImage(file=impath + 'sleep.gif', format='gif -index %i' % (i)) for i in range(3)]
sleep_to_idle = [tk.PhotoImage(file=impath + 'back-from-sleep.gif', format='gif -index %i' % (i)) for i in range(8)]
walk_positive = [tk.PhotoImage(file=impath + 'output-onlinegiftools.gif', format='gif -index %i' % (i)) for i in range(8)]
walk_negative = [tk.PhotoImage(file=impath + 'output-onlinegiftools (1).gif', format='gif -index %i' % (i)) for i in range(8)]

# Конфігурація вікна
window.config(highlightbackground='black')
label = tk.Label(window, bd=0, bg='black')
window.overrideredirect(True)
window.wm_attributes('-alpha', 0.9)
window.wm_attributes('-topmost', True)

label.pack()

# Додаємо обробники для перетягування
label.bind("<Button-1>", on_drag_start)  # Початок перетягування
label.bind("<B1-Motion>", on_drag_motion)  # Процес перетягування
label.bind("<ButtonRelease-1>", on_drag_end)  # Завершення перетягування

# Запуск циклу
window.after(1, update, cycle, check, event_number, x)
window.mainloop()
