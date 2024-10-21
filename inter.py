import tkinter as tk
from tkinter import messagebox

# Створюємо головне вікно
root = tk.Tk()
root.title("My Python App")
root.geometry("300x200")

# Функція для натискання кнопки
def on_button_click():
    messagebox.showinfo("Information", "Привіт! Ця кнопка працює!")

# Додаємо віджет (кнопку)
button = tk.Button(root, text="Натисни мене", command=on_button_click)
button.pack(pady=20)

# Запускаємо головний цикл програми
root.mainloop()