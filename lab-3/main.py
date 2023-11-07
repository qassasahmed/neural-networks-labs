import pandas as pd
import perc as p
import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.filedialog as fd
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)


if __name__ == '__main__':

    def show_res():
        x, t = read_file()
        pre = p.Perceptron(len(x[0]), 1)
        epochs = pre.run(x, t)
        labels = []
        for i in range(len(epochs)):
            labels.append(ttk.Label(frame,
                                    text=f"Epoch {i + 1}: {[int(c) for c in epochs[i]]}",
                                    style="TLabel"))
            labels[i].pack(padx=10, pady=10)
        labels.append(ttk.Label(frame,
                                text=f"Operation Complete!",
                                style="TLabel"))
        labels[-1].pack(padx=10, pady=10)


    def read_file():
        try:
            file_path = fd.askopenfilename()
            df = pd.read_csv(file_path)
            # print(df)
            target = df.iloc[:, -1].values.tolist()
            features = df.iloc[:, :-1].values.tolist()
            return features, target
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    root = tk.Tk()
    root.title("Perceptron")
    root.geometry("500x500")
    # root.resizable(False, False)
    frame = tk.Frame(root)
    frame.pack()

    style_label = ttk.Style()
    style_label.configure("TLabel",
                          font=("Sans Serif", 15),
                          padding=10)

    style_btn = ttk.Style()
    style_btn.configure("TButton",
                        font=("Sans Serif", 15),
                        padding=10)

    btn_run = ttk.Button(frame,
                         text="Run",
                         command=show_res,
                         style="TButton")
    btn_run.pack(padx=10, pady=10)

    root.mainloop()
