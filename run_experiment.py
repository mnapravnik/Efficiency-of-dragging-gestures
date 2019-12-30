import tkinter as tk
from exp_ui import ExperimentWindow

"""
#===THIS IS COMMENTED BECAUSE THE CONNECTION DATA IS IN AN UNCOMMITTED FILE===
# TODO: Uncomment this for the final experiment
# import to connect to Google Sheets API
from connect import connect
#import to get the current timestamp
import time as time
"""

class MainWindow(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # connect() uses Google Sheets API to connect to 
        # the spreadsheet that we'll write to
        # TODO: Uncomment this for the final experiment
        #self.sheet = connect()
            
        # Creating the UI for entering the participants data
        # and for mode selection 

        # Entering the participants name
        name_label = tk.Label(text='Participants name:')
        name_label.grid(row=0, column=0)
        name = tk.StringVar()
        name_entry = tk.Entry(root,textvariable=name, width=30)
        name_entry.grid(row=0, column=1)

        # Entering the participants age
        age_label = tk.Label(text='Participants age:')
        age_label.grid(row=1, column=0)
        age = tk.StringVar()
        age_entry = tk.Entry(root,textvariable=age, width=30)
        age_entry.grid(row=1, column=1)
        
        # Entering the participants input device
        AVAILABLE_DEVICES = ["Mouse", "Trackpad", "Graphic tablet"]
        device_label = tk.Label(text='Input device:')
        device_label.grid(row=2, column=0)  
        device = tk.StringVar(root)
        device.set(AVAILABLE_DEVICES[0])
        device_entry = tk.OptionMenu(root, device, *AVAILABLE_DEVICES)
        device_entry.grid(row=2, column=1)

        # Entering the difficulty
        DIFFICULTY = ["Easy", "Medium", "Hard"]
        difficulty_label = tk.Label(text='Difficulty:')
        difficulty_label.grid(row=3, column=0)  
        difficulty = tk.StringVar(root)
        difficulty.set(DIFFICULTY[0])
        difficulty_entry = tk.OptionMenu(root, difficulty, *DIFFICULTY)
        difficulty_entry.grid(row=3, column=1)

        # Choosing the order of tasks
        ORDER = ["(1, 2, 3)", "(1, 3, 2)", "(2, 3, 1)", "(2, 1, 3)", "(3, 1, 2)", "(3, 2, 1)"]
        order_label = tk.Label(text='Order of tasks:')
        order_label.grid(row=4, column=0)  
        order = tk.StringVar(root)
        order.set(ORDER[0])
        order_entry = tk.OptionMenu(root, order, *ORDER)
        order_entry.grid(row=4, column=1)

        # Button that starts the experiment
        b = tk.Button(root, text="Start experiment")
        b.grid(row=5, column=0, columnspan=2)    
        b.bind("<Button-1>", lambda e: self.start_experiment(name.get(), age.get(), device.get(), difficulty.get(), order.get(), e))

    """
    # Writes the data entered in the beginning of the 
    # experment to our Google Sheet
    # TODO: Uncomment this for the final experiment
    def write_participants_data(self, row_to_write):
        self.sheet.append_row(row_to_write)  
    """
    
    # Starts the experiment window
    def start_experiment(self, name, age, device, difficulty, order, event):
        # TODO: Uncomment this for the final experiment
        # epoch_time = int(time.time())
        # self.write_participants_data([name, age, device, difficulty, order, epoch_time])
        ExperimentWindow(self.parent) 

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Experiment parameters")
    MainWindow(root).grid(row=0, column=0) 
    root.mainloop()
