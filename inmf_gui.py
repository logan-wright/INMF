#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inmf_gui.py - INMF GUI

"""
import tkinter
from tkinter import ttk
from tkinter import filedialog

def run(root):
    print('Running!')
    root.destroy()

def popup_dialog():
    file = filedialog.askopenfilename()
    filename.set(file)

root = tkinter.Tk()
root.title('Informed NMF Algorithm')

s = ttk.Style()
s.configure('my.Label', font = ('Helvetica',16))

topframe = ttk.Frame(root, padding = 20)
topframe.grid(column = 0, row = 0, sticky = ('N','W','E','S'), ipadx = 20, ipady = 20)

## Add Text Frame with Attribution, email address, etc,
title_txt = ttk.Label(topframe, text = 'Informed Non-negative Matrix Factorization Algorithm (INMF)', style = 'my.Label')
# title_txt.configure()
attrib_txt = ttk.Label(topframe, text = 'INMF was developed by Logan Wright (logan.wright(at)colorado.edu)')
descript_txt = ttk.Label(topframe, wraplength = 640, text = 'This simplifies the use of the INMF algorithm by allowing you to access and modify INMF constraints, normalizations, initial guesses and end conditions without modifying the underlying code.')

title_txt.grid(column = 1, row = 0, sticky = 'N')
attrib_txt.grid(column = 0, row = 2, columnspan = 3, sticky = 'W')
descript_txt.grid(column = 0, row = 1, columnspan = 3, sticky = 'W')

# Entries to select HICO file
midframe = ttk.Labelframe(root, text = 'INMF Options:', padding = 20)
midframe.grid(column = 0, row = 1, sticky = ('N','W','E','S'))

file_entry_label = ttk.Label(midframe, text = 'HICO Image Path:')
file_entry_label.grid(column = 1, row = 10, sticky = 'W')
filename = tkinter.StringVar()
file_entry = ttk.Entry(midframe, textvariable = filename)
file_entry.grid(column = 2, row = 10, sticky = 'W')
filebrowse_button = ttk.Button(midframe, text = 'Select Image File', command = popup_dialog)
filebrowse_button.grid(column = 3, row = 10, sticky = 'W')

outfile = tkinter.StringVar()
outfile_label = ttk.Label(midframe, text = 'Save INMF Result as:')
outfile_entry = ttk.Entry(midframe, textvariable = outfile)
outfile_label.grid(column = 1, row = 11, sticky = 'W')
outfile_entry.grid(column = 2, row = 11, sticky = 'W')

# NMF Options:
#   Initial Endmembers

end_num_label = ttk.Label(midframe, text = 'Number of Endmembers:')
end_num = tkinter.StringVar(midframe, value = '5')
end_num_entry = ttk.Entry(midframe, textvariable = end_num)

end_type = tkinter.StringVar(value = '1')
end_type_label = ttk.Label(midframe, text = 'Type of Initialization:')
end_type_random = ttk.Radiobutton(midframe, text = 'Random', variable = end_type, value = '0')
end_type_informed = ttk.Radiobutton(midframe, text = 'Informed', variable = end_type, value = '1')

endmembers = ['1','2','3','4','5','6','7','8','9','10']
endmember_listbox = tkinter.Listbox(midframe, height = 10, listvariable = endmembers)
end_num_label.grid(column = 1, row = 15, sticky = ('W'))
end_num_entry.grid(column = 2, row = 15, sticky = ('W') )
end_type_label.grid(column = 1, row = 16, sticky = ('W') )
end_type_random.grid(column = 1, row = 17 )
end_type_informed.grid(column = 2, row = 17 )
endmember_listbox.grid(column = 3, row = 18, sticky = ('W','N') )

#   Normalization
norm_type = tkinter.StringVar(value = '0')
wgt_none = ttk.Radiobutton(midframe, text = 'None', variable = norm_type, value = '0')
wgt_refl = ttk.Radiobutton(midframe, text = 'Reflectance', variable = norm_type, value = '1')
wgt_aso = ttk.Radiobutton(midframe, text = 'ASO', variable = norm_type, value = '2')
wgt_bypixel = ttk.Radiobutton(midframe, text = 'Weight by pixel', variable = norm_type, value = '3')
wgt_byspectral = ttk.Radiobutton(midframe, text = 'Weight by Wavelength', variable = norm_type, value = '4')

norm_label = ttk.Label(midframe, text = 'Normalization:')
norm_label.grid(column = 1, row = 20, sticky = ('W'))
wgt_none.grid(column = 1, row = 21, sticky = ('W'))
wgt_aso.grid(column = 1, row = 22, sticky = ('W'))
wgt_bypixel.grid(column = 2, row = 21, sticky = ('W'))
wgt_byspectral.grid(column = 2, row = 22, sticky = ('W'))

#   Ending Conditions:
stop_cond_label = ttk.Label(midframe, text = 'Stop Condition Value:')
stop_cond = tkinter.StringVar(midframe,value = '1e-15')
stop_cond_entry = ttk.Entry(midframe, textvariable = stop_cond)

max_iter_label = ttk.Label(midframe, text = 'Maximum Number of Iterations:')
max_i = tkinter.StringVar(midframe,value = '500')
max_iter_entry = ttk.Entry(midframe, textvariable = max_i)

stop_cond_label.grid(column = 1, row = 30, sticky = ('W'))
stop_cond_entry.grid(column = 2, row = 30, sticky = ('W'))
max_iter_label.grid(column = 1, row = 32, sticky = ('W'))
max_iter_entry.grid(column = 2, row = 32, sticky = ('W'))

# Run Button,
bottomframe = ttk.Frame(root, padding = 20)
bottomframe.grid(column = 0, row = 2, sticky = ('N','W','E','S'))

run_label = ttk.Label(bottomframe, text = 'Warning: Depending on the size of your image and INMF settings, running INMF may take a long time.', padding = 5)
run_btn = tkinter.Button(bottomframe, text = 'Run', command = lambda: run(root), pady = 20, padx = 20)

run_label.grid(column = 0, row = 99)
run_btn.grid(column = 0, row = 100)

root.mainloop()
