#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inmf_gui.py - INMF GUI

Version 0.2
Created on: Apr 24, 2018
Last Modified: Apr 26, 2018
Author: Logan Wright, logan.wright@colorado.edu
"""

import tkinter
from tkinter import ttk
from tkinter import filedialog

def run(root,endmember_entries):
    import subprocess
    # Generate INMF Input File
    # Parse tkinter variables into a dictionary that is understood by gen_input_file
    endmember_names = list()
    spectral_smooth_windows = list()
    spatial_smooth_windows = list()
    for item in endmember_entries:
        endmember_names.append(item[0].get())
        spectral_smooth_windows.append(item[1].get())
        spatial_smooth_windows.append(item[2].get())

    input_params = {'name':outfile.get(),
                    'file':filename.get(),
                    'endmember_file':endfile.get(),
                    'members': endmember_names,
                    # 'SZA' : 36.79,
                    # 'SunElliptic' : 1.00787,
                    'max_i' : max_i.get(),
                    'epsilon' : stop_cond.get(),
                    'Normaliztion' : norm_type.get(),
                    'spectral_smooth' : spec_smooth.get(),
                    'smooth_spatial' : spatial_smooth.get(),
                    'spectral_win' : spectral_smooth_windows,
                    'spatial_win' : spatial_smooth_windows,
                    'spectral_strength' : spec_str.get(),
                    'spatial_strength' : spatial_str.get(),
                    'spectral_gamma' : spec_gamma.get(),
                    'spatial_gamma' : spatial_gamma.get(),
                    #'roi' : [100, 200, 101, 201],
                    'wvl_rng' : [9,96] }

    # Write to File
    inputpath = input_params['file'].split('.')[0] + '.in'
    f = open(inputpath, 'w')
    for key, value in input_params.items():
        f.write('{0!s} = {1!s}\n'.format(key,value))

    # Destroy GUI
    root.destroy()

    # Command to run inmf_master.py with the newly created input file_entry
    subprocess.run(['python','inmf_master.py',inputpath])


def popup_dialog():
    file = filedialog.askopenfilename()
    filename.set(file)

def endmember_update(entries,vars):
    n = len(entries)

    try:
        N = int(end_num.get())
    except:
        return

    if N < n:
        for i in range(n - N):
            for item in entries[-1]:
                item.grid_forget()
            entries.pop()
    elif N > n:
        for i in range(N - n):
            temp = list()
            temp.append(ttk.Combobox(endframe))
            temp[0].grid(column = 0, row = row0.get())
            vars.append(tkinter.StringVar(value = 5))
            temp.append(ttk.Entry(endframe, textvariable = vars[-1]))
            temp[1].grid(column = 1, row = row0.get())
            vars.append(tkinter.StringVar(value = 5))
            temp.append(ttk.Entry(endframe, textvariable = vars[-1]))
            temp[2].grid(column = 2, row = row0.get())
            row0.set(int(row0.get()) + 1)
            entries.append(temp)

root = tkinter.Tk()
root.title('Informed NMF Algorithm')

s = ttk.Style()
s.configure('my.Label', font = ('Helvetica',16))

topframe = ttk.Frame(root, padding = 10)
topframe.grid(column = 0, row = 0, sticky = ('NWES'), ipadx = 10, ipady = 10)

inmfframe = ttk.Labelframe(root, text = 'INMF Options:', padding = 10)
inmfframe.grid(column = 0, row = 1, sticky = ('NWES'))
inputsframe = ttk.Frame(inmfframe)
inputsframe.grid(column = 0, row = 1, sticky = ('NWES'))
inputs_sep = ttk.Separator(inmfframe, orient = 'horizontal')
inputs_sep.grid(column = 0, row = 9, sticky = 'WE', pady = 10)
endframe = ttk.Frame(inmfframe)
endframe.grid(column = 0, row = 11, sticky = ('NWES'))
end_sep = ttk.Separator(inmfframe, orient = 'horizontal')
end_sep.grid(column = 0, row = 19, sticky = 'WE', pady = 10)
normframe = ttk.Frame(inmfframe)
normframe.grid(column = 0, row = 21, sticky = ('NWES'))
norm_sep = ttk.Separator(inmfframe, orient = 'horizontal')
norm_sep.grid(column = 0, row = 29, sticky = 'WE', pady = 10)
conframe = ttk.Frame(inmfframe)
conframe.grid(column = 0, row = 31, sticky = ('NWES'))
con_sep = ttk.Separator(inmfframe, orient = 'horizontal')
con_sep.grid(column = 0, row = 39, sticky = 'WE', pady = 10)
stopframe = ttk.Frame(inmfframe)
stopframe.grid(column = 0, row = 41, sticky = ('NWES'))

bottomframe = ttk.Frame(root, padding = 10)
bottomframe.grid(column = 0, row = 2, sticky = ('NWES'))

## Topframe Description Text
title_txt = ttk.Label(topframe, text = 'Informed Non-negative Matrix Factorization Algorithm (INMF)', style = 'my.Label')
attrib_txt = ttk.Label(topframe, text = 'INMF was developed by Logan Wright (logan.wright(at)colorado.edu)')
descript_txt = ttk.Label(topframe, wraplength = 640, text = 'This simplifies the use of the INMF algorithm by allowing you to access and modify INMF constraints, normalizations, initial guesses and end conditions without modifying the underlying code.')

title_txt.grid(column = 1, row = 0, sticky = 'N', ipady = 10)
attrib_txt.grid(column = 0, row = 2, columnspan = 3, sticky = 'W')
descript_txt.grid(column = 0, row = 1, columnspan = 3, sticky = 'W')

# INMF Frame Contents
file_entry_label = ttk.Label(inputsframe, text = 'HICO Image Path:')
file_entry_label.grid(column = 1, row = 0, sticky = 'W')
filename = tkinter.StringVar()
file_entry = ttk.Entry(inputsframe, textvariable = filename)
file_entry.grid(column = 2, row = 0, sticky = 'W')
filebrowse_button = ttk.Button(inputsframe, text = 'Select Image File', command = popup_dialog)
filebrowse_button.grid(column = 3, row = 0, sticky = 'W')

outfile = tkinter.StringVar(value = '*_INMF_Results')
outfile_label = ttk.Label(inputsframe, text = 'Save INMF Result as:')
outfile_entry = ttk.Entry(inputsframe, textvariable = outfile)
outfile_label.grid(column = 1, row = 2, sticky = 'W')
outfile_entry.grid(column = 2, row = 2, sticky = 'W')

endfile = tkinter.StringVar()
endfile_label = ttk.Label(inputsframe, text = 'Initialization Endmember Data:')
endfile_entry = ttk.Entry(inputsframe, textvariable = endfile)
endfile_label.grid(column = 1, row = 3, sticky = 'W')
endfile_entry.grid(column = 2, row = 3, sticky = 'W')

inputs_sep = ttk.Separator(inmfframe, orient = 'horizontal')
inputs_sep.grid(column = 0, row = 9, sticky = 'WE', pady = 10)

# Initialization and Endmember Selection
end_num_label = ttk.Label(endframe, text = 'Number of Endmembers:')
end_num = tkinter.IntVar(endframe, value = '5')
end_num.trace('w', lambda a, b, c: endmember_update(endmember_entries))

end_num_entry = ttk.Entry(endframe, textvariable = end_num)

end_type = tkinter.StringVar(value = '1')
end_type_label = ttk.Label(endframe, text = 'Type of Initialization:')
end_type_random = ttk.Radiobutton(endframe, text = 'Random', variable = end_type, value = '0')
end_type_informed = ttk.Radiobutton(endframe, text = 'Informed', variable = end_type, value = '1')

# endmembers = ('1','2','3','4','5','6','7','8','9','10')
# var_endmembers = tkinter.StringVar(value = endmembers)
# endmember_listbox = tkinter.Listbox(endframe, height = 10, listvariable = var_endmembers)
end_num_label.grid(column = 0, row = 1, sticky = ('W'))
end_num_entry.grid(column = 1, row = 1, sticky = ('W') )
end_type_label.grid(column = 0, row = 2, sticky = ('W') )
end_type_random.grid(column = 0, row = 3 )
end_type_informed.grid(column = 1, row = 3 )


end_select_label1 = ttk.Label(endframe, text = 'Endmember Name')
end_select_label2 = ttk.Label(endframe, text = 'Spectral Smoothing Window:')
end_select_label3 = ttk.Label(endframe, text = 'Spatial Smoothing Window:')
end_select_label1.grid(column = 0, row = 5)
end_select_label2.grid(column = 1, row = 5)
end_select_label3.grid(column = 2, row = 5)


endmember_entries = list()
endmember_vars = list()
row0 = tkinter.StringVar(value = 6)

endmember_update(endmember_entries,endmember_vars)

# for i in range(int(end_num.get())):
#     temp = list()
#     temp.append(ttk.Combobox(endframe))
#     temp[0].grid(column = 0, row = row0.get())
#     vars.append(tkinter.StringVar(value = 5))
#     temp.append(ttk.Entry(endframe, textvariable = vars[-1]))
#     temp[1].grid(column = 1, row = row0.get())
#     vars.append(tkinter.StringVar(value = 5))
#     temp.append(ttk.Entry(endframe, textvariable = vars[-1]))
#     temp[2].grid(column = 2, row = row0.get())
#     row0.set(int(row0.get()) + 1)
#     endmember_entries.append(temp)

#   Normalization
norm_type = tkinter.StringVar(value = 'none')
wgt_none = ttk.Radiobutton(normframe, text = 'None', variable = norm_type, value = 'none')
wgt_refl = ttk.Radiobutton(normframe, text = 'Reflectance', variable = norm_type, value = 'refl')
wgt_aso = ttk.Radiobutton(normframe, text = 'ASO', variable = norm_type, value = 'aso')
wgt_bypixel = ttk.Radiobutton(normframe, text = 'Weight by pixel', variable = norm_type, value = 'pixel')
wgt_byspectral = ttk.Radiobutton(normframe, text = 'Weight by Wavelength', variable = norm_type, value = 'spectral')

norm_label = ttk.Label(normframe, text = 'Normalization:')
norm_label.grid(column = 1, row = 11, sticky = ('W'))
wgt_none.grid(column = 1, row = 12, sticky = ('W'))
wgt_refl.grid(column = 1, row = 13, sticky = 'W')
wgt_aso.grid(column = 2, row = 12, sticky = ('W'))
wgt_bypixel.grid(column = 2, row = 13, sticky = ('W'))
wgt_byspectral.grid(column = 3, row = 12, sticky = ('W'))

# Constraints
constraints_label = ttk.Label(conframe, text = 'INMF Constraints:')
constraints_label.grid(column = 1, row = 21, sticky = 'W')

spec_smooth = tkinter.StringVar(value = 'Yes')
spatial_smooth = tkinter.StringVar(value = 'Yes')
spec_smooth_label = ttk.Label(conframe, text = 'Spectral Smoothing:')
spatial_smooth_label = ttk.Label(conframe, text = 'Spatial Smoothing:')
spec_check = ttk.Checkbutton(conframe, variable = spec_smooth, onvalue = 'Yes', offvalue = 'No')
spatial_check = ttk.Checkbutton(conframe, variable = spatial_smooth, onvalue = 'Yes', offvalue = 'No')

spec_str_label = ttk.Label(conframe, text = 'Strength (α):')
spec_str = tkinter.StringVar(value = '0.5')
spec_str_entry = ttk.Entry(conframe, textvariable = spec_str)
spec_gamma_label = ttk.Label(conframe, text = 'Width (γw):')
spec_gamma = tkinter.StringVar(value = '0.01')
spec_gamma_entry = ttk.Entry(conframe, textvariable = spec_gamma)

spatial_str_label = ttk.Label(conframe, text = 'Strength (β):')
spatial_str = tkinter.StringVar(value = '0.5')
spatial_str_entry = ttk.Entry(conframe, textvariable = spatial_str)
spatial_gamma_label = ttk.Label(conframe, text = 'Width (γh):')
spatial_gamma = tkinter.StringVar(value = '0.01')
spatial_gamma_entry = ttk.Entry(conframe, textvariable = spatial_gamma)

spec_smooth_label.grid(column = 1, row = 22, sticky = ('W'))
spec_check.grid(column = 2, row = 22, sticky = ('W'))
spec_str_label.grid(column = 3, row = 22, sticky = ('W'))
spec_str_entry.grid(column = 4, row = 22, sticky = ('W'))
spec_gamma_label.grid(column = 5, row = 22, sticky = ('W'))
spec_gamma_entry.grid(column = 6, row = 22, sticky = 'W')
spatial_smooth_label.grid(column = 1, row = 23, sticky = ('W'))
spatial_check.grid(column = 2, row = 23, sticky = ('W'))
spatial_str_label.grid(column = 3, row = 23, sticky = ('W'))
spatial_str_entry.grid(column = 4, row = 23, sticky = ('W'))
spatial_gamma_label.grid(column = 5, row = 23, sticky = ('W'))
spatial_gamma_entry.grid(column = 6, row = 23, sticky = ('W'))

#   Ending Conditions:
stop_cond_label = ttk.Label(stopframe, text = 'Stop Condition Value:')
stop_cond = tkinter.StringVar(value = '1e-15')
stop_cond_entry = ttk.Entry(stopframe, textvariable = stop_cond)

max_iter_label = ttk.Label(stopframe, text = 'Maximum Number of Iterations:')
max_i = tkinter.StringVar(value = '500')
max_iter_entry = ttk.Entry(stopframe, textvariable = max_i)

stop_cond_label.grid(column = 1, row = 30, sticky = ('W'))
stop_cond_entry.grid(column = 2, row = 30, sticky = ('W'))
max_iter_label.grid(column = 1, row = 32, sticky = ('W'))
max_iter_entry.grid(column = 2, row = 32, sticky = ('W'))

# Run Button,
run_label = ttk.Label(bottomframe, text = 'Warning: Depending on the size of your image and INMF settings, running inmfframe may take a long time.', padding = 5)
run_btn = tkinter.Button(bottomframe, text = 'Run', command = lambda: run(root, endmember_entries), pady = 10, padx = 20)

run_label.grid(column = 0, row = 99)
run_btn.grid(column = 0, row = 100)

root.mainloop()
