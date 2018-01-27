# -*- coding: utf-8 -*-
"""
bitget.py - Extracts the value of a single bit within an unsigned integer

Version 1.1
Created on: Oct 14 2016
Last Modified: Jun 09 2017
Author: Logan Wright, logan.wright@colorado.edu

Description:
    Inputs:
        x = 
        n = 

    Output:
        ret_val = 

"""
import numpy as N

def bitget(x, n):
    ref = N.full(x.shape, 2**n, dtype = x.dtype)
    ret_val = N.bitwise_and(x, ref)
    return ret_val