# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:49:34 2022

@author: tnsak
"""
#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
#%%
wigPath = './WigNP_WL128_Int8'
npzList = os.listdir(wigPath)

#%%
npArr = np.load(os.path.join(wigPath, npzList[0]))['arr_0']
#%%
oneArr = npArr[:,:,0]
# have to flip to match image format of R
# acess is identical tho, row 6 R == row 5 py
plt.imshow(np.flip(npArr[:,:,0], 0))

#%%
# csv