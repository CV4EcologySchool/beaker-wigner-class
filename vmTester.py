# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:17:27 2022
Testing VM data load
@author: tnsak
"""
import numpy as np
import os

# zcDir = '/home/tsakai/CV4Ecology/data/WigNP_WL128_Int8_Indiv/ZC'
zcDir = 'data/WigNP_WL128_Int8_Indiv/ZC'
zcNp = os.listdir(zcDir)

oneArr = np.load(os.path.join(zcDir, 'PASCAL_28_J_OE16_257007734_C2.npy'))
# print(zcNp[0])
print(oneArr.mean())
#%%
# [(index, value) for index, value in enumerate(zcNp) if value == 'PASCAL_28_J_OE16_257007734_C2.npy']
