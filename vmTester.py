# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:17:27 2022
Testing VM data load
@author: tnsak
"""
import numpy as np
import os

zcDir = 'WigNP_WL128_Int8_Indiv/ZC'
zcNp = os.listdir(zcDir)

oneArr = np.load(os.path.join(zcDir, zcNp[0]))
print(zcNp[0])
print(oneArr.mean())