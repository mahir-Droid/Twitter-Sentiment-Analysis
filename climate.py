# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 00:16:16 2021

@author: soheb
"""

import pandas as pd

dataset = pd.read_excel("climate_change_download_0.xls")
print(dataset.head())

print(dataset["Series name"].value_counts())