# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:08:32 2021

@author: katha
"""

# load modules
import requests
import json
import datetime
from additional_functions import *
import pandas as pd

# ------------------
# import via API
# ------------------
url = 'https://www.opengov-muenchen.de/api/3/action/package_search?q=Raddauerz%C3%A4hlstellen&rows=60'
page = requests.get(url)
jpage = page.json()['result']['results']
df_fahrrad = pd.read_csv(jpage[1]['resources'][0]['url']) # juni 2017
df_fahrrad[['Zeit']] = make_datetime(df_fahrrad.datum, df_fahrrad.uhrzeit_start)
# merge
for i, j in enumerate(jpage):
    if i > 1:
        # url_list.append(j['resources'][0]['url'])
        df_temp = pd.read_csv(j['resources'][0]['url'])
        if len(df_temp.columns) == 1:
            df_temp = pd.read_csv(j['resources'][0]['url'], sep=';')
        elif len(df_temp.columns) == 12:
            df_temp = pd.read_csv(j['resources'][1]['url'])
        df_temp[['Zeit']] = make_datetime(df_temp.datum, df_temp.uhrzeit_start)
        # merge files
        df_fahrrad = df_fahrrad.append(df_temp)

# ----------------
# Alternative: download all csv files manually and merge them
# --------------------
# import os
# os.chdir(r"C:/Users\katha\OneDrive\Dokumente\GitHub\Fahrrad_Muenchen_2020")
# from os import walk
# _, _, filenames = next(walk(r"C:/Users\katha\OneDrive\Dokumente\GitHub\Fahrrad_Muenchen_2020\Daten"))
# df_fahrrad = pd.read_csv(filenames[0], sep=',')
# for file in filenames[1:]:
#     if file == 'rad20180315min.csv':
#         df_temp = pd.read_csv(file, sep=';')
#     else:
#         df_temp = pd.read_csv(file, sep=',')
#     df_fahrrad = df_fahrrad.append(df_temp)


# -----------------------------------------------------------
# clean and sum per hour
# -----------------------------------------------------------
# del df_fahrrad['richtung_1']
# del df_fahrrad['richtung_2']
# del df_fahrrad['uhrzeit_start']
# del df_fahrrad['uhrzeit_ende']
# del df_fahrrad['datum']

df_fahrrad_summarized = df_fahrrad.groupby(['Zeit', 'zaehlstelle'], as_index=False).agg({"gesamt": "sum"})
df_fahrrad_summarized = df_fahrrad_summarized.reset_index(drop=True)

df_fahrrad_summarized.to_csv('radzaehldaten_stunde.csv', index=False)      
df_fahrrad_summarized.to_pickle('radzaehldaten_stunde')

