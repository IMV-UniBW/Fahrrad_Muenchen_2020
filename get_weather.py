b# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:00:35 2021
import urllib
@author: katha
"""
# -------------------------------------------------------------
# import modules
# -------------------------------------------------------------

import bs4
from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime

# -------------------------------------------------------------
# set variables
# -------------------------------------------------------------
years = ['2017','2018', '2019', '2020']
weeks = range(1,53+1)
columns = ['Datum','Zeit','Temperatur', 'Luftfeuchte', 'Luftdruck', 
           'Regen', 'Windgeschw.', 'Windrichtung', 'Sonnenschein', 
           'UV-Index', 'Solarstrahl.', 'Taupunkt', 'Windchill', 'Windböen']
df_weather = pd.DataFrame(columns= columns)   

# -------------------------------------------------------------
# scrape weather data
# -------------------------------------------------------------

for year in years:
    for week in weeks:
        url = 'http://www.mingaweda.de/archiv/w' + year + '_' + "{:02d}".format(week)+ '.htm'
        page = requests.get(url)
        if page.status_code != 200:
            print([year, week])
        else:
            soup =  BeautifulSoup(page.content,'html.parser')
            table = soup.find_all('table')
            raw_data = [row.get_text(" ").splitlines() for row in table]
            data = raw_data[1]
            str1 = ''.join(data)
            array = str1.split("  ");
            # make matrix
            row = []
            new_array = []
            for i in array[16:array.index('Min-Datum')]:
                if i == '':
                    new_array.extend([row])
                    row= []
                else:
                    row.append(i)
            # labels 1-14     
            df_week = pd.DataFrame(new_array, index = range(array[16:array.index('Min-Datum')].count('')), columns=array[1:15])   
            df_weather = pd.concat([df_weather, df_week])       
        
# -------------------------------------------------------------
# re-format
# -------------------------------------------------------------
# reset index
df_weather = df_weather.reset_index(drop=True)
# remove units and replace decimal comma
# temperature
unit = "°C"
res = [sub.replace(unit, "").strip() for sub in df_weather.Temperatur]
comma = ','
res2 = [sub.replace(comma, ".").strip() for sub in res]
df_weather.Temperatur = res2

# Luftfeuchte
unit = " %"
res = [sub.replace(unit, "").strip() for sub in df_weather.Luftfeuchte]
comma = ','
res2 = [sub.replace(comma, ".").strip() for sub in res]
df_weather.Luftfeuchte = res2


# Luftdruck
unit = " hPa"
res = [sub.replace(unit, "").strip() for sub in df_weather.Luftdruck]
comma = ','
res2 = [sub.replace(comma, ".").strip() for sub in res]
df_weather.Luftdruck = res2

#Regen
unit = " l/m²"
res = [sub.replace(unit, "").strip() for sub in df_weather.Regen]
comma = ','
res2 = [sub.replace(comma, ".").strip() for sub in res]
df_weather.Regen = res2

# Windgeschw.
unit = " km/h"
res = [sub.replace(unit, "").strip() for sub in df_weather['Windgeschw.']]
comma = ','
res2 = [sub.replace(comma, ".").strip() for sub in res]
df_weather.Windgeschwindigkeit = res2
del df_weather['Windgeschw.']

# Sonnenschein
unit = " h"
res = [sub.replace(unit, "").strip() for sub in df_weather.Sonnenschein]
comma = ','
res2 = [sub.replace(comma, ".").strip() for sub in res]
df_weather['Sonnenschein'] = res2


# UV-I
unit = " UV-I"
res = [sub.replace(unit, "").strip() for sub in df_weather['UV-Index']]
comma = ','
res2 = [sub.replace(comma, ".").strip() for sub in res]
df_weather['UV'] = res2
del df_weather['UV-I']

# Solarstrahl
unit = " W/m²"
res = [sub.replace(unit, "").strip() for sub in df_weather['Solarstrahl.']]
comma = ','
res2 = [sub.replace(comma, ".").strip() for sub in res]
df_weather.Solarstrahlung = res2
del df_weather['Solarstrahl.']

# Taupunkt
unit = " °C"
res = [sub.replace(unit, "").strip() for sub in df_weather.Taupunkt]
comma = ','
res2 = [sub.replace(comma, ".").strip() for sub in res]
df_weather.Taupunkt = res2

# Windchill
unit = " °C"
res = [sub.replace(unit, "").strip() for sub in df_weather.Windchill]
comma = ','
res2 = [sub.replace(comma, ".").strip() for sub in res]
df_weather.Windchill = res2

# Windböen
unit = " km/h"
res = [sub.replace(unit, "").strip() for sub in df_weather.Windböen]
comma = ','
res2 = [sub.replace(comma, ".").strip() for sub in res]
df_weather.Windböen = res2
# -------------------------------------------------------------
# save
# -------------------------------------------------------------
df_weather.to_csv('weather2.csv', index=False)      
 