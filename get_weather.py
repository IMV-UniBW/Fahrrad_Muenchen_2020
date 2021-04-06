# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:00:35 2021

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


# -------------------------------------------------------------
# save
# -------------------------------------------------------------
df_weather.to_csv('weather.csv', index=False)      
 