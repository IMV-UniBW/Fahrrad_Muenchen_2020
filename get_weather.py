# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:00:35 2021

@author: katha
"""

import bs4
from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime

#url='http://www.estesparkweather.net/archive_reports.php?date=200901'

month = 11
year = 2020
url = 'https://www.timeanddate.de/wetter/deutschland/muenchen/rueckblick?month=' + str(month) + '&year=' + str(year)

page = requests.get(url)
print(page)
soup =  BeautifulSoup(page.content,'html.parser')
#print(soup)

table = soup.find_all('table')
raw_data = [row.get_text(" ").splitlines() for row in table]
test = raw_data[1][0]
test = test.replace(u'\xa0', u'')
test = test.repla ce(u'k.A.', u'NA')
colon_list = [i for i in range(len(test)) if test.startswith(':', i)]
colon_list.extend([i for i in range(len(test)) if test.startswith('Wetterdaten', i)])
weather_info = []
for i, t_i in enumerate(colon_list[:-1]):
    if i == (len(colon_list)-2):
        weather_info.append(test[(t_i-2):colon_list[i+1]-1])
    else:     
        weather_info.append(test[(t_i-2):colon_list[i+1]-3])

# make weather variables
import datetime
date_dict = {'Jan': 1, 'Feb':2, 'Mar':3,
             'Apr':4, 'Mai':5, 'Jun':6, 
             'Jul':7, 'Aug':8, 'Sep':9,
             'Okt':10, 'Nov':11, 'Dez':12}
datum = []
uhrzeit = []
temperatur = []
wetter = []
datum_ii = weather_info[0][6:(weather_info[0].find('°')-2)].replace('.', '').replace(',', '')
day = [int(s) for s in datum_ii.split() if s.isdigit()]
datum_i = datetime.date(year,month,day[0])
for i in weather_info:
    datum.append(datum_i)
    uhrzeit.append(datetime.time(int(i[:2]), int(i[3:5])))
    temperatur.append(str(i[(i.find('°')-2):(i.find('°'))])) 
    i_short = i[(i.find('°')+3):]
    wetter.append(i_short[:i_short.find('.')])

# in df
df = pd.DataFrame({'datum':datum, 'uhrzeit':uhrzeit, 'temperatur':temperatur, 'wetter':wetter})
# drop down menu
#Dates_r = pd.date_range(start = '1/1/2009',end = '08/05/2020',freq = 'M')
#dates = [str(i)[:4] + str(i)[5:7] for i in Dates_r]
#dates[0:5]
#for k in range(len(dates)):
#    url = "http://www.estesparkweather.net/archive_reports.php?date="
#    url += dates[k]