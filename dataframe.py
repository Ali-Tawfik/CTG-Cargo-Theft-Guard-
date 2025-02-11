import numpy as np 
import pandas as pd  
import gmplot
import matplotlib.pyplot as plt
import datetime as dt

df_org = pd.read_csv('ITM_20190121_updated.csv')

#print df.describe

#print(list(df_org.columns.values))  # column values


#print (df[['CollectionTimestamp','ManufacturerSerial']])


for i,col in zip(df_org['ManufacturerSerial'].drop_duplicates(),gmplot.color_dicts.html_color_codes.keys()[:11]):
    df = df_org.loc[df_org['ManufacturerSerial'] == i]
    print(i,col)

    dt=1

    lat = df['Latitude'].values
    lon = df['Longitude'].values
    vel = ((np.diff(lon)/dt)**2+(np.diff(lat)/dt)**2)**0.5
    vel=np.append(vel,[0])
    
    df["velocity"]=vel
    latitude_list = list(df['Latitude'])
    longitude_list = list(df['Longitude'])
    
    plt.plot(latitude_list, longitude_list,label=i)
    plt.legend()
plt.show()

