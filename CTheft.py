import numpy as np 
import pandas as pd  
import gmplot 

df = pd.read_csv('ITM_20190121.csv')

#print df.describe

print(list(df.columns.values))  # column values


df = (df.sort_values(by='CollectionTimestamp', ascending=True))


df = df.loc[df['ManufacturerSerial'] == 816839]

#print(len(df.ManufacturerSerial.unique().tolist()) ) #unique values

latitude_list = list(df['Latitude'])[:1000]
longitude_list = list(df['Longitude'])[:1000]
print(df)

gmap3 = gmplot.GoogleMapPlotter(list(df['Latitude'])[0],list(df['Longitude'])[0],11)

gmap3.marker(list(df['Latitude'])[0], list(df['Longitude'])[0], 'cornflowerblue')

gmap3.plot(latitude_list, longitude_list, 'cornflowerblue', edge_width=10)


gmap3.draw( 'map.html' ) 
