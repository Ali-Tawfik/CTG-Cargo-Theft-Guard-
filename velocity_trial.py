import numpy as np 
import pandas as pd  
import gmplot
import matplotlib.pyplot as plt
import datetime as dt

df_org = pd.read_csv('ITM_20190121_updated.csv')
df_org['CollectionTimestamp']= pd.to_datetime(df_org['CollectionTimestamp'])
#dt.d
#print df.describe

#print(list(df_org.columns.values))  # column values


#print (df[['CollectionTimestamp','ManufacturerSerial']])
df_org["speed"]=df_org.MessageXML.str.extract('<Speed>(\d+)').values

for i,col in zip(df_org['ManufacturerSerial'].drop_duplicates(),gmplot.color_dicts.html_color_codes.keys()[:11]):
    df = df_org.loc[df_org['ManufacturerSerial'] == i]
    df=df.dropna(subset=["speed"])
    print(i,col)
    latitude_list = list(df['Latitude'])
    longitude_list = list(df['Longitude'])
    plt.plot(df["CollectionTimestamp"],df["speed"].astype(int),'.',label=i)
    
    plt.legend()
    plt.show()

