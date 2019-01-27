import numpy as np 
import pandas as pd  
import gmplot
import matplotlib.pyplot as plt

df = pd.read_csv('ITM_20190121.csv')

df = (df.sort_values(by='CollectionTimestamp', ascending=True))




df.to_csv('nogga_update.csv')

#print df.describe



# print(list(df_org.columns.values))  # column values


# df_org = (df_org.sort_values(by=['ManufacturerSerial','CollectionTimestamp'], ascending=[True,True]))
# #print (df[['CollectionTimestamp','ManufacturerSerial']])

# gmap3 = gmplot.GoogleMapPlotter(list(df_org['Latitude'])[0],list(df_org['Longitude'])[0],11)

# for i,col in zip(df_org['ManufacturerSerial'].drop_duplicates(),list(gmplot.color_dicts.html_color_codes.keys())[:11]):
#     df = df_org.loc[df_org['ManufacturerSerial'] == i]
#     print(df["CollectionTimestamp"])
    

#     latitude_list = list(df['Latitude'])[:10]
#     longitude_list = list(df['Longitude'])[:10]

    
#     #gmap3.marker(list(df['Latitude'])[0], list(df['Longitude'])[0], 'cornflowerblue')

#     gmap3.plot(latitude_list, longitude_list,col, edge_width=2)

#     plt.plot(latitude_list, longitude_list)
# # plt.show()
# gmap3.draw( 'nigaa_map.html' ) 

