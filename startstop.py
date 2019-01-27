import numpy as np 
import pandas as pd  
import gmplot
import matplotlib.pyplot as plt
import datetime as dt
from sklearn import svm

df_org = pd.read_csv('ITM_20190121_updated.csv')
df_org['CollectionTimestamp']= pd.to_datetime(df_org['CollectionTimestamp'])
#dt.d
#print df.describe

#print(list(df_org.columns.values))  # column values

#print (df[['CollectionTimestamp','ManufacturerSerial']])
df_org.index=df_org['CollectionTimestamp']
df_org["speed"]=df_org.MessageXML.str.extract('<Speed>(\d+)').values
trips=[]
maxlen=0
for i,col in zip(df_org['ManufacturerSerial'].drop_duplicates(),gmplot.color_dicts.html_color_codes.keys()[:11]):
    df = df_org.loc[df_org['ManufacturerSerial'] == i]
    print(i,col)
    
    df=df.dropna(subset=["speed"])
    df["speed"]=df["speed"].astype(int)
    startind=df[df["TripState"]=="Engine On"].index
 #   endind=df[df["TripState"]=="Engine Off"].index
  #  head=df[df["ReportType"]=="Trip Start"].index
    movin=df[df["speed"]>=1]
    stopped=df[df["speed"]<1]
    
    starts=df[(df["speed"]>=1).shift(-1)>(df["speed"]>=1)].index
    stops=df[(df["speed"]>=1).shift(-1)<(df["speed"]>=1)].index
    for sta,stp in zip(starts,stops):
        temp=df.loc[sta:stp,:]
        leng=len(temp)
        if maxlen<leng:
            maxlen=leng
        if leng>3:
            trips.append(temp)
    

    print (len(movin),len(stopped))

    plt.plot(df["CollectionTimestamp"],df["speed"].astype(int),'.',label=i)
    plt.legend()

    

    #plt.plot(df["CollectionTimestamp"],df["speed"].astype(int),'.',label=i)

#plt.show()
X=[]
labels=[]
for trip in trips:
    X.append(trip["speed"].tolist())
    labels=labels+trip["ManufacturerSerial"].unique().tolist()
result = np.zeros([len(X),maxlen])
for i in range (len(X)):
    if len(X[i])<maxlen:
        X[i]=X[i]+[0]*(maxlen-len(X[i]))
X=np.array(X)


clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

clf.fit(X)

Y = clf.predict(X)

n = Y[Y == -1].size

labels=np.array(labels)
sus=labels[Y==-1].ravel()
unique,counts=np.unique(labels[Y==-1],return_counts=True)

print(dict(zip(unique,counts)))
    


