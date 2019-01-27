import pandas as pd
import numpy as np  
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans



df=pd.read_csv('/Users/Samrudh/Desktop/VIBRRATION project/mini_dataset.csv')

accl1_data=df['Accl1'].tolist()

accl2_data=df['Accl2'].tolist()

accl3_data=df['Accl3'].tolist()

temperature_data=df['Temperature'].tolist()

dataset_=[]

for _ in range(len(accl1_data)):
	dataset_.append([accl1_data[_],accl2_data[_],accl3_data[_]])

pca = decomposition.PCA(n_components=2) #n_components are the components you want to conserve
pca.fit(dataset_)
X = pca.transform(dataset_)

plot_x=[]
plot_y=[]

for i in range(len(X)):
	plot_x.append(X[i][0])
	plot_y.append(X[i][1])

#fitting the model


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)


# h=.02
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)

# plt.clf()
# plt.imshow(Z,extent=(xx.min(), xx.max(), yy.min(), yy.max()))

# plt.plot(X[:, 0], X[:, 1], 'k.')

# plt.show()


# print list(kmeans.labels_).count(2) # outliers
# print list(kmeans.labels_).count(1) #normal
# print list(kmeans.labels_).count(0) #normal

km_x_1=[]
km_y_1=[]
km_x_0=[]
km_y_0=[]
def datagen():

	if list(kmeans.labels_).count(2) < 200:

		#print len(list(kmeans.labels_))==len(plot_x)

		k_means_result=list(kmeans.labels_)

		labelled_data=[]
		for _ in range(len(plot_x)):
			if k_means_result[_]==2:
				labelled_data.append([accl1_data[_],accl2_data[_],accl3_data[_],1])
			else:
				labelled_data.append([accl1_data[_],accl2_data[_],accl3_data[_],0])


		# print labelled_data[0]
		# print labelled_data[1]
		# print labelled_data[2]

	#verify 
		count=0
		for _ in range(len(labelled_data)):
			if labelled_data[_][3] == 1:
				count+=1
				km_x_1.append(plot_x[_])
				km_y_1.append(plot_y[_])
			else:
				km_x_0.append(plot_x[_])
				km_y_0.append(plot_y[_])



		print count==list(kmeans.labels_).count(2)
		print count
		df = pd.DataFrame.from_records(labelled_data, columns=['a1','a2','a3','Label'])
		# df.to_csv('/Users/Samrudh/Desktop/VIBRRATION project/labelled_dataset.csv',index=False)
		print "done"

	else:
		print "try again"
		datagen()


datagen()
normal_no=list(kmeans.labels_).count(0)+list(kmeans.labels_).count(1)
outlier_no=list(kmeans.labels_).count(2)

plt.plot(km_x_0,km_y_0,'b.',label='Normal Data = '+str(normal_no))
plt.plot(km_x_1,km_y_1,'r.',label='Outliers = '+str(outlier_no))
plt.legend()

plt.show()



