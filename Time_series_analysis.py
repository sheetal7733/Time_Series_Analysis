#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Exercise 1

#Problem 1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'data.csv')

#data1 = pd.read_csv(r'C:\Users\sheet\Downloads\NGC_1275_MJD_54675_59823 - NGC_1275_MJD_54675_59823 .csv')


# In[2]:


data['DATE'] = pd.to_datetime(data['DATE'],infer_datetime_format = True)
index = data.set_index(['DATE'])
from datetime import datetime
index.head()


# In[3]:


plt.xlabel('DATE')
plt.ylabel('Number of Patients')
plt.plot(index)
plt.show()


# In[5]:


#Checking the stationarity of time series
rmean = index.rolling(window = 7).mean()
rstd = index.rolling(window = 7).std()
#print(rmean,rstd)
orig = plt.plot(index,color = 'black',label ='Original')
mean = plt.plot(rmean,color ='red',label ='Rolling mean')
std = plt.plot(rstd,color ='blue',label = 'Rolling Standard Deviation')
plt.legend()
plt.grid()
plt.show()

# Making the time series stationary
if 'Patients' in data.columns:
    data['Patients_diff'] = data['Patients'] - data['Patients'].shift(1)
    data['Patients_diff'].dropna().plot()
    rmean = data['Patients_diff'].rolling(window=12).mean()
    rstd = data['Patients_diff'].rolling(window=12).std()

    # Plotting
    orig = plt.plot(data['Patients_diff'], color='black', label='Original')
    mean = plt.plot(rmean, color='red', label='Rolling mean')
    std = plt.plot(rstd, color='blue', label='Rolling Standard Deviation')
    plt.legend()
    plt.show()

    # Rest of your code that uses the 'Patients_diff' column
else:
    print("The 'Patients' column does not exist in the DataFrame.")



# In[ ]:


#Exercise 1
#Problem 2


# In[7]:


data1 = np.loadtxt("flux_data.txt")


# In[9]:


MJD_start = data1[:,0]
print(MJD_start)
MJD_end = data1[:,1]
#print(MJD_end)
MJD_e = np.array(MJD_start)
MJD_s = np.array(MJD_end)
X = (MJD_e+MJD_s)/2
Observed_flux = data1[:,4]
yerr = data1[:,5]
#print(X)
#print(Observed_flux)
#print(yerr)

#plt.plot(X)
#plt.show()
plt.plot( X,Observed_flux)
plt.show()
#plt.scatter(X,Observed_flux)


# In[ ]:


#checking the stationarity of time series
dataframe = pd.DataFrame(data1[:,4])
rolmean = dataframe.rolling(120).mean()
rolstd = dataframe.rolling(120).std()
orig = plt.plot(dataframe,color = 'black',label ='Original')
mean = plt.plot(rolmean,color ='red',label ='Rolling mean')
std = plt.plot(rolstd,color ='blue',label = 'Rolling Standard Deviation')
plt.legend()
plt.show()


# In[ ]:


#transform the data to make it stationary 
dataframe_diff = pd.DataFrame(data1[:,4])-pd.DataFrame(data1[:,4]).shift(1)
plt.plot(dataframe_diff)
plt.title('Stationary time series')
#plt.savefig("sationary time series of flux data.pdf")
plt.show()
#Stationary time series
rolmean = dataframe_diff.rolling(120).mean()
rolstd = dataframe_diff.rolling(120).std()
orig = plt.plot(dataframe_diff,color = 'black',label ='Original')
mean = plt.plot(rolmean,color ='red',label ='Rolling mean')
std = plt.plot(rolstd,color ='blue',label = 'Rolling Standard Deviation')
plt.legend()
plt.show()


