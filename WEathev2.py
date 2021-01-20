#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow  as tf
import os


# In[2]:


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu_devices))
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
tf.config.optimizer.set_jit(False)


# In[3]:


frame = pd.read_csv(r'weekcount.csv')
df = pd.DataFrame(frame)
df =pd.get_dummies(df, prefix=['station' ])

#reading and dumming datta
col = list(df.columns.values)

df = df[['year', 'week', 'station_5 Corners Library', 'station_Astor Place', 'station_Baldwin at Montgomery', 'station_Bayside Park', 'station_Bergen Ave', 'station_Bethune Center', 'station_Brunswick & 6th', 'station_Brunswick St', 'station_Central Ave', 'station_Christ Hospital', 'station_City Hall', 'station_Columbia Park', 'station_Columbus Dr at Exchange Pl', 'station_Columbus Drive', 'station_Communipaw & Berry Lane', 'station_Danforth Light Rail', 'station_Dey St', 'station_Dixon Mills', 'station_Essex Light Rail', 'station_Exchange Place', 'station_Fairmount Ave', 'station_Garfield Ave Station', 'station_Glenwood Ave', 'station_Grand St', 'station_Grove St PATH', 'station_Hamilton Park', 'station_Harborside', 'station_Heights Elevator', 'station_Hilltop', 'station_Hoboken Ave at Monmouth St', 'station_JC Medical Center', 'station_JCBS Depot', 'station_Jackson Square', 'station_Jersey & 3rd', 'station_Jersey & 6th St', 'station_Journal Square', 'station_Lafayette Park', 'station_Leonard Gordon Park', 'station_Liberty Light Rail', 'station_Lincoln Park', 'station_MLK Light Rail', 'station_Manila & 1st', 'station_Marin Light Rail', 'station_McGinley Square', 'station_Monmouth and 6th', 'station_Montgomery St', 'station_Morris Canal', 'station_NJCU', 'station_Newark Ave', 'station_Newport PATH', 'station_Newport Pkwy', 'station_North St', 'station_Oakland Ave', 'station_Paulus Hook', 'station_Pershing Field', 'station_Riverview Park', 'station_Sip Ave', 'station_Union St', 'station_Van Vorst Park', 'station_Warren St', 'station_Washington St', 'station_West Side Light Rail', 'station_York St', 'anzahl']]



print(col)
print(df)


# In[4]:


wframe = pd.read_csv(r'weather.csv')
weather = pd.DataFrame(wframe)
weather.head()


# In[5]:


df = pd.merge(df,weather,on=['year','week'],how='outer').dropna()
#df = df.reset_index()
print(df[df.index.duplicated()])
df = pd.DataFrame(df,columns=['year', 'week', 'station_5 Corners Library', 'station_Astor Place', 'station_Baldwin at Montgomery', 'station_Bayside Park', 'station_Bergen Ave', 'station_Bethune Center', 'station_Brunswick & 6th', 'station_Brunswick St', 'station_Central Ave', 'station_Christ Hospital', 'station_City Hall', 'station_Columbia Park', 'station_Columbus Dr at Exchange Pl', 'station_Columbus Drive', 'station_Communipaw & Berry Lane', 'station_Danforth Light Rail', 'station_Dey St', 'station_Dixon Mills', 'station_Essex Light Rail', 'station_Exchange Place', 'station_Fairmount Ave', 'station_Garfield Ave Station', 'station_Glenwood Ave', 'station_Grand St', 'station_Grove St PATH', 'station_Hamilton Park', 'station_Harborside', 'station_Heights Elevator', 'station_Hilltop', 'station_Hoboken Ave at Monmouth St', 'station_JC Medical Center', 'station_JCBS Depot', 'station_Jackson Square', 'station_Jersey & 3rd', 'station_Jersey & 6th St', 'station_Journal Square', 'station_Lafayette Park', 'station_Leonard Gordon Park', 'station_Liberty Light Rail', 'station_Lincoln Park', 'station_MLK Light Rail', 'station_Manila & 1st', 'station_Marin Light Rail', 'station_McGinley Square', 'station_Monmouth and 6th', 'station_Montgomery St', 'station_Morris Canal', 'station_NJCU', 'station_Newark Ave', 'station_Newport PATH', 'station_Newport Pkwy', 'station_North St', 'station_Oakland Ave', 'station_Paulus Hook', 'station_Pershing Field', 'station_Riverview Park', 'station_Sip Ave', 'station_Union St', 'station_Van Vorst Park', 'station_Warren St', 'station_Washington St', 'station_West Side Light Rail', 'station_York St','sunHour','totalSnow_cm','FeelsLikeC','cloudcover','humidity','precipMM','tempC','windspeedKmph', 'anzahl'])
df.drop(df[df['year'] == 2020].index, inplace = True) #'REMOVE2020'
#df.drop(df[df['year'] == 2015].index, inplace = True) #remove 2015
#df.drop(df[df['year'] == 2019].index, inplace = True) #'REMOVE2020'


# In[6]:


df.to_csv('kitrainweathe.csv',index=False)


# In[ ]:





# In[7]:


x = df.iloc[:,0:-1].values
y = df.iloc[:,73:74].values
np.set_printoptions(suppress=True)
print(x[21])
print(y[21])
print(x)
print(y)


# In[8]:


from sklearn.preprocessing import MinMaxScaler
x_scaler = MinMaxScaler(feature_range=(-1,1))
y_scaler = MinMaxScaler(feature_range=(-1,1))
# Fit train data 
x_scaler.fit(x)
y_scaler.fit(y)
x = x_scaler.transform(x)
y = y_scaler.transform(y)


# In[9]:


split_horizontally_idx = int(x.shape[0]* 0.8)
x_train = x[:split_horizontally_idx , :]
x_test = x[split_horizontally_idx: , :]
y_train = y[:split_horizontally_idx , :]
y_test = y[split_horizontally_idx: , :]
print(x_train.shape); print(y_test.shape)


# In[10]:


#reshape
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
print(x_test.shape)
print(x_train.shape)


# In[11]:


#with tf.device('/device:GPU:0'):
#    model = keras.models.load_model('kiweatherv2')


# In[12]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

from tensorflow.keras.callbacks import ModelCheckpoint

def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=4, dropout=0.3,
                loss='mse', optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mae",'mape'], optimizer=optimizer)
    return model

model = create_model(x_train.shape[1],x_train.shape[2])
#checkpoint = ModelCheckpoint('kiweatherv2', monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq=83750)


# In[ ]:


#0.0001
#model.compile(optimizer=Adam(),loss='mse', metrics=['mae','mape'])


# In[ ]:


with tf.device('/device:GPU:0'):
    print(tf.device)
    model.fit(x_train, y_train, epochs=100 , batch_size=100,shuffle=True )


# In[ ]:


with tf.device('/device:GPU:0'):
    results = model.evaluate(x_test, y_test )
print("test loss, test acc:", results)


# In[ ]:


test_data = x_test[21] 
print(test_data.shape)
test_data = np.reshape(test_data,(test_data.shape[1],test_data.shape[0],1))
print(test_data.shape)
with tf.device('/device:CPU:0'):
    print(y_scaler.inverse_transform(model.predict(test_data)))
print(y_scaler.inverse_transform(y_test)[21])


# In[13]:


#save model
model.save('kiweatherv2')


# In[ ]:


predict_prices = model.predict(x_train)
x = np.arange(y_train.shape[0]).reshape( y_train.shape[0])
y1 = y_scaler.inverse_transform(y_train).reshape(  y_train.shape[0])
y2 = y_scaler.inverse_transform(predict_prices).reshape(  y_train.shape[0])
test = pd.DataFrame({ 'y':y1 ,'y2':y2})
test = test.sort_values(by=['y'])
plt.plot(x,test['y2'] ,x,test['y'] )


# In[ ]:


predict_prices = model.predict(x_test)
x = np.arange(y_test.shape[0]).reshape( y_test.shape[0])
y1 = y_scaler.inverse_transform(y_test).reshape(  y_test.shape[0])
y2 = y_scaler.inverse_transform(predict_prices).reshape(  y_test.shape[0])
test = pd.DataFrame({ 'y':y1 ,'y2':y2})
test = test.sort_values(by=['y'])
plt.plot(x,test['y2'] ,x,test['y'] )


# In[ ]:




