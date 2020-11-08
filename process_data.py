import pandas as pd
import numpy as np
import h5py


input_step_size = 50
output_size = 1
sliding_window = True
file_name= 'eth.h5'


# In[19]:


df = pd.read_csv('data/eth.csv').dropna().tail(1000000)
df['Datetime'] = pd.to_datetime(df['Timestamp'],unit='s')
print(df.head())


# In[30]:
feature_columns = ['Close','High','Low','Open','Volume']
input_features = df.loc[:,feature_columns].values
prices= df.loc[:,'Close'].values
times = df.loc[:,'Timestamp'].values


# In[31]:


outputs = []
inputs = []
output_times = []
input_times = []
if sliding_window:
    for i in range(len(prices)-input_step_size-output_size):
        inputs.append(input_features[i:i + input_step_size,:])
        input_times.append(times[i:i + input_step_size])
        outputs.append(prices[i + input_step_size: i + input_step_size+ output_size])
        output_times.append(times[i + input_step_size: i + input_step_size+ output_size])
else:
    for i in range(0,len(prices)-input_step_size-output_size, input_step_size):
        inputs.append(input_features[i:i + input_step_size,:])
        input_times.append(times[i:i + input_step_size])
        outputs.append(prices[i + input_step_size: i + input_step_size+ output_size])
        output_times.append(times[i + input_step_size: i + input_step_size+ output_size])
inputs= np.array(inputs)
outputs= np.array(outputs)
output_times = np.array(output_times)
input_times = np.array(input_times)


# In[34]:
print(inputs.shape)

with h5py.File(file_name, 'w') as f:
    f.create_dataset("inputs", data = inputs)
    f.create_dataset('outputs', data = outputs)
    f.create_dataset("input_times", data = input_times)
    f.create_dataset('output_times', data = output_times)