#!/usr/bin/env python
# coding: utf-8

# In[10]:


import scipy.io as sio
import numpy as np
import pickle
import mne
# %matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from multiprocessing import Pool


# In[65]:


#Define Classes
clas=[['4000.0', '40000.0'], ['5000.0', '50000.0']]
folder='one'
try:
    os.mkdir(folder)
dataFolder='allData/'


# In[66]:


files=[]
for file in os.listdir(dataFolder):
    if file.endswith(".set"):
        files.append(file)
print(files)


# In[13]:


offset=0.1 #Seconds before the event to ignore
sampleLen=5 #Seconds


# In[14]:


li=[[1, 2], [3, 44]]
[3, 44] in li


# In[15]:


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 
  


# In[42]:


def getData(name):
    
    raw=mne.io.read_raw_eeglab(dataFolder+name)
    sRate=raw.info['sfreq']

    (events,
     event_dict)=mne.events_from_annotations(raw)
    importantEvents=[]
    ke=list(event_dict.keys())
    found=[False]*len(clas)
    for i in range(len(clas)):

        for k in range(len(clas[i])):
            if clas[i][k] in ke:
                found[i]=True
        clas[i]=intersection(event_dict, clas[i])
    for i in range(len(clas)):
        marker=[]
        for k in range(len(clas[i])):
            for p in range(len(events)):
                if events[p][2]==event_dict[clas[i][k]]:
                    corrupted=False
                    #Check to see if error marker present in sample
                    end=events[p][0]-((sampleLen+offset)*sRate)
    #                 star=events[p][0]-offset
                    if end<0:
                        end=0
                    for j in range(p, -1, -1):
                        if events[j][0]>=end:
                            if events[j][2]==event_dict['-999.0']:
                                corrupted=True
                        else:
                            break
                    if not corrupted:    
                        marker.append(events[p][0])
        importantEvents.append(marker)
    data=[]
    for i in range(len(importantEvents)):
        feature=[]
        for k in range(len(importantEvents[i])):
            star=int(importantEvents[i][k]-((sampleLen+offset)*sRate))
            end=int(importantEvents[i][k]-(offset*sRate))
            out, times=raw[:, star:end]
    #         print(k)
            feature.append(out.transpose())
        data.append(np.array(feature))
    data=np.array(data)
    if found == [True]*len(found):
        pickle.dump(data, open(folder+'/'+name, 'wb'))
    else:
        print('SKIPPING', name, clas)
    return data


# In[46]:


p = Pool(20)
out=p.map(getData, files)


# In[64]:


master=[]
classes=len(clas)
for i in range(classes):
    chunk=[]
    for k in range(len(out)):
        for p in range(len(out[k][i])):
            chunk.append(out[k][i][p])
    master.append(chunk)
pickle.dump(master, open(folder+'/'+'generic', 'wb'))


# In[ ]:




