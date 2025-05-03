#!/usr/bin/env python
# coding: utf-8

# # Clustrering

# ### K-Means Clustering

# In[17]:


import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


np.random.seed(0)


# In[3]:


X_synthetic, y_synthetic = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
print(X_synthetic)
print(y_synthetic)


# In[4]:


plt.scatter(X_synthetic[:, 0], X_synthetic[:, 1], marker=".")


# In[5]:


k_means_synthetic = KMeans(init="k-means++", n_clusters=4, n_init=12)


# In[6]:


k_means_synthetic.fit(X_synthetic)


# In[7]:


k_means_labels_synthetic = k_means_synthetic.labels_
k_means_labels_synthetic


# In[8]:


k_means_cluster_centers_synthetic = k_means_synthetic.cluster_centers_
k_means_cluster_centers_synthetic


# In[9]:


fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels_synthetic))))
ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
    my_members = (k_means_labels_synthetic == k)
    cluster_center = k_means_cluster_centers_synthetic[k]
    ax.plot(X_synthetic[my_members, 0], X_synthetic[my_members, 1], "w", markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title("KMeans")
ax.set_xticks(())
ax.set_yticks(())
plt.show()


# In[14]:


cust_df = pd.read_csv("customers.csv")
cust_df['Gender'] = cust_df['Gender'].replace({'Male': 1, 'Female': 0})
df = cust_df


# In[15]:


cust_df['Gender'] = cust_df['Gender'].replace({'Male':1,'Female':0})
df = cust_df
df.head()


# In[18]:


scaler = StandardScaler()
Clus_dataSet = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])


# In[19]:


clusterNum = 3
k_means_real = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
k_means_real.fit(Clus_dataSet)
labels_real = k_means_real.labels_
print(labels_real)


# In[20]:


df['cust'] = labels_real
df.head(200)


# In[ ]:




