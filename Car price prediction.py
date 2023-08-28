#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn


# In[2]:


df= pd.read_csv(r'C:\\Users\\HP\\Downloads\\Car.csv') 
df


# In[3]:


# check null value
df.isnull().sum()


# In[4]:


# drop null values
df=df.dropna()  


# In[5]:


# reset- index after drop value
df=df.reset_index(drop=True)  


# In[6]:


df.isnull().sum()


# In[7]:


# seats

print(df['seats'].value_counts())  # show unique seats cars and unka count 


# In[8]:


print(df['seats'].unique())  # unique seats


# In[9]:


# torque: 
print(df['torque'].dtype)  
                                           


# In[10]:


df=df.drop(columns='torque')


# In[11]:


# create function to split first index
def fxn(x):
    return x.split(' ')[0]


# In[12]:


# apply function on max_power column
df['max_power']=df['max_power'].apply(fxn)


# In[13]:


# change datatype
corrupt=[]
for i in range(len(df)):
    try:
            float(df.iloc[i,-2])  
    except:                        
        corrupt.append(i)    


# In[14]:


df=df.drop(index=corrupt)  


# In[15]:


df=df.reset_index(drop=True) 


# In[16]:


# change datatype
df['max_power']=df['max_power'].astype("float32")
print(df['max_power'].dtype)                              


# In[17]:


# engine
print(df['engine'].dtype)
df['engine']=df['engine'].apply(fxn)   


# In[18]:


corrupt=[]
for i in range(len(df)):
    try:
            float(df.iloc[i,-3])    
    except:
        corrupt.append(i)


# In[19]:


df=df.drop(index=corrupt)
df=df.reset_index(drop=True)


# In[20]:


df['engine']=df['engine'].astype("float32")
print(df['engine'].dtype)


# In[21]:


# mileage
print(df['mileage'].dtype)    
df['mileage']=df['mileage'].apply(fxn)
df['mileage']=df['mileage'].astype("float32")
print(df['mileage'].dtype)


# In[22]:


#owner
print(df['owner'].value_counts())
print(df['owner'].dtype)

# merge owners
df['owner']=df['owner'].replace({"Fifth":"Fourth & Above Owner"}) 
print(df['owner'].value_counts())  


# In[23]:


# #Filtering:
f=df['owner']=="Test Drive Car" 
df=df.drop(index=df[f].index)  
df=df.reset_index(drop=True)
print(df['owner'].value_counts())


# In[24]:


df


# In[25]:


# transmission:
print(df['transmission'].value_counts())


# In[26]:


# seller type
print(df['seller_type'].value_counts())   
f=df['seller_type']=='Individual'       
data=df.loc[f,'selling_price']            
plt.violinplot(data)   
plt.title('individual')
plt.show()


# In[27]:


print(df['seller_type'].value_counts())
f=df['seller_type']=='Dealer'
data=df.loc[f,'selling_price']
plt.violinplot(data)   
plt.title('Dealer')
plt.show()


# In[28]:


print(df['seller_type'].value_counts())
f=df['seller_type']=='Trustmark Dealer'
data=df.loc[f,'selling_price']
plt.violinplot(data)   
plt.title('Trustmark')
plt.show()


# In[30]:


#fuel
print(df['fuel'].value_counts())


# In[31]:


f=df['fuel']=='Diesel' 
data=df.loc[f,'selling_price'] 
plt.violinplot(data)   
plt.title('Diesel')
plt.show()


# In[32]:


f=df['fuel']=='Petrol'
data=df.loc[f,'selling_price']
plt.violinplot(data)   
plt.title('Petrol')
plt.show()


# In[33]:


f=df['fuel']=='CNG'
data=df.loc[f,'selling_price']
plt.violinplot(data)   
plt.title('CNG')
plt.show()


# In[34]:


f=df['fuel']=='LPG'
data=df.loc[f,'selling_price']
plt.violinplot(data)   
plt.title('LPG')
plt.show()


# In[35]:


# merge 
df['fuel']=df['fuel'].replace({'Petrol':0,'Diesel':0,'CNG':1,'LPG':1}) 


# In[36]:


plt.hist(df['km_driven'])  # normal distribution
plt.show()


# In[37]:


plt.scatter(df['selling_price'],df['km_driven'])  # normal distribution
plt.show()



# In[38]:


def convert(x):                                     
    temp=x.split()
    return temp[0]      

df['name']=df['name'].apply(convert) 




# In[39]:


print(df['name'].value_counts())

print(len(df['name'].value_counts()))  


# In[40]:


#Group name of data
avg_of_brand=df.groupby("name")['selling_price'].mean()
avg_of_brand=avg_of_brand.sort_values(ascending=False) 

print(avg_of_brand)




# In[41]:


def changer(x):
    if x in avg_of_brand[0:10]:      
        return 0
    elif x in avg_of_brand[10:20]:
        return 1
    else:
        return 2 
    
    
df['name']=df['name'].apply(changer)



# In[42]:


df.isnull().sum()


# In[43]:


# divide the colums in numeric and categorical

numeric=df[['max_power','engine','mileage','km_driven','selling_price','year']]
categorical=df[['name','fuel','seller_type','transmission','owner','seats']]



# In[45]:


#  pearson correlation method we found the correlation of two columns
print(numeric.corr()) 


# In[46]:


sn.heatmap(numeric.corr())  # if color is light means corelation jada hai ( jiska 0.00 ya relation kam hoga usko hi remove karna hai)
plt.show() # if color is dark means corelation kam hoga


# In[47]:


from sklearn.preprocessing import LabelEncoder
encoder1=LabelEncoder()
categorical['owner']=encoder1.fit_transform(categorical["owner"])



# In[48]:


encoder2=LabelEncoder()
categorical['transmission']=encoder2.fit_transform(categorical['transmission'])



# In[49]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encode",OneHotEncoder(drop="first",sparse=False),[2])],
                      remainder="passthrough")
categorical=ct.fit_transform(categorical)            


# In[50]:


# ANOVA TEST: use for find the important colums of numeric and numeric

from sklearn.feature_selection import SelectKBest,f_classif 
kbest=SelectKBest(f_classif,k=categorical.shape[1],)
kbest.fit_transform(categorical,numeric['selling_price'])
print("F values\n",kbest.scores_)




# In[51]:


# outlier detection:


# In[52]:


plt.hist(numeric['max_power'])
plt.title('Max power')
plt.show()


# In[53]:


plt.hist(numeric['engine'])
plt.title("Engine")
plt.show()


# In[54]:


plt.hist(numeric['mileage'])
plt.title("Mileage")
plt.show()


# In[55]:


plt.hist(numeric['km_driven'])
plt.title("Km_Driven")
plt.show()


# In[56]:


def zscore(x):
    mean=np.mean(x)  
    std=np.std(x)
    z=(x-mean)/std
    z=z.abs() 
    return x[z>3]

outliers1=zscore(numeric['mileage']) 
print(outliers1)    
outliers2=zscore(numeric['max_power'])
print(outliers2)
outliers3=zscore(numeric['km_driven'])
print(outliers3)


    
    


# In[57]:


# feature scaling
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
data=minmax.fit_transform(numeric[['engine','selling_price']])


# In[58]:


from sklearn.neighbors import NearestNeighbors 
neighbors=NearestNeighbors(n_neighbors=2)
n=neighbors.fit(data) 


# In[59]:


# give distance and index
dis,ind=n.kneighbors(data)


# In[60]:


dis=dis[:,-1] 
dis.sort()



# In[61]:


plt.plot(dis)
plt.show()


# In[62]:


# use DBSCAN algoridhm
from sklearn.cluster import DBSCAN
model=DBSCAN(eps=0.04,min_samples=2)
model.fit(data)  




# In[63]:


print(model.labels_) 


# In[64]:


# check the outliers:-- from engine and selling price 

outliers =pd.DataFrame(data[model.labels_==-1])
print(outliers)
outliers4=outliers[0]  
outliers5=outliers[1]   
print(outliers4)
print(outliers5)



# In[65]:


# # remove the outliers:

numeric=numeric[~numeric['mileage'].isin(outliers1)]  
numeric=numeric[~numeric['max_power'].isin(outliers2)] 
numeric=numeric[~numeric['km_driven'].isin(outliers3)]
numeric=numeric[~numeric['engine'].isin(outliers4)]
numeric=numeric[~numeric['selling_price'].isin(outliers5)]




# In[66]:


# removing the corresponding rows of categorical columns:
categorical=pd.DataFrame(categorical)      
categorical=categorical[categorical.index.isin(numeric.index)]


# In[67]:


print(numeric.head())
print(categorical.head())


# In[68]:


# join categorical and numeric
df=pd.concat([numeric,categorical],axis=1,join='inner') 


# In[69]:


df.head()


# In[70]:


df.columns


# In[71]:


# slicing dependent and independent:

x=df.drop(columns=['selling_price'],axis=1)
x


# In[72]:


y=df['selling_price']
y


# In[73]:


x.columns=x.columns.astype(str) 


# In[74]:


# standard scaling:
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)                      
x


# In[75]:


# splitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) # random state use hogi taki value bar bar change na ho
print(x_train)
print(y_train)
print(x_test)
print(y_test)


# In[76]:


# linear regression
# training( use machine learning algoridhim)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()                              
regressor.fit(x_train,y_train) 


# In[77]:


y_pred=regressor.predict(x_test)


# In[78]:


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))     


# In[79]:


# svm
from sklearn.svm import SVC   
classifier=SVC(C=5000,kernel="poly") 
classifier.fit(x_train,y_train)


# In[80]:


y_pred=classifier.predict(x_test)
y_pred


# In[81]:


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)) 


# In[82]:


# decision Tree
from sklearn.tree import DecisionTreeClassifier  
classifier=DecisionTreeClassifier(max_depth=10,min_samples_split=20) 
classifier.fit(x_train,y_train)


# In[83]:


y_pred=classifier.predict(x_test)


# In[84]:


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

