
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# In[2]:


plot(arange(5))


# In[3]:


import pandas as  pd


# In[6]:


df= pd.read_csv("C:\Users\Ido&Talya\Documents\studies\ProgrammingXalone\dataXscienceXprojects\projectXloanXprediction\train.csv")


# In[7]:


df= pd.read_csv("C:\Users\Ido&Talya\Documents\studies\ProgrammingXalone\dataXscienceXprojects\projectXloanXprediction\train.csv")


# In[8]:


df= pd.read_csv("C:\Users\Ido&Talya\Documents\studies\Programming_alone\data_science_projects\project_loan_prediction")


# In[9]:


df= pd.read_csv(r"C:\Users\Ido&Talya\Documents\studies\Programming_alone\data_science_projects\project_loan_prediction\train.csv")


# In[10]:


df.head(10)


# In[13]:


df.describe()


# In[14]:


df['ApplicantIncome'].hist(bins=50)


# In[15]:


df.boxplot(column='ApplicantIncome')


# In[16]:


df.boxplot(column='ApplicantIncome', by = 'Education')


# In[17]:


df['LoanAmount'].hist(bins=50)


# In[18]:


df.apply(lambda x: sum(x.isnull()),axis=0) 


# In[20]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)


# In[21]:


df.apply(lambda x: sum(x.isnull()),axis=0) 


# In[22]:


table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


# In[23]:


df.apply(lambda x: sum(x.isnull()),axis=0) 


# In[24]:


df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)


# In[25]:


df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 


# In[26]:


df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 

