#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/Navjotkhatri/CARDIOVASCULAR-RISK-PREDICTION/main/data_cardiovascular_risk.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df.dtypes


# In[10]:


#df.drop(columns=['weight_(kg)','height_(cm)'],inplace=True)


# In[12]:


for i in df.columns.tolist():
  print("No. of unique values in ", i , "is" , df[i].nunique(), ".")


# In[13]:


# Separating the categorical and continous variable and storing them
categorical_variable=[]
continous_variable=[]

for i in df.columns:
  if i == 'id':
    pass
  elif df[i].nunique() <5:
    categorical_variable.append(i)
  elif df[i].nunique() >= 5:
    continous_variable.append(i)

print(categorical_variable)
print(continous_variable)


# In[14]:


# Summing null values
print('Missing Data Count')
df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False)


# In[15]:


print('Missing Data Percentage')
print(round(df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False)/len(df)*100,2))
     
    


# In[16]:


null_column_list= ['glucose','education','BPMeds','totChol','cigsPerDay','BMI','heartRate']
# plotting box plot
plt.figure(figsize=(10,8))
df[null_column_list].boxplot()


# In[17]:


colors = sns.color_palette("rocket", len(null_column_list))


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 8))


axes = axes.flatten()


for i, column in enumerate(null_column_list):
    ax = axes[i]

    sns.distplot(df[column], ax=ax, color=colors[i])
    ax.set_title(column)

for j in range(len(null_column_list), len(axes)):
    axes[j].remove()

plt.show()


# In[18]:


df.fillna({'glucose': df['glucose'].median(),
           'education': df['education'].mode()[0],
           'BPMeds': df['BPMeds'].mode()[0],
           'totChol': df['totChol'].median(),
           'cigsPerDay': df['cigsPerDay'].median(),
           'BMI': df['BMI'].median(),
           'heartRate': df['heartRate'].median()}, inplace=True)


# In[19]:


fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(x="sex", y="age", hue="TenYearCHD", data= df, ax=ax)
ax.set_title("Age Distribution of Patients by Sex and CHD Risk Level")
ax.set_xlabel("Sex")
ax.set_ylabel("Age")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["No Risk", "At Risk"], loc="best")
plt.show()
     


# In[20]:


plt.figure(figsize=(10,8))
sns.countplot(x='sex', hue='TenYearCHD', data= df)
plt.title('Frequency of CHD cases by gender')
plt.legend(['No Risk', 'At Risk'])
plt.show()


# In[21]:


plt.figure(figsize=(10,8))
sns.countplot(x='is_smoking', hue='TenYearCHD', data= df)
plt.title('A Comparison of Smokers and Non-Smokers')
plt.legend(['No Risk', 'At Risk'])
plt.show()


# In[22]:


plt.figure(figsize=(10,8))
sns.countplot(x= df['cigsPerDay'],hue= df['TenYearCHD'])
plt.title('How much smoking affect CHD?')
plt.legend(['No Risk','At Risk'])
plt.show()


# In[23]:


ct = pd.crosstab(df['BPMeds'], df['TenYearCHD'], normalize='index')
ct.plot(kind='bar', stacked=True, figsize=(10, 8))
plt.title('Relationship between BP Medication and CHD Risk')
plt.xlabel('BP Medication')
plt.xticks(rotation=0)
plt.ylabel('Proportion')
plt.legend(['No Risk', 'At Risk'])
plt.show()
     


# In[24]:


plt.figure(figsize=(10,8))
sns.countplot(x=df['prevalentStroke'], hue=df['TenYearCHD'])
plt.title('Are people who had a stroke earlier more prone to CHD?')
plt.legend(['No Risk', 'At Risk'], loc='best')
plt.show()
     


# In[25]:


plt.figure(figsize=(10,8))
sns.countplot(x=df['prevalentHyp'], hue=df['TenYearCHD'])
plt.title('Are hypertensive patients at more risk of CHD?')
plt.legend(title='CHD Risk', labels=['No Risk', 'At Risk'])
plt.show()
     


# In[26]:


plt.figure(figsize=(10,8))
sns.barplot(x=df['diabetes'], y=df['TenYearCHD'], hue=df['TenYearCHD'], estimator=lambda x: len(x) / len(df) * 100)
plt.title('Proportion of patients with and without diabetes at CHD risk')
plt.xlabel('Diabetes')
plt.ylabel('Percentage')
plt.legend(title='CHD Risk', labels=['No Risk', 'At Risk'])
plt.show()
     


# In[27]:


plt.figure(figsize=(10,8))
sns.boxplot(x='TenYearCHD', y='totChol', data=df)
plt.title('Total Cholesterol Levels and CHD')
plt.xlabel('TenYearCHD')
plt.ylabel('Total Cholesterol Levels')
plt.legend(['No Risk', 'At Risk'])
plt.show()
     


# In[28]:


cols = ['glucose', 'sysBP', 'diaBP', 'TenYearCHD']

# create the scatter plot matrix
plt.figure(figsize=(15,10))
sns.pairplot(df[cols], hue='TenYearCHD', markers=['o', 's'])
plt.show()


# In[29]:


plt.figure(figsize=(10,8))
cols = ['sex', 'cigsPerDay', 'TenYearCHD']
sns.scatterplot(x='cigsPerDay', y='TenYearCHD', hue='sex', data=df)
plt.show()


# In[30]:


plt.figure(figsize=(10,8))
sns.violinplot(x='prevalentStroke',y="age",data=df, hue='sex', split='True', palette='rainbow')
plt.show()
     


# In[31]:


plt.figure(figsize=(10,8))
sns.scatterplot(x='cigsPerDay', y='sysBP', hue='prevalentHyp', data=df)
plt.title('Relationship between Systolic Blood Pressure and Cigarettes Smoked per Day, by Hypertension Status')
plt.xlabel('Cigarettes Smoked per Day')
plt.ylabel('Systolic Blood Pressure')
plt.show()


# In[32]:


# Correlation Heatmap visualization code
plt.figure(figsize=(12,12))
correlation = df.corr()
sns.heatmap((correlation), annot=True, cmap=sns.color_palette("mako", as_cmap=True))


# In[33]:


plt.figure(figsize=(15,10))
sns.pairplot(df[continous_variable])
plt.show()


# In[34]:


import scipy.stats as stats

# Separate the dataset into two groups based on CHD status
chd = df[df['TenYearCHD'] == 1] # Patients with CHD
no_chd = df[df['TenYearCHD'] == 0] # Patients without CHD

# Perform a two-sample t-test to compare the mean total cholesterol levels of the two groups
t_stat, p_val = stats.ttest_ind(chd['totChol'], no_chd['totChol'], equal_var=False)

# Print the calculated t-statistic and p-value
print('t_stat=%.3f, p_val=%.3f' % (t_stat, p_val))

# Determine if the null hypothesis should be rejected based on the p-value
if p_val < 0.05:
    print('Reject the null hypothesis')
else:
    print('Fail to reject the null hypothesis')

# Print the p-value
print('p-value:', p_val)


# In[35]:


# Perform Statistical Test to obtain P-Value
diabetic = df[df['diabetes'] == 1]
non_diabetic = df[df['diabetes'] == 0]

# Perform a two-sample t-test to compare the mean TenYearCHD rates of the two groups
t_stat, p_val = stats.ttest_ind(diabetic['TenYearCHD'], non_diabetic['TenYearCHD'], equal_var=False)

print('t_stat=%.3f, p_val=%.3f' % (t_stat, p_val))
if p_val > 0.05:
    print('Accept Null Hypothesis')
else:
    print('Reject Null Hypothesis')

# Print the p-value
print('p-value:', p_val)


# In[36]:


# Perform Statistical Test to obtain P-Value
import statsmodels.stats.proportion as smp

above_50 = df[df['age'] > 50]
below_50 = df[df['age'] <= 50]

# Calculate the proportion of patients with TenYearCHD in each group
prop_above_50 = above_50['TenYearCHD'].mean()
prop_below_50 = below_50['TenYearCHD'].mean()

# Perform a one-tailed z-test to compare the proportions of the two groups
z_score, p_val = smp.proportions_ztest([prop_above_50 * len(above_50), prop_below_50 * len(below_50)], [len(above_50), len(below_50)], alternative='larger')

print('z_score=%.3f, p_val=%.3f' % (z_score, p_val))

if p_val < 0.05:
    print('Reject Null Hypothesis')
else:
    print('Accept Null Hypothesis')

# Print the p-value
print('p-value:', p_val)


# In[37]:


# Handling Missing Values & Missing Value Imputation
df.isnull().sum()


# In[38]:


# Handling Outliers & Outlier treatments
fig, axes = plt.subplots(2, 4, figsize=(15, 10))
axes = axes.flatten()
for ax, col in zip(axes, continous_variable):
    sns.boxplot(df[col], ax=ax)
    ax.set_title(col.title(), weight='bold')
plt.tight_layout()


# In[39]:


df[continous_variable] = np.log(df[continous_variable] +1 )


# In[40]:


fig, axes = plt.subplots(2, 4, figsize=(15, 10))
axes = axes.flatten()
for ax, col in zip(axes, continous_variable):
    sns.boxplot(df[col], ax=ax)
    ax.set_title(col.title(), weight='bold')
plt.tight_layout()
     


# In[41]:


df.head()


# In[42]:


df['sex'] = pd.get_dummies(df['sex'], drop_first=True)
df['is_smoking'] = pd.get_dummies(df['is_smoking'], drop_first=True)
     


# In[43]:


df.head()


# In[44]:


df.info()


# In[45]:


# Manipulate Features to minimize feature correlation and create new features
df.head()


# In[46]:


df['pulsePressure'] = df['sysBP'] - df['diaBP']


# In[47]:


df.head()


# In[48]:


# Select your features wisely to avoid overfitting
for col in df.describe().columns.tolist():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()
    feature = df[col]
    label = df['TenYearCHD']
    correlation = feature.corr(label)
    sns.scatterplot(x=feature, y=label, color="gray")
    plt.xlabel(col)
    plt.ylabel('TenYearCHD')
    ax.set_title('TenYearCHD vs ' + col + '- correlation: ' + str(correlation))
    z = np.polyfit(df[col], df['TenYearCHD'], 1)
    y_hat = np.poly1d(z)(df[col])
    plt.plot(df[col], y_hat, "r--", lw=1)
    plt.show()


# In[49]:


f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(abs(round(df.corr(),3)), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
     


# In[50]:


plt.figure(figsize=(10,8))
print("Before Applying Transformation")
sns.distplot(df['pulsePressure'])
plt.title('Distribution of pulsePressure')


# In[ ]:




