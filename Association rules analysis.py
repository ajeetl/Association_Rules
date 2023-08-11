#!/usr/bin/env python
# coding: utf-8

# ## Association Rules Analysis

# In[1]:

# Importing the important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlxtend.preprocessing
import mlxtend.frequent_patterns


# In[2]:


# Loading the data
data_ass = pd.read_csv('market_basket.csv')

data_ass.head()


# In[3]:


# Most frequent items
data_ass.Item.value_counts()[:10]


# ## Creating transactional format

# In[4]:


# Creating transactional format for afternoon

data = data_ass[data_ass.period_day == 'afternoon']

grocery_list = data.groupby(['Transaction'])['Item'].apply(list).values.tolist()

encoder = mlxtend.preprocessing.TransactionEncoder().fit(grocery_list)

encoded_data = encoder.transform(grocery_list)

grocery_trans = pd.DataFrame(encoded_data, columns = encoder.columns_)

# Creating transactional format for morning

data_1 = data_ass[data_ass.period_day == 'morning']

grocery_list_1 = data_1.groupby(['Transaction'])['Item'].apply(list).values.tolist()

encoder_1 = mlxtend.preprocessing.TransactionEncoder().fit(grocery_list_1)

encoded_data_1 = encoder_1.transform(grocery_list_1)

grocery_trans_1 = pd.DataFrame(encoded_data_1, columns = encoder_1.columns_)


# In[5]:


# Most frequent products in afternoon
print('Afternoon')
print(grocery_trans.sum().sort_values(ascending = False)[:10])

# Most frequent products in morning
print('Morning')
print(grocery_trans_1.sum().sort_values(ascending = False)[:10])


# In[6]:


# Itemsets for afternoon
frequent_itemsets = mlxtend.frequent_patterns.apriori(grocery_trans, min_support = 0.001, max_len = 4, use_colnames = True)
frequent_itemsets.shape[0]


# In[7]:


# Itemsets for morning
frequent_itemsets_1 = mlxtend.frequent_patterns.apriori(grocery_trans_1, min_support = 0.001, max_len = 4, use_colnames = True)
frequent_itemsets_1.shape[0]


# In[8]:


# Rules for afternoon
rules = mlxtend.frequent_patterns.association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)
rules.head()


# In[9]:


# Rules for morning
rules_1 = mlxtend.frequent_patterns.association_rules(frequent_itemsets_1, metric = "confidence", min_threshold = 0.6)
rules_1.head()


# ## Associating Egg in the morning

# In[10]:


# Egg in the morning: Bread
selection_1 = rules_1['antecedents'].apply(lambda x: 'Eggs' in x)
print(rules_1[selection_1])


# ## Associating coke (cocacola) and juice in the afternoon

# In[11]:


# Coke and Juice in the afternoon: Sandwich
selection_2 = rules['antecedents'].apply(lambda x: 'Coke' in x and 'Juice' in x)
print(rules[selection_2])


# ## Preference of the toast in the morning or in the afternoon

# In[12]:


# Toast in the morning or in the afternoon: Coffee
selection = rules['antecedents'].apply(lambda x: 'Toast' in x)
selection_1 = rules_1['antecedents'].apply(lambda x: 'Toast' in x)
print(rules[selection])
print(rules_1[selection_1])

