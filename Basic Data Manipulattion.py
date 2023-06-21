#!/usr/bin/env python
# coding: utf-8

# # Basic Exercises on Data Importing - Understanding - Manipulating - Analysis - Visualization

# ## Section-1: The pupose of the below exercises (1-7) is to create dictionary and convert into dataframes, how to diplay etc...
# ## The below exercises required to create data 

# ### 1. Import the necessary libraries (pandas, numpy, datetime, re etc)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import re

# set the graphs to show in the jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# set seabor graphs to a better style
sns.set(style="ticks")


# ### 2. Run the below line of code to create a dictionary and this will be used for below exercises

# In[2]:


raw_data = {"name": ['Bulbasaur', 'Charmander','Squirtle','Caterpie'],
            "evolution": ['Ivysaur','Charmeleon','Wartortle','Metapod'],
            "type": ['grass', 'fire', 'water', 'bug'],
            "hp": [45, 39, 44, 45],
            "pokedex": ['yes', 'no','yes','no']                        
            }


# ### 3. Assign it to a object called pokemon and it should be a pandas DataFrame

# In[3]:


pokemon=pd.DataFrame(raw_data)
pokemon


# ### 4. If the DataFrame columns are in alphabetical order, change the order of the columns as name, type, hp, evolution, pokedex

# In[4]:


pokemon=pokemon[["name","type","hp","evolution","pokedex"]]
pokemon


# ### 5. Add another column called place, and insert places (lakes, parks, hills, forest etc) of your choice.

# In[5]:


pokemon["place"]=["lakes","parks","hills","forest"]
pokemon


# ### 6. Display the data type of each column

# In[6]:


pokemon.dtypes


# ### 7. Display the info of dataframe

# In[7]:


pokemon.info()


# ## Section-2: The pupose of the below exercise (8-20) is to understand deleting data with pandas.
# ## The below exercises required to use wine.data

# ### 8. Import the dataset *wine.txt* from the folder and assign it to a object called wine
# 
# Please note that the original data text file doesn't contain any header. Please ensure that when you import the data, you should use a suitable argument so as to avoid data getting imported as header.

# In[8]:


wine=pd.read_csv(r"C:\Users\lenovo\Desktop\wine.txt",header=None)
wine


# ### 9. Delete the first, fourth, seventh, nineth, eleventh, thirteenth and fourteenth columns

# In[9]:


wine.drop(columns=[0,3,6,8,10,12,13],inplace=True)
wine


# ### 10. Assign the columns as below:
# 
# The attributes are (dontated by Riccardo Leardi, riclea '@' anchem.unige.it):  
# 1) alcohol  
# 2) malic_acid  
# 3) alcalinity_of_ash  
# 4) magnesium  
# 5) flavanoids  
# 6) proanthocyanins  
# 7) hue 

# In[10]:


wine.columns=["alcohol","malic_acid","alcalininity_of_ash","magnesium","flavanoids","proanthocyanis","hue"]
wine


# ### 11. Set the values of the first 3 values from alcohol column as NaN

# In[11]:


wine.alcohol.iloc[0:3]=np.nan
wine


# ### 12. Now set the value of the rows 3 and 4 of magnesium as NaN

# In[12]:


wine.magnesium.iloc[2:4]=np.nan
wine


# ### 13. Fill the value of NaN with the number 10 in alcohol and 100 in magnesium

# In[13]:


wine.alcohol=wine.alcohol.fillna(10)
wine.magnesium=wine.magnesium.fillna(100)
wine


# ### 14. Count the number of missing values in all columns.

# In[14]:


wine.isna().sum()


# ### 15.  Create an array of 10 random numbers up until 10 and save it.

# In[15]:


array=np.random.randint(0,11,10)
array


# ### 16.  Set the rows corresponding to the random numbers to NaN in the column *alcohol*

# In[16]:


wine.alcohol.iloc[array]=np.nan
wine.head(20)


# ### 17.  How many missing values do we have now?

# In[17]:


wine.isna().sum()


# ### 18. Print only the non-null values in alcohol

# In[18]:


wine[["alcohol"]].dropna()


# ### 19. Delete the rows that contain missing values

# In[19]:


wine=wine.dropna()
wine


# ### 20.  Reset the index, so it starts with 0 again

# In[20]:


wine.reset_index(drop=True)


# ## Section-3: The pupose of the below exercise (21-27) is to understand ***filtering & sorting*** data from dataframe.
# ## The below exercises required to use chipotle.tsv

# This time we are going to pull data directly from the internet.  
# Import the dataset directly from this link (https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv) and create dataframe called chipo

# In[21]:


chipo=pd.read_table(r"C:\Users\lenovo\Desktop\chipotle.tsv")
chipo
k=chipo
chipo.item_price= chipo['item_price'].map(lambda x: float(x[1:-1]))
chipo


# ### 21. How many products cost more than $10.00? 
# 
# Use `str` attribute to remove the $ sign and convert the column to proper numeric type data before filtering.
# 

# In[22]:


cnt = chipo.item_price[chipo.item_price > 10].count()
cnt


# ### 22. Print the Chipo Dataframe & info about data frame

# In[23]:


chipo.head()


# In[24]:


chipo.info()


# ### 23. What is the price of each item? 
# - Delete the duplicates in item_name and quantity
# - Print a data frame with only two columns `item_name` and `item_price`
# - Sort the values from the most to less expensive

# In[25]:


chipo[['item_name','item_price']].sort_values(by= 'item_price', ascending= False).drop_duplicates()


# ### 24. Sort by the name of the item

# In[26]:


chipo.sort_values(by='item_name')


# ### 25. What was the quantity of the most expensive item ordered?

# In[27]:


chipo[chipo.item_price == chipo.item_price.max()].iat[0,1]


# ### 26. How many times were a Veggie Salad Bowl ordered?

# In[28]:


chipo.item_name[chipo.item_name == 'Veggie Salad Bowl'].count()


# ### 27. How many times people orderd more than one Canned Soda?

# In[29]:


chipo.item_name[chipo.item_name == 'Canned Soda'][chipo.quantity > 1].count()


# ## Section-4: The purpose of the below exercises is to understand how to perform aggregations of data frame
# ## The below exercises (28-33) required to use occupation.csv

# ###  28. Import the dataset occupation.csv and assign object as users  

# In[30]:


df=pd.read_csv(r"C:\Users\lenovo\occupation.csv",sep="|")
df


# ### 29. Discover what is the mean age per occupation

# In[31]:


df.groupby(["occupation"])[["age"]].mean()


# ### 30. Discover the Male ratio per occupation and sort it from the most to the least.
# 
# Use numpy.where() to encode gender column.

# In[32]:


Q30=(pd.crosstab(['occupation'],df['gender'])['M']/df.occupation.value_counts().sort_index()).reset_index()
Q30.rename(columns={0:'Male ratio'}).sort_values(by='Male ratio',ascending=False).reset_index(drop=True)


# ### 31. For each occupation, calculate the minimum and maximum ages

# In[33]:


df.groupby('occupation')['age'].agg(['min','max'])


# ### 32. For each combination of occupation and gender, calculate the mean age

# In[34]:


df.groupby(['occupation','gender']).age.mean().reset_index()


# ### 33.  For each occupation present the percentage of women and men

# In[35]:


df=pd.DataFrame(df)
(df.groupby(['occupation','gender'])['gender'].count())/(df.groupby(['occupation'])['gender'].count())*100


# ## Section-6: The purpose of the below exercises is to understand how to use lambda-apply-functions
# ## The below exercises (34-41) required to use student-mat.csv and student-por.csv files 

# ### 34. Import the datasets *student-mat* and *student-por* and append them and assigned object as df

# In[36]:


student_m=pd.read_csv(r"C:\Users\lenovo\Desktop\student-mat.csv")
student_p=pd.read_csv(r"C:\Users\lenovo\Desktop\student-por.csv")
df=student_m.append(student_p).reset_index(drop=True)
df


# ### 35. For the purpose of this exercise slice the dataframe from 'school' until the 'guardian' column

# In[37]:


df=df.loc[:,'school':'guardian']
df


# ### 36. Create a lambda function that captalize strings (example: if we give at_home as input function and should give At_home as output.

# In[38]:


df.select_dtypes('object').applymap(lambda x: x.capitalize())


# ### 37. Capitalize both Mjob and Fjob variables using above lamdba function

# In[39]:


df[['Mjob','Fjob']].applymap(lambda x: x.capitalize())


# ### 38. Print the last elements of the data set. (Last few records)

# In[40]:


df.tail()


# ### 39. Did you notice the original dataframe is still lowercase? Why is that? Fix it and captalize Mjob and Fjob.

# In[41]:


df[['Mjob','Fjob']]=df[['Mjob','Fjob']].applymap(lambda x: x.capitalize())


# ### 40. Create a function called majority that return a boolean value to a new column called legal_drinker

# In[42]:


majority=lambda x: True if x>=18 else False

df['legal_drinker']=df.age.apply(majority)
df.head(10)


# ### 41. Multiply every number of the dataset by 10. 

# In[43]:


y=df.select_dtypes(include='number')*10
df[y.columns]=y
df


# ## Section-6: The purpose of the below exercises is to understand how to perform simple joins
# ## The below exercises (42-48) required to use cars1.csv and cars2.csv files 

# ### 42. Import the datasets cars1.csv and cars2.csv and assign names as cars1 and cars2

# In[44]:


cars1=pd.read_csv(r"C:\Users\lenovo\Desktop\cars1.csv")
cars1
cars2=pd.read_csv(r'C:\Users\lenovo\Desktop\cars2.csv')
cars2


#    ### 43. Print the information to cars1 by applying below functions 
#    hint: Use different functions/methods like type(), head(), tail(), columns(), info(), dtypes(), index(), shape(), count(), size(), ndim(), axes(), describe(), memory_usage(), sort_values(), value_counts()
#    Also create profile report using pandas_profiling.Profile_Report

# In[45]:


type(cars1)


# In[46]:


cars1.head(5)


# In[47]:


cars1.tail(5)


# In[48]:


cars1.columns


# In[49]:


cars1.info()


# In[50]:


cars1.shape


# In[51]:


cars1.count()


# In[52]:


cars1.size


# In[53]:


cars1.ndim


# In[54]:


cars1.axes


# In[55]:


cars1.describe()


# In[56]:


cars1.memory_usage()


# In[57]:


cars1.sort_values(by='car')


# In[58]:


cars1.car.value_counts()


# ### 44. It seems our first dataset has some unnamed blank columns, fix cars1

# In[59]:


cars1=cars1.dropna(axis=1)
cars1


# ### 45. What is the number of observations in each dataset?

# In[60]:


print('the number of observations in cars1 is',cars1.index.size)
print('the number of observations in cars2 is',cars2.index.size)


# ### 46. Join cars1 and cars2 into a single DataFrame called cars

# In[61]:


cars=cars1.append(cars2)
cars


# ### 47. There is a column missing, called owners. Create a random number Series from 15,000 to 73,000.

# In[62]:


random=np.random.randint(15000,73000,cars.index.size)
cars


# ### 48. Add the column owners to cars

# In[63]:


cars['owners']=random
cars


# ## Section-7: The purpose of the below exercises is to understand how to perform date time operations

# ### 49. Write a Python script to display the
# - a. Current date and time
# - b. Current year
# - c. Month of year
# - d. Week number of the year
# - e. Weekday of the week
# - f. Day of year
# - g. Day of the month
# - h. Day of week

# In[64]:


today=dt.datetime.now()
print('current date and time  : ',today)
print('current year           : ',today.year)
print('month of year          : ',today.month)
print('week number of the year: ',today.isocalendar()[1])
print('weekday of the week    : ',today.strftime('%A'))
print('Day of year            : ',today.timetuple().tm_yday)
print('Day of month           : ',today.day)
print('Day of week            : ',today.weekday())


# ### 50. Write a Python program to convert a string to datetime.
# Sample String : Jul 1 2014 2:43PM 
# 
# Expected Output : 2014-07-01 14:43:00

# In[65]:


from datetime import datetime
date_object=datetime.strptime('Jul 1 2014 2:43PM', '%b %d %Y %I:%M%p')
print(date_object)


# ### 51. Write a Python program to subtract five days from current date.
# 
# Current Date : 2015-06-22
# 
# 5 days before Current Date : 2015-06-17

# In[66]:


from datetime import date,timedelta
dt=date.today()-timedelta(5)
print('Current Date :',date.today())
print('5 days before Current Date :',dt)


# ### 52. Write a Python program to convert unix timestamp string to readable date.
# 
# Sample Unix timestamp string : 1284105682
#     
# Expected Output : 2010-09-10 13:31:22

# In[67]:


import datetime
print(datetime.datetime.fromtimestamp(int(1284105682)))


# ### 53. Convert the below Series to pandas datetime : 
# 
# DoB = pd.Series(["07Sep59","01Jan55","15Dec47","11Jul42"])
# 
# Make sure that the year is 19XX not 20XX

# In[68]:


DoB = pd.Series(["07Sep59","01Jan55","15Dec47","11Jul42"])
pd.to_datetime(DoB)-pd.DateOffset(years=100)


# ### 54. Write a Python program to get days between two dates. 

# In[69]:


x=pd.to_datetime('2022-2-15')
y=pd.to_datetime('2022-3-16')
y-x


# ### 55. Convert the below date to datetime and then change its display format using the .dt module
# 
# Date = "15Dec1989"
# 
# Result : "Friday, 15 Dec 98"

# In[70]:


pd.to_datetime("15Dec1989").strftime('%A, %d %b %y')


# ## The below exercises (56-66) required to use wind.data file 

# ### About wind.data:
# 
# The data have been modified to contain some missing values, identified by NaN.  
# 
# 1. The data in 'wind.data' has the following format:
"""
Yr Mo Dy   RPT   VAL   ROS   KIL   SHA   BIR   DUB   CLA   MUL   CLO   BEL   MAL
61  1  1 15.04 14.96 13.17  9.29   NaN  9.87 13.67 10.25 10.83 12.58 18.50 15.04
61  1  2 14.71   NaN 10.83  6.50 12.62  7.67 11.50 10.04  9.79  9.67 17.54 13.83
61  1  3 18.50 16.88 12.33 10.13 11.17  6.17 11.25   NaN  8.50  7.67 12.75 12.71
"""
The first three columns are year, month and day.  The remaining 12 columns are average windspeeds in knots at 12 locations in Ireland on that day. 
# ### 56. Import the dataset wind.data and assign it to a variable called data and replace the first 3 columns by a proper date time index

# In[71]:


data=pd.read_csv(r"C:\Users\lenovo\wind.data")
data["Date"] = pd.to_datetime(data[["Yr","Mo","Dy"]].astype(str).agg('-'.join, axis=1))
data = data.drop(columns=["Yr","Mo","Dy"])
data


# ### 57. Year 2061 is seemingly imporoper. Convert every year which are < 70 to 19XX instead of 20XX.

# In[72]:


data["Date"] = np.where(pd.DatetimeIndex(data["Date"]).year < 2000,data.Date,data.Date - pd.offsets.DateOffset(years=100))
data


# ### 58. Set the right dates as the index. Pay attention at the data type, it should be datetime64[ns].

# In[73]:


Data = data.set_index("Date")
Data.index.astype("datetime64[ns]")


# ### 59. Compute how many values are missing for each location over the entire record.  
# #### They should be ignored in all calculations below. 

# In[74]:


data.isna().sum()


# ### 60. Compute how many non-missing values there are in total.

# In[75]:


x=Data.count()
print("Total Non-missing values are :",x.sum())


# ### 61. Calculate the mean windspeeds over all the locations and all the times.
# #### A single number for the entire dataset.

# In[76]:


y = Data.mean()
y.mean()


# ### 62. Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and standard deviations of the windspeeds at each location over all the days 
# 
# #### A different set of numbers for each location.

# In[77]:


loc_stats=data.describe().loc[['min','max','mean','std'],:]
loc_stats


# ### 63. Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at each day.
# 
# #### A different set of numbers for each day.

# In[78]:


day_stats = pd.DataFrame()

day_stats['min'] = data.min(axis = 1) 
day_stats['max'] = data.max(axis = 1) 
day_stats['mean'] = data.mean(axis = 1) 
day_stats['std'] = data.std(axis = 1) 

day_stats.head()


# ### 64. Find the average windspeed in January for each location.  
# #### Treat January 1961 and January 1962 both as January.

# In[79]:


january_data = Data[Data.index.month == 1]
print ("January windspeeds:")
print (january_data.mean())


# ### 65. Calculate the mean windspeed for each month in the dataset.  
# #### Treat January 1961 and January 1962 as *different* months.
# #### (hint: first find a  way to create an identifier unique for each month.)

# In[80]:


print( "Yearly:\n", Data.resample('A').mean())


# ### 66. Calculate the min, max and mean windspeeds and standard deviations of the windspeeds across all locations for each week (assume that the first week starts on January 2 1961) for the first 52 weeks.

# In[81]:


first_year = Data[Data.index.year == 1961]
stats1 = Data.resample('W').mean().apply(lambda x: x.describe())
print (stats1)


# ## The below exercises (67-70) required to use appl_1980_2014.csv  file

# ### 67. Import the file appl_1980_2014.csv and assign it to a variable called 'apple'

# In[82]:


apple=pd.read_csv(r'C:\Users\lenovo\appl_1980_2014.csv')
apple


# ### 68.  Check out the type of the columns

# In[83]:


apple.dtypes


# ### 69. Transform the Date column as a datetime type

# In[84]:


apple.Date=pd.to_datetime(apple.Date)
apple.dtypes


# ### 70.  Set the date as the index

# In[85]:


apple.set_index('Date',inplace=True)
apple


# ### 71.  Is there any duplicate dates?

# In[86]:


apple.index.duplicated().sum()


# ### 72.  The index is from the most recent date. Sort the data so that the first entry is the oldest date.

# In[87]:


apple=apple.sort_values(by='Date',ascending=True)
apple


# ### 73. Get the last business day of each month

# In[88]:


apple['day']=apple.index.day
apple.groupby([apple.index.year,apple.index.month_name()]).day.last()


# ### 74.  What is the difference in days between the first day and the oldest

# In[89]:


apple.index.max()-apple.index.min()


# ### 75.  How many months in the data we have?

# In[90]:


len(apple.groupby([apple.index.year,apple.index.month]).count())


# ## Section-8: The purpose of the below exercises is to understand how to create basic graphs

# ### 76. Plot the 'Adj Close' value. Set the size of the figure to 13.5 x 9 inches

# In[91]:


plt.figure(figsize=(13.5,9))
plt.hist(apple['Adj Close'],bins=10)
plt.xlabel('Adj Close')
plt.ylabel('Frequency')
plt.show()


# ## The below exercises (77-80) required to use Online_Retail.csv file

# ### 77. Import the dataset from this Online_Retail.csv and assign it to a variable called online_rt

# In[92]:


online_rt=pd.read_csv(r'C:\Users\lenovo\Online_Retail.csv', encoding= 'unicode_escape')
online_rt


# ### 78. Create a barchart with the 10 countries that have the most 'Quantity' ordered except UK

# In[93]:


countries = online_rt.groupby('Country').sum()
countries = countries.sort_values(by = 'Quantity',ascending = False)[1:11]
countries['Quantity'].plot(kind='bar')
plt.xlabel('Countries')
plt.ylabel('Quantity')
plt.title('10 Countries with most orders')


# ### 79.  Exclude negative Quatity entries

# In[94]:


online_rt = online_rt[online_rt.Quantity > 0]
online_rt.head()


# ### 80. Create a scatterplot with the Quantity per UnitPrice by CustomerID for the top 3 Countries
# Hint: First we need to find top-3 countries based on revenue, then create scater plot between Quantity and Unitprice for each country separately
# 

# In[95]:


customers = online_rt.groupby(['CustomerID','Country']).sum()
customers = customers[customers.UnitPrice > 0]
customers['Country'] = customers.index.get_level_values(1)
top_countries =  ['Netherlands', 'EIRE', 'Germany']
customers = customers[customers['Country'].isin(top_countries)]
g = sns.FacetGrid(customers, col="Country")
g.map(plt.scatter, "Quantity", "UnitPrice", alpha=1)
g.add_legend()


# ## The below exercises (81-90) required to use FMCG_Company_Data_2019.csv file

# ### 81. Import the dataset FMCG_Company_Data_2019.csv and assign it to a variable called company_data

# In[96]:


company_data=pd.read_csv(r"C:\Users\lenovo\FMCG_Company_Data_2019.csv")
company_data


# ### 82. Create line chart for Total Revenue of all months with following properties
# - X label name = Month
# - Y label name = Total Revenue

# In[97]:


plt.figure(figsize=(10,6))
plt.plot(company_data.Month,company_data.Total_Revenue,marker='o')
plt.ticklabel_format(style='plain',axis='y')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.show()


# ### 83. Create line chart for Total Units of all months with following properties
# - X label name = Month
# - Y label name = Total Units
# - Line Style dotted and Line-color should be red
# - Show legend at the lower right location.

# In[98]:


plt.figure(figsize=(10,6))
plt.plot(company_data.Month,company_data.Total_Units,label='Total Units for each month',linestyle='dotted',color='r',marker='o')
plt.xlabel('Month')
plt.ylabel('Total Units')
plt.legend(loc='lower right')
plt.show()


# ### 84. Read all product sales data (Facecream, FaceWash, Toothpaste, Soap, Shampo, Moisturizer) and show it  using a multiline plot
# - Display the number of units sold per month for each product using multiline plots. (i.e., Separate Plotline for each product ).

# In[99]:



plt.plot(company_data.Month,company_data.FaceCream,label='FaceCream',marker='o')
plt.plot(company_data.Month,company_data.FaceWash,label='FaceWash',marker='o')
plt.plot(company_data.Month,company_data.ToothPaste,label='Toothpaste',marker='o')
plt.plot(company_data.Month,company_data.Soap,label='Soap',marker='o')
plt.plot(company_data.Month,company_data.Shampo,label='Shampo',marker='o')
plt.plot(company_data.Month,company_data.Moisturizer,label='Moisturizer',marker='o')
plt.xlabel('Month')
plt.ylabel('Number of units sold')
plt.legend(loc='upper left')
plt.show()


# ### 85. Create Bar Chart for soap of all months and Save the chart in folder

# In[100]:


sns.barplot(company_data.Month,company_data.Soap)
plt.ylabel('soap units sold')
plt.savefig('bar chart')
plt.show()


# ### 86. Create Stacked Bar Chart for Soap, Shampo, ToothPaste for each month
# The bar chart should display the number of units sold per month for each product. Add a separate bar for each product in the same chart.

# In[101]:


x=pd.pivot_table(data=company_data,index='Month',values=['Soap','Shampo','ToothPaste']).plot(kind='bar',stacked=True,figsize=(15,6))
for i in x.containers:
    x.bar_label(i,size=15,label_type='center')
plt.legend(bbox_to_anchor=(1.12,1))


# ### 87. Create Histogram for Total Revenue

# In[102]:


plt.hist(company_data.Total_Revenue,bins=20)
plt.xlabel('Total Revenue')
plt.ylabel('frequency')
plt.show


# ### 88. Calculate total sales data (quantity) for 2019 for each product and show it using a Pie chart. Understand percentage contribution from each product

# In[103]:


X=company_data.loc[:,'FaceCream':'Moisturizer'].sum()
plt.pie(X,labels=X.index,autopct='%.2f %%')
plt.title('total sales for each product')
plt.show()


# ### 89. Create line plots for Soap & Facewash of all months in a single plot using Subplot

# In[104]:


f, axes = plt.subplots(1,2,figsize=(20,5))
axes[0].plot(company_data.Month,company_data.Soap,marker='o')
axes[0].set_title('Soap sold per month ')
axes[1].plot(company_data.Month,company_data.FaceWash,marker='o')
axes[1].set_title('FaceWash sold per month ')
plt.show()


# ### 90. Create Box Plot for Total Profit variable

# In[105]:


company_data.Total_Profit.plot(kind='box')


# In[ ]:





# In[ ]:




