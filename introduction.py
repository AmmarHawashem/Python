# Import an entire library and give it an alias
from os import getcwd
from typing import Type
from numpy.lib.function_base import delete
import pandas as pd # For DataFrame and handling
import numpy as np # Array and numerical processing
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame # Low level plotting
import seaborn as sns # High level Plotting
import statsmodels.api as sm # Modeling, e.g. ANOVA
import math # Functions beyond the basic maths
import plotnine # Functions beyond the basic maths

# Import only specific modules from a library
# we'll use this for the t-test function
from scipy import stats 
# ols is for ordinary least squares
from statsmodels.formula.api import ols


# 3.1 Key types:
  a= True       #bool = binary = True and False 
  b= 1          #int
  c= 1.5        #float
  d="Kick off"  #str
    type(a)
    type(b)
    type(c)
    type(d)

# 3.2 Arithmetic operators
    #Addition (+)
    2 + 3
    #Subtraction (-)
    2 - 3
    #Multiplication (*)
    2 * 3
    #Exponentiation (**)
    4 ** 2
    4 ** 0.5
    #Division (float) (/)
    25 / 4
    #Division (floor) (//) just the inteeger, no reminder
    25 // 4
    #Modulus (reminder) (%)
    25 % 4

    # Exercise 3.1 (Differences in handling types) What do expect when executing the following functions?
    1 + 1     
    '1' + '1'
    '1' * 5     #will repeate it 5 times
    '1' '1'

    # Try it with strings
    First = 'Ammar'
    Space =" "
    Last = "Hawashem"  
    myName = First + Space + Last  
    myName
    myName * 2 #will be repeated as well

# 3.6 Lists, Data Containers part I
    # list --> [] 

    # len()	        The number of values, n
    # np.mean()	    The mean
    # stats.mode()	The mode
    # np.median()	The median
    # np.var()	    The variance
    # np.std()	    The standard deviation
    # stats.iqr()	The inter-quartile range
    # max()	        The maximum value
    # min()	        The minimum value
    # range()	    The range (sequence)
    #sum()




    [1, 6, 9, 36] + 100                     #Error
    [1, 6, 9, 36] + [10, 100]
    [1, 6, 9, 36] + [100, 200, 300, 400]
    [1, 6, 9, 36] + [100, "dogs", 300, 400]
 #You can conatenate list inside a list so the 1st one will not work

# Let's try some of the aforementioned functions:
heights = [167, 188, 178, 194, 171, 169]
    n    =  len(heights)
    AVG  =  np.mean(heights)
            sum(heights)
            stats.mode(heights)
            np.median(heights)	
            np.var(heights)	
    stdD =  np.std(heights)	
            stats.iqr(heights)	# Q3-Q1
            max(heights)	
            min(heights)
 #  Exercise 3.3 (Functions and math) Given the definition of the Normal 95% confidence interval
 # calculate the 95% confidence interval for the list of heights.
 lower = AVG - 1.96 * (stdD/(n**0.5))
 upper = AVG + 1.96 * (stdD/(n**0.5))
 ci= [round(lower,3), round(upper,3)]   #three decimals
 ci

# 3.7 Making your own functions
 def addNumbs(x, y):
    z = x + y   #indented with 4 spaces (one tap)
    return z   
 addNumbs(2,3)
 
 # A better way
 def addNumbs(x, y):
    """Add two numbers together"""  #docstrings
    z = x + y
    return z
    # SEE DOCTOR's EXAMPLE

 #  lambda: function without a name (anonymus):
  #Classic function:

  def RaisePower(x, y):
     """Exponents"""
     return x**y
   Try_RaisePower = RaisePower(2, 3)
   type(RaisePower) # -->  function
  
  #lambda:
   # <name> = lambda parameter:body
   TryLambda_RaisePower = lambda x, y : x**y
   TryLambda_RaisePower (2, 3)
   type(TryLambda_RaisePower)  # -->  function
  #Note: Try_RaisePower is an assigned varible from the classic function,
   # but there is no one for the almbda function     


 #Review  maping


# Review 3.9 & 3.10 (attrib + methods)

#3.11 Deictionaries 
   # {'key1' :values, 'key2':values}
   # NOTE: dictionaries has no order(no pisition is linked wuth its value), so indexing won't work 

   d1 = {'int_value':3, 
     'bool_value':False, 
     'str_value':'hello'}
     type(d1) # --> dictionary
    d1
    print(d1['str_value']) #print the values of this key
      # <Dic>['<key>'] --> values of that key 

   organizations = {'name': ['Volkswagen', 'Daimler', 'Allianz', 'Charite'],
                 'structure': ['company', 'company', 'company', 'research']}
      organizations['name']
         # <Dic>['<key>'] --> values of that key  
           
      # organizations[0] --> Error
         # becuase Dictionaries has no order

   heights = [167, 188, 178, 194, 171, 169]
   persons = ["Mary", "John", "Kevin", "Elena", "Doug", "Galin"]
   weights = [67, 88, 78, 94, 71, 69]

      # zip(key, value) --> order is important
      heights_persons = dict(zip(persons, heights))
      # ZIP also can have a zip value --> no nade to make it as a list
      hw = zip(heights,weights)  #--> DON'T make it as dict
      heights_weight_persons = dict(zip(persons, hw))

      # There are specific methods to use here also:
      heights_persons.values()   # dict.values()
      heights_persons.keys()     # dict.keys()
      heights_weight_persons.values()
      heights_weight_persons.keys()

   # Exercise 3.16 Returning to the two lists containing the cities and their distances from Berlin, above,
   # create a dictionary with two key/value pairs: city, containing a list of five cities, and distance, containing a list of 5 integers.
   #  Assign this to the variable distDict
      cities = ['Munich', 'Paris', 'Amsterdam', 'Madrid', 'Istanbul']
      dist = [584, 1054, 653, 2301, 2191]
      # Solustion
      distDict = dict(zip(cities, dist))
      type(distDict)
      distDict1 = {"City":cities, "Dist":dist}

# 3.12  NumPy Arrays, Data Containers
   # Arrays are similar to lists, but are more dynamic:
   # They can be n-dimensional. The nd in ndarray stands for n-dimensional, we’ll only be dealing with 1 and 2-dimensional arrays in this workshop.
   # They only take one data type!
   #The primary reason we use NumPy arrays is to vectorize functions over all values

   # lists
   xx = [3, 8, 9, 23]
   print(xx)   
   type(xx) #--> list      
   # but what you really want is a NumPy array:
      #import numpy as np --> I already have imported it
   xx = np.array([3, 8, 9, 23])  # <variable> = np.array( [VofList] )
   print(xx)   
   type(xx) #--> numpy.ndarray

   #NumPy arrays come in all shapes and sizes:
   np.array([[5,7,8,9,3], 
          [0,3,6,8,2],
          range(5)])
          #Note: range(n) = range (0,n) where [0,n) or [0,n-1]
         range(4, 10)
         list(range(4, 10)) #--> [4,5,6,7,8,9]
         
   



# Chapter 4 DataFrames in pandas

   # construct a data frame from stractch:
      # 1- make the lists (values of each column) []:
         foo1 = [True, False, False, True, True, False ]
         foo2 = ["Liver", "Brain", "Testes", "Muscle", "Intestine", "Heart"]
         foo3 = [13, 88, 1233, 55, 233, 18]
      # 2- use: key:values as <columnHeader>:<list contains the values
         foo_dict= {"healthy":foo1, 'tissue':foo2, "quantity":foo3}
      # 3- Use <DF_Name> = pd.DataFrame( <dict> )
         foo_df = pd.DataFrame(foo_dict)
   # Exercise 4.1 convert distDict to a DataFrame
      distDict_df = pd.DataFrame(distDict1)
      distDict_df = pd.DataFrame(distDict)
      #Another method with zip sec:4.2
      list_headers = ['City', "Distance"]
      list_values = [cities, dist]
      zip_list = list(zip(list_headers, list_values))
      type(zip_list) 
      #Convert it into a dict to be able to convet it into dataframe
      zip_dic = dict(zip_list)
      zip_df = pd.DataFrame(zip_dic)

# 4.3 Accessing columns by name
   # Just values of a column as a series (without the header)
      foo_df['healthy']
      # Or
      foo_df.healthy
      #Series = a list = 1-dimensional NumPy array, in that it can only have a single type   
   # As a dataframe (with the header)
      foo_df[['healthy']]

   # Exercise 4.4 Select both the quantity and tissue columns.
      foo_df[['quantity','tissue']]        # as A dataframe

      # As a series
         #foo_df['quantity','tissue']       # NOT WORKING

      # but there is  a cheap trick
         [foo_df['quantity'], foo_df['tissue']]
         #OR
         [foo_df.quantity, foo_df.tissue]  
         type([foo_df.quantity, foo_df.tissue] )

   

# 4.8 mcars exercise
   # mpg	   Miles/(US) gallon
   # cyl	   Number of cylinders
   # disp	Displacement (cu.in.)
   # hp	   Gross horsepower
   # drat	Rear axle ratio
   # wt	   Weight (1000 lbs)
   # qsec	1/4 mile time
   # vs	   Engine (0 = V-shaped, 1 = straight)
   # am	   Transmission (0 = automatic, 1 = manual)
   # gear	Number of forward gears 

   # What to do?
      # Import the data set and assign it to the variable mtcars.
       mtcars = pd.read_csv('data/mtcars.csv')
      # Calculate the correlation between mpg and wt and test if it is significant.
         # specify the model
         """ import statsmodels.api as sm
         from statsmodels.formula.api import ols """
         model = ols("mpg ~ wt", data=mtcars)
         results = model.fit()
          # Explore model results
            results.summary()
           # Since r^2 is 0.753 there is a relationship but nit significant

      # Visualize the relationship in an XY scatter plot. (CH6)
         # plt.scatter(<dataframe>[<columnX],<dataframe>[<columny] )
         plt.scatter(mtcars["wt"], mtcars["mpg"], alpha=0.65)
         plt.title('A basic scatter plot')
         plt.xlabel('weight')
         plt.ylabel('miles per gallon')
         plt.show()

         # Or by adding another variable
         sns.scatterplot(x="wt", y="mpg", hue="cyl", data = mtcars)

      # Convert weight from pounds to kg.
            # 1kg = 2.205 lbs
         mtcars['wt'] =round (mtcars["wt"] /2.205, 3) #3 decimal places
         mtcars

# 5.0.1 Indexing
 foo_df
 # Using ['<columnHeader>']:   
  foo_df['tissue']    #As a series
 # using . notation
  foo_df.tissue       #As a  series

 #   We can also select items using using position.
 #  e.g. we can index rows by index position with .iloc[]:
  # First row, as a Series (without a header)
   foo_df.iloc[0] 
  # First row, as a DataFrame (with a header)
   foo_df.iloc[[0]] 
  # a list of integers, the first two rows
   foo_df.iloc[[0, 1]] 

 # But more explicitly, we can use [ <rows> , <columns> ] notation. 
 # In this case we must also use : notation to specify ranges,
 #  even if we want all rows or columns.

  # To get all columns, use : after the comma
   foo_df.iloc[0, :] 
  # a list of integers, the first two rows
   foo_df.iloc[[0, 1], :]
  # The first two columns, all rows
   foo_df.iloc[:,:2]
  # A single column, all rows
   foo_df.iloc[:,1:2]
 
 #Exercise 5.1
   # The 2nd to 3rd rows?
      foo_df.iloc[[1,2]]
      # or
      foo_df.iloc[1:3 , :]

   # The last 2 rows?
      foo_df.iloc[[-2,-1]]
      # or
      #foo_df.iloc[-2:-1 , : ] HOW
   
   #A random row in foo_df?
      foo_df.sample()

   # From the 4th to the last row?
      foo_df.iloc[ 3: , :]  #Reminder: Python is stupid starts indexing at 0

 # Exercise 5.3 (Indexing at intervals) Use indexing to obtain all the odd and even rows only from the foo_df data frame.
   # seq[start:end:step]  = <dataframe>.iloc[::]
   Even = foo_df.iloc[::2]
   Odd  = foo_df.iloc[1::2]

# 5.1.2 Logical Operators
   # A- Relational Operators
      # <	Less than
      # <=	Less than or equal to
      # >	Greater than
      # >=	Greater than or equal to
      # ==	Exactly equal to
      # !=	Not equal to, i.e. the opposite of ==
      # ~x	Not x (logical negation)

   # B- Logical Operators
      #   x | y	x OR y
      #   x & y	x AND y

   # <DataFrame>[<DataFrame>.<columnHeader>  Log/Rel OPE ]      #just one
   #<DataFrame>[ (<DataFrame>.<columnHeader>)  Log/Rel OPE (<DataFrame>.<columnHeader>)]  #Two or more (| or &)
   foo_df[foo_df.quantity == 233]
   foo_df[(foo_df.tissue == "Heart") | (foo_df.quantity == 233)]

   # Exercise 5.4 Subset for boolean(T/F) data:

      # Only “healthy” samples:
       foo_df[foo_df.healthy == True]

      #Only “unhealthy” samples.
       foo_df[foo_df.healthy == False]
       #OR
       foo_df[foo_df.healthy != True]

   # Exercise 5.5 Subset for numerical data:

      # Only low quantity samples, those below 100:
       foo_df[foo_df.quantity < 100]

      # Quantity between 100 and 1000,
       foo_df[(foo_df.quantity > 100) & (foo_df.quantity < 1000)]

      # Quantity below 100 and beyond 1000.
       foo_df[(foo_df.quantity < 100) & (foo_df.quantity > 1000)]

   # Exercise 5.6 Subset for strings:
      # Only “heart” samples.
       foo_df[foo_df.tissue == "Heart"]

      # “Heart” and “liver” samples
       foo_df[(foo_df.tissue == "Heart") | (foo_df.tissue == "Liver")  ]

      # Everything except “intestines”
       foo_df[foo_df.tissue != "Intestine"]



# Class Exercise
   mtcars = pd.read_csv('data/mtcars.csv')

   # Replace one value with a new value for a DataFrame column:
      # <DataFrame>['<column name>'] = <DataFrame>['<column name>'].replace('<OLD_V>', '<NEW_V>')
   mtcars['model'] = mtcars['model'].replace("Mazda RX4" , "Hyunday Elantra")
   mtcars

   #  Replace multiple values with multiple new values for an individual DataFrame column:
      # <DataFrame>['column name'] = <DataFrame>['column name'].replace(['1st OLD_V ','2nd OLD_V',...],['1st NEW_V','2nd NEW_V',...])
   mtcars['model'] = mtcars['model'].replace(["Mazda RX4",'Mazda RX4 Wag'],["Hyunday Elantra","Hyunday Accent"])
   mtcars
      

   #  Replace multiple values with multiple new values for an individual DataFrame row:
   #    # <DataFrame>.loc[<pos>, '<col1>', <col2>, ...] = ['NEW_V1', 'NEW_V2', ... ]
   mtcars.loc[0, ['model', 'cyl', 'disp']] = ['Ford Expedetion', '4', '100'] 
   mtcars

# While loop

#
i=1
while i<=10:
   i = i+1
   if i ==6:
     continue  #/break
   print(i)

#
nylist =[]
listnames = ["Ammar", "Yasser"]
for NV in range(len(listnames)):
   if listnames[NV] == "Ammar":
      print("Ammar")
   else:
      print("Yasser")


if __name__ == '__main__':
    n = int(raw_input())

n = 5
i = 0
while i<n:
   print(i**2)
   i = i+ 1

   
#   If  is odd, print Weird
# If  is even and in the inclusive range of  to , print Not Weird
# If  is even and in the inclusive range of  to , print Weird
# If  is even and greater than , print Not Weird # 

n=6
if (n % 2 != 0):
   print("Weird")
elif (n >=2) & (n<=5):
      print("Not Weird")
elif (n >=6) & (n<=20):
      print("Weird")
else:
   print("Not Weird")


   
   
   
 



