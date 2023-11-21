
'''

Ensembled Techniques: 
Random forest:
     



Problem Statements:

1.	A cloth manufacturing company is interested to know about the different attributes contributing to high sales.
    Build a decision tree & random forest model with Sales as the target variable (first convert it into categorical variable).



 phases of crisp ML-Q (cross industry standard process for machine learning with quality assurance):
     
 1] data understanding and business understanding 
 2] data preparation(data cleaning)
 3] model building (data mining)    
 4] model evaluation
 5] model deployment
 6] Monitoring and Maintainance      
   

1] step : data understanding and business understanding: 

business objectives  :  The primary business objective is to identify the factors that contribute to high sales.
business constraints :  The company's risk tolerance can be a constraint. Some strategies or changes may involve higher risks, and the company may be risk-averse.
    

Success Criteria:-
    
Business success criteria        : The most direct success criterion is an increase in sales revenue 
Machine Learning success criteria: The Decision Tree and Random Forest models should have high predictive accuracy in categorizing sales (e.g., high sales and low sales).
Economic success criteria        : If the model identifies areas where cost-saving measures can be implemented without compromising sales, it can be an economic success criterion.



Data Collection: 
data is collected from cloth manufacturing company:-




Data description:
Sales: The target variable, which represents the sales of products.
CompPrice: The price of the company's products.
Income: The income of consumers.
Advertising: The amount of advertising for the products.
Population: The population in the area where the products are sold.
Price: The price of the products.
ShelveLoc: The shelf location or condition of the products, categorized as "Bad," "Good," or "Medium."
Age: The age of the products or customers.
Education: The level of education of consumers.
Urban: A binary variable indicating whether the area is urban ("Yes") or not ("No").
US: A binary variable indicating whether the product is in the United States ("Yes") or not ("No").	








'''






'''
2] step : data preprocessing (data cleaning) :

'''


import pandas as pd

data = pd.read_csv(r"D:\DATA SCIENTIST\DATA SCIENCE\ASSIGNMENTS\SUPERVISED LEARNING\CLASSIFICATION\decision tree\ClothCompany_Data.csv")
   

# MySQL Database connection
# Creating engine which connect to MySQL
user = 'user1' # user name
pw = 'user1' # password
db = 'sales_db' # database

from sqlalchemy import create_engine
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# dumping data into database 
data.to_sql('sales_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# loading data from database
sql = 'select * from sales_tbl'
df = pd.read_sql_query(sql, con = engine) 
 
       
df.shape
df.dtypes
df.info()  # not any null values are present here.
df.describe()


df.duplicated().sum()
# not any duplicated rows are present here 

# outlier treatment :
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
# outliers are not present here


# Split data into features (X) and target variable (y)
X = df.drop("Sales", axis = 1) 
Y = df["Sales"]





# For X: 

# in independent features we have both categorical as well numerical variable 

numeric_features = X.select_dtypes(exclude = ['object']).columns 
numeric_features


categorical_features = X.select_dtypes(include=['object']).columns
categorical_features


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  OneHotEncoder

# Encoding - One Hot Encoder to convert Categorical data to Numeric values 
# Categorical features 
encoding_pipeline = Pipeline(steps = [('onehot', OneHotEncoder(sparse_output = False))]) #(sparse_output = Flase) dosent give output as sparse_matrix 
from sklearn.compose import ColumnTransformer
# Creating a transformation of variable with ColumnTransformer()
preprocessor = ColumnTransformer(transformers = [('categorical', encoding_pipeline, categorical_features)])
# Fit the data
clean = preprocessor.fit(X) 

import joblib
# Save the pipeline
joblib.dump(clean, 'clean_DT ') 

# Transform the original data
clean2 = pd.DataFrame(clean.transform(X), columns = clean.get_feature_names_out()) 
clean2.columns

# Concatenate the one-hot encoded columns with the original data
X1 = pd.concat([X, clean2], axis=1)


# Drop the original categorical columns
X1.drop(categorical_features, axis=1, inplace=True)

# complete EDA for X variables
  
    
    



# for Y :
# convert sales data into categorical values because we want set sales column as a target values.
unique_sales_values = df['Sales'].unique()  # if sales <= 8 set as low sales and if sales > 8 set as high sales
#convert numerical into categorical we have descretization method 

Y1 = pd.DataFrame(Y) # convert series into dataframe 


Y1['Sales'] = pd.cut(Y1["Sales"], 
                     bins = 2, 
                     include_lowest = True,
                     labels = ["low sales", "high sales"]) # below the 8 is low sales and above the 8 is high sales








'''
step: 3] model building (data mining)
    
'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)  

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth= 6) 
clf.fit(X_train,y_train) 



# post prunning : after constructing decision tree then apply hyperparameter tunning: maxdepth = 6

# visualization tree:
import matplotlib.pyplot as plt
from sklearn import tree

plt.figure(figsize=(45,15))
tree.plot_tree(clf,filled=True)




# now check training accuracy:
from sklearn.metrics import accuracy_score,classification_report    
y_train_pred = clf.predict(X_train)
score = accuracy_score(y_train_pred, y_train)
score  
report = classification_report(y_train, y_train_pred) 
report 
# now check testing accuracy:  

y_test_pred = clf.predict(X_test)    
score2 = accuracy_score(y_test_pred, y_test) 
score2 
report2 = classification_report(y_test_pred, y_test) 
report2 

# model is overfitting:  training accuracy  is high and testing accuracy is low









# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# overcome the overfitting we do preprunning technique . this is best technique.
# pre prunning : while constructing decision tree that time apply hyperparameter tunning . chooose best1 parameters

#preprunning :
    
parameters = {'criterion':['gini', 'entropy', 'log_loss'],
              'splitter' : ['best', 'random'],
              'max_depth':[1,2,3,4,5,6,7,8],
              'max_features':['auto','sqrt','log2']
    }
    # this partameters use while constructing decision tree model
    
    
from sklearn.model_selection import GridSearchCV   
from sklearn.tree import DecisionTreeClassifier
treemodel = DecisionTreeClassifier()    
cv = GridSearchCV(treemodel, param_grid = parameters, cv = 5, scoring = 'accuracy')    # for better accuracy
cv.fit(X_train,y_train)    

cv.best_params_    # it gives best parameters we put this parameters into our model to get beat testing accuracy
        

# now check training accuracy:
y_train_predd = cv.predict(X_train)
score5 = accuracy_score(y_train_predd, y_train)
score5  



# now check testing accuracy:  

y_test_pred1 = cv.predict(X_test)    
score3 = accuracy_score(y_test_pred1, y_test) 
score3     
    
import pickle 
pickle.dump(cv, open('decisiontree.pkl', 'wb'))

    
    
    
    
'''
step: 3] model building (data mining)
    
'''
    
# RandomForestclassifier:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)  
from sklearn.metrics import accuracy_score,classification_report    
        
from sklearn.ensemble import RandomForestClassifier   
rf = RandomForestClassifier()  
rf.fit(X_train,y_train)      
    
# training accuracy:
Y_trainnn = rf.predict(X_train)    
score = accuracy_score(Y_trainnn,y_train)    
score     


# testing_accuracy:
y_testtt = rf.predict(X_test)    
score2 = accuracy_score(y_testtt,y_test)    
score2 
    
    
    
    
#hyperparameter tunning for better accuracy:
    
parameter = {'criterion':['gini', 'entropy', 'log_loss'], 
              'max_depth':[1,2,3,4,5,6,7,8],
              'max_features':['auto','sqrt','log2']
    }
    
    
from sklearn.model_selection import GridSearchCV   
cv2 = GridSearchCV(rf, param_grid = parameter, cv = 8, scoring = 'accuracy')    # for better accuracy
cv2.fit(X_train,y_train)    

cv2.best_params_    # it gives best parameters we put this parameters into our model to get beat testing accuracy
        

# now check training accuracy:
y_train_pr = cv2.predict(X_train)
score11 = accuracy_score(y_train_pr, y_train)
score11 
report3 = classification_report(y_train_pr, y_train) 
report3


# now check testing accuracy:  

y_test_pr2 = cv2.predict(X_test)    
score22 = accuracy_score(y_test_pr2, y_test) 
score22
report4 = classification_report(y_test_pr2, y_test) 
report4   

# save this model
import pickle 
pickle.dump(cv2, open('GSCV_RF.pkl', 'wb'))

    
    
  
    
'''
solution:
    
    Develop strategies to retain existing customers by offering products or services that align with attributes
    driving high sales.
        
'''    
    
    
    
    









