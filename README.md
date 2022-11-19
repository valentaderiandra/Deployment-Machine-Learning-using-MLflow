# Deployment-Machine-Learning-using-MLflow


[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)

# Machine Learning Model-Deployment
# Model Tracking Using ML FLow
mini project by Group 1 Data Science B - MyEduSolve

# Use Case
* **Use Case Summary**


* **Objective Statement** :
    
    * Get business insight about the average, maximum and minimum minimum visitor and buyer
    * Get business insight about how many visitor in 30 day
    * Get business insight about how many buyer in 30 day
    * Get business insight about how many visitors buy more than once
    * Get business insight about how to predict buyers based on visitors in one day using Machine Learning Simple Linear Regression
    * Deployment using ML Flow
    
    
* **Challanges** :
    * Do not know the time of data collection
    * Don't know what the opening and closing hours are


* **Methodology / Analytic Technique** :
    * Descriptive analysis
        * Describe the information such as, min/max value of each column, average, and the total count 
    * Graph analysis
    * Using Machine Learning to predict 
        * Using Simple Linear Regression
    * Using ML Flow to Deployment


* **Business Benefit**:

    * Gain insight to treat and keep customers based on segment
    * Gain insight to improve the quality of company services so that customers remain loyal and gain more profit for the company
    * Build machine learning using Simple Linear Regression
    * Deployment using MLFlow


* **Expected Outcome**:

    * Know about the average, maximum and minimum minimum visitor and buyer
    * Know how many visitor in 30 day?
    * Know how many buyer in 30 day?
    * know how many visitors buy more than once?
    * Know how to predict total buyers based on visitors in one day using Machine Learning Simple Linear Regression ?
    * Know how to Deployment using MLFlow?

# Business Understanding
* Data is a sales data in supermarkets based on visitors and buyers within 30 days:
    * How about the average, maximum and minimum minimum visitor and buyer
    * How many visitor in 30 day?
    * How many buyer in 30 day?
    * How many visitors buy more than once?
    * How to predict buyers based on visitors in one day using Machine Learning Simple Linear Regression?
    * How to Deployment using ML Flow?
    
# Data Preparation

* **Code use** :
    * Python 3.9.13
* **Package** : 
    * Pandas, Numpy, Matplotlib, Seaborn, Scipy, Sklearn, and Warning 
    
# Data Cleansing

the data is clean **because there are no missing values and the data types are appropriate** so **no need for data cleansing**

# Exploratory Data Analysis

In statistics, exploratory data analysis is the approach of analyzing a data set to summarize its main characteristics, often using statistical charts and other data visualization methods

* **Average, maximum and minimum minimum visitor and buyer**

![1 describe](https://user-images.githubusercontent.com/101789879/202834553-ab1ee81b-fc30-4f7e-9569-2ab81b319c8a.jpg)

From the results of a descriptive analysis using 30 rows of data, some insights were obtained. The distribution of the data is still normal, this can be seen from the small standard deviation. **On average** buyers are 35 visitors with 33 purchases.**In busy times** , the maximum number of visitors is 42 visitors with **the most purchases** are 38. **The quietest visitors** are 29 visitors and **the minimum buyer** is 29 so that **it can be concluded that most likely when it is quiet every visitor buys 1 goods**. When there were 38 visitors, the average purchase was 36. When there were 35 visitors, the average buyer was 33. When there were 32 visitors, the average buyer was 31.

* **How many visitor in 30 day**

![2 how many visitors](https://user-images.githubusercontent.com/101789879/202834603-bd915bca-c166-46ee-af8d-e7dd4a15e4b4.jpg)

From visitor data in the last month, **the most frequent** (4 days) visitors came to the supermarket **as many as 40 and 36 visitors**. this might happen because **weekends** in one month can happen 4 times (Saturdays and Sundays). **the loneliest day** happened when one day only had **29 visitors**, fortunately it only happened for 1 day. **possibly** there will be no visitors because it falls on **Monday and it's raining**

* **How many buyers in 30 day**

![3 how many buyers](https://user-images.githubusercontent.com/101789879/202834711-59e55568-3ca9-4e8e-b9da-37c76d3e64bd.jpg)

from the last month's data, most often (**6 days**) there are **32 buyers in one day**. it might happen because it coincides with **weekend and there is a promotion**. and **the loneliest** day of buyers occurred where there were only **29 buyers in one day.**

* **How many visitors buy more than once**

![3 visitors who buy more than one time](https://user-images.githubusercontent.com/101789879/202834742-4b18def7-1611-44bd-8383-29f5ef4b9e55.jpg)

can be seen here, the possibility of a visitor coming in one day to buy **more than one item is 26%**. this might happen because **visitor's friends / family ask to buy the same item as the visitor**

# Modelling using Simple Linear Regression

* **Definitionn**
Simple linear regression is a regression model that estimates the relationship between one independent variable and one dependent variable using a straight line. Both variables should be quantitative.

* **Model Result by Bar Chart**

![1 bar chart actual predict](https://user-images.githubusercontent.com/101789879/202834816-34c38a03-5ddb-4303-a4e4-5659db73a636.jpg)

from the bar chart above it can be concluded that **predictions** and **actual are not too far away**

* **Model Result by Scatter Plot**

![2 scatter plot ](https://user-images.githubusercontent.com/101789879/202834919-e13a3382-102e-4e1c-930d-3e66864e9c09.jpg)


The red dot on the scatter plot is the actual training data and the blue line is the predicted Y value line with X test values. The results of the scatter plot show that there is indeed a linear relationship between visitors and buyers. The more visitors, the more buyers. However, there are cases such as:

* In a day there are 42 visitors but a total of only 36 buyers, possibly because the goods run out or the visitors only look at the supermarket.
* Of the 29 visitors who bought, there were 30 possibilities because visitors bought more than 1 time.

# Evaluate Model

* **RMSE**

Root Mean Squared Error (RMSE) is one way to evaluate a linear regression model by measuring the accuracy of the predict results of a model.

* **MAE**

MAE (Mean Absolute Error) is the average absolute difference between the actual (actual) value and the predicted (forecasting) value.

* **MAPE**

Mean Absolute Percentage error (MAPE) is the absolute average percentage error.

* **R Squared**

R squared is a number that ranges from 0 to 1 which indicates the magnitude of the combination of independent variables that jointly affect the value of the dependent variable. The R-squared value (R2) is used to assess how much influence certain independent variables have on the dependent variable.

![3 evaluate model without](https://user-images.githubusercontent.com/101789879/202835080-5e3084c2-78f7-43ed-ae11-3f12f11e54b5.jpg)

* here we can get the **RMSE of 2.9032768988296604**. This RMSE is **relatively small** so that the **model formed is good for predicting** the data.
* **MAE of 2.30407177363699**. This RMSE is classified as **small enough so that the model formed is good** to predict the data.
* Mape : absolute percentage of **mean error** is 0.06482235882578176 or **6%** . This mape is classified as **small enough** so that the model formed is **good for predicting the data**.
* here we can get the MAE of -0.17929580290702618. This R2 means that the correlation is not too strong between visitors and buyers

# Simple Linear Regression Model With Cross Validation and Hyperparameter Tuning

We can combine simple linear regression models with hyperparameter tuning to obtain the best parameters to produce even better models.

![4 evaluate model with](https://user-images.githubusercontent.com/101789879/202835174-324c5cc5-967e-4271-ab8e-163232c08ddf.jpg)

here it is obtained if before and after the hyperparameter there is no change, either because the data is too small or something

# Comparing RMSE, MAE, MAPE and R Squared Without and With CV and Hyperparameter Tuning

![5 compare](https://user-images.githubusercontent.com/101789879/202835222-e6e78698-ad22-4ea2-86d6-adc1500669c8.jpg)

The accuracy value after using CV and hyperparameter tuning is not different from the model without CV and hyperparameter tuning. This can happen because the data is too little to do Machine learning and indeed the parameters used have been optimal before.

# Deployment 

* **Import Package** and **Load Dataset**

![1 import package, load dataset](https://user-images.githubusercontent.com/101789879/202835288-b4378fa3-5e3d-43a0-9099-1ee058383c10.jpg)

* **Preprocessing Model** and **Split Train Test*

![2 preprocessing model and split](https://user-images.githubusercontent.com/101789879/202835381-b35ada50-05f8-4e92-aabe-b25b5157591c.jpg)

* **Modelling**

![3 modelling](https://user-images.githubusercontent.com/101789879/202835402-fec56ef0-956d-4eca-88e9-0b6af18ac591.jpg)

* **Result**

![4 result mae mape rmse r2](https://user-images.githubusercontent.com/101789879/202835463-f00e1bfc-bb39-4fb9-8bf2-35210982b782.jpg)

# Result

From the results of a descriptive analysis using 30 rows of data, some insights were obtained. The distribution of the data is still normal, this can be seen from the small standard deviation. **On average** buyers are 35 visitors with 33 purchases.**In busy times** , the maximum number of visitors is 42 visitors with **the most purchases** are 38. **The quietest visitors** are 29 visitors and **the minimum buyer** is 29 so that **it can be concluded that most likely when it is quiet every visitor buys 1 goods**. When there were 38 visitors, the average purchase was 36. When there were 35 visitors, the average buyer was 33. When there were 32 visitors, the average buyer was 31.

From visitor data in the last month, **the most frequent** (4 days) visitors came to the supermarket **as many as 40 and 36 visitors**. this might happen because **weekends** in one month can happen 4 times (Saturdays and Sundays). **the loneliest day** happened when one day only had **29 visitors**, fortunately it only happened for 1 day. **possibly** there will be no visitors because it falls on **Monday and it's raining**

from the last month's data, most often (**6 days**) there are **32 buyers in one day**. it might happen because it coincides with **weekend and there is a promotion**. and **the loneliest** day of buyers occurred where there were only **29 buyers in one day.**

can be seen here, the possibility of a visitor coming in one day to buy **more than one item is 26%**. this might happen because **visitor's friends / family ask to buy the same item as the visitor**

from Visitor and Buyer distribution. the graph above it can be concluded that **data distribution is not normal**. **because the mean, mode and median are not the same**

from heatmap, there is a strong relationship of total visitors to buyers. **the more visitors the more buyers**. with a correlation value on the heatmap of **0.57** where the figure is above **0.5**


we can get the **RMSE of 2.9032768988296604**. This RMSE is **relatively small** so that the **model formed is good for predicting** the data.


we can get **MAE of 2.30407177363699**. This RMSE is classified as **small enough so that the model formed is good** to predict the data.

Mape : absolute percentage of **mean error** is 0.06482235882578176 or **6%** . This mape is classified as **small enough** so that the model formed is **good for predicting the data**.

here we can get the MAE of -0.17929580290702618. This R2 means that the correlation is not too strong between visitors and buyers

**Results** of **RMSE, MAE, and MAPE** from simple linear regression models without and with Cross Validation and Hyperparameter Tuning have **values that are relatively small**. This shows that **the prediction error is also quite small** so **the model is classified as very good in terms of predicting the data**.

When we compate with Cross model, the results obtained both before and after the results are the same

with deployment code, we got same score like :   

RMSE: 2.9032768988296604
  
  MAE: 2.30407177363699
  
  MAPE: 0.06482235882578176
  
  R2: -0.17929580290702618
  
# Recomendation

It's been good between visitors and total buyers, but **needs to increase the value of existing goods** because you can see there are still **visitors just looking at them without buying**

on **weekend** you can **multiply goods** because buyers occur at that time

on **weekday** you can **promote goods** and **discounts for the number of buyers**

# Deployment Result

MLFlow is a tool used to make it easier for Data Scientists so they don't have to worry about organizing models, recording each experiment, and facilitating the deployment process.

there are several features in mlflow, but what we use is model tracking on rmse, mae, mape, and r2

![5 mlflow component](https://user-images.githubusercontent.com/101789879/202835540-cab8cf02-7023-4d23-bc86-7aa9b0f73984.png)

Model tracking is a feature for recording experiments so that each experiment will be properly recorded in mlflow.

![5 mlflow ui ](https://user-images.githubusercontent.com/101789879/202835571-4ff69978-6e11-4b57-aeda-ac19a9f49612.jpg)


