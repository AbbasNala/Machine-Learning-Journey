import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr    
from scipy.stats import chi2_contingency    
 

warnings.filterwarnings('ignore')

df = pd.read_csv('insurance.csv')


# -----------EDA (Exploratory Data Analysis)-----------------
print(df.shape) #shows the number of rows and columns in the dataset
print("\n")
print(df.head())  #shows the first 5 rows of the dataset
print("\n")
print(df.info()) #shows the summary of the dataset
print("\n")
print(df.describe()) #shows the statistical summary of the dataset
print("\n")
print(df.isnull().sum()) #shows the number of missing values in each column
print("\n")

# For graphical representation of the data, we can use seaborn and matplotlib libraries
numerical_columns = ['age', 'bmi', 'children', 'charges']

# for cols in numerical_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[cols], kde=True,bins=20)   #Histrogram with Kernel Density Estimation (KDE) for numerical columns
    plt.title(f'Distribution of {cols}')
    plt.xlabel(cols)
    plt.ylabel('Frequency')
    plt.show()

# for cols in numerical_columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[cols])   #Boxplot for numerical columns
    plt.title(f'Boxplot of {cols}')
    plt.xlabel(cols)
    plt.show()


plt.figure(figsize=(10, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')  #Heatmap for correlation between numerical columns
plt.show()

#---------------------------Data Cleaning-----------------------------

df_cleaned = df.copy()  # Create a copy of the original dataframe to clean
print(df_cleaned.shape)
df_cleaned.drop_duplicates(inplace=True)  # Remove duplicate rows
print(df_cleaned.shape)

# print("\n")
print(df_cleaned['sex'].value_counts() ) # Check the distribution of the 'sex' column

# print("\n")
# print(df_cleaned.dtypes)

# # now converting all the string columns to numeric columns

print("\n")
# # converting 'sex' column to numeric using labelled encoding
df_cleaned['sex'] = df_cleaned['sex'].map({'male': 0, 'female': 1}) 
# print(df_cleaned.head())   

# print("\n")
# # converting 'smoker' column to numeric using labelled encoding
df_cleaned['smoker'] = df_cleaned['smoker'].map({'no': 0, 'yes': 1})
# print(df_cleaned.head())    
# print("\n")
# print(df_cleaned.dtypes)


# print("\n")
# # renaming sex and smoker columns to is_female and is_smoker for better understanding
df_cleaned.rename(columns={'sex': 'is_female', 'smoker': 'is_smoker'}, inplace=True)
# print(df_cleaned.head())
# print("\n")

# print(df_cleaned['region'].value_counts())  # Check the distribution of the 'region' column    

# # converting 'region' column to numeric using one-hot encoding , by creating new columns for each category in the 'region' column
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)  # drop_first=True to avoid dummy variable trap
# print("\n")
# print(df_cleaned.head())

# #converting all columns to integer type
df_cleaned=df_cleaned.astype(int)
 
# print("\n")
# print(df_cleaned.head())  # Check the data types of the columns after conversion

#---------------------------Feature Engineering-----------------------------
sns.histplot(df_cleaned['bmi'])
# plt.show()

df_cleaned['bmi_category'] = pd.cut(df_cleaned['bmi'],bins=[0,18.5,24.9,29.9,float('inf')],labels=['Underweight','Normal weight','Overweight','Obese'])
# print(df_cleaned.head())

# print(df_cleaned.head())
sns.countplot(x='bmi_category', data=df_cleaned)
# plt.show()

print("\n")
# Converting the 'bmi_category' column to numeric using one-hot encoding
df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=False)  # drop_first=False to keep all categories
# print(df_cleaned.head())


# Be carefull while entering column names in the below code, as they are case sensitive and should match exactly with the column names in the dataframe

print("\n")
# Converting the 'bmi_category' columns to binary values (0 and 1) for better understanding
df_cleaned['bmi_category_Underweight'] = df_cleaned['bmi_category_Underweight'].map({False: 0, True: 1})
df_cleaned['bmi_category_Normal weight'] = df_cleaned['bmi_category_Normal weight'].map({False: 0, True: 1})
df_cleaned['bmi_category_Overweight'] = df_cleaned['bmi_category_Overweight'].map({False: 0, True: 1})
df_cleaned['bmi_category_Obese'] = df_cleaned['bmi_category_Obese'].map({False: 0, True: 1})
# print(df_cleaned.head())

## Here instead of df_cleaned we can also use df_cleaned.astype(int) to convert all the columns to integer type, but it will convert all the columns to integer type, which may not be desired in some cases. So we are converting only the 'bmi_category' columns to binary values (0 and 1) for better understanding.
print("\n")

# Feature Scaling

# This is an important step in the data preprocessing pipeline, as it helps to bring all the features to a similar scale, which can improve the performance of some machine learning algorithms. Here we are using StandardScaler from sklearn.preprocessing module to scale the numerical features 'age', 'bmi', and 'children' to have a mean of 0 and a standard deviation of 1.

cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
print(df_cleaned.head())

# Feature Extraction
# - Here we will use different techniques to help co-relating our output data with the input data.  
# -We will use different techniques to extract features from the input data, which can help to improve the performance of the machine learning model. 
# Here we will use Pearson correlation coefficient to find the correlation between the input features and the output feature 'charges'.

# following columns are selected for correlating with the charges column
selected_features = ['age', 'bmi', 'children', 'is_female', 'is_smoker', 'region_northwest', 'region_southeast', 'region_southwest', 'bmi_category_Underweight', 'bmi_category_Normal weight', 'bmi_category_Overweight', 'bmi_category_Obese']

# correlation is a dictinary which will store the correlation between each column and charges and will store it as key-pair values  ( Key = column name, Value = correlation value)
correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features    
}

# for displaying the correlation values in a tabular format, we can use pandas DataFrame to create a table with two columns: 'Feature' and 'Correlation_with_charges'. We will then sort the table by the correlation values in descending order to see which features have the highest correlation with the output feature 'charges'.

correlation_df=pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation_with_charges'])
correlation_df = correlation_df.sort_values(by='Correlation_with_charges', ascending=False)
print(correlation_df)

# Categorical features
# -- here for comparing categorical features with the output we will use chi square test but it can be done only when both the features are categorical. but charges is a continuous variable so we will create bins for charges columns

cat_features = ['is_female', 'is_smoker',
                'region_northwest', 'region_southeast', 'region_southwest',
                'bmi_category_Normal weight', 'bmi_category_Overweight', 'bmi_category_Obese']

alpha = 0.05  # significance level

df_cleaned['charges_bins'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)  # creating 4 bins for charges column using quantiles

chi2_results = {}

for col in cat_features:
    contingency_table = pd.crosstab(df_cleaned[col], df_cleaned['charges_bins'])
    chi2_state,p_val,_,_ = chi2_contingency(contingency_table)
    decision   = "Reject Null Hypothesis" if p_val < alpha else "Accept Null (Drop feature)"
    chi2_results[col] = {'Chi2 Statistic': chi2_state, 'p-value': p_val, 'Decision': decision}  

chi2_df = pd.DataFrame(chi2_results).T
chi2_df= chi2_df.sort_values(by='p-value')
print("\n")
print(chi2_df)  # Display the chi-square test results for categorical features  

# final df_cleaned data frame will be 
final_df = df_cleaned[['age', 'bmi', 'children', 'is_female', 'is_smoker', 'region_southeast', 'bmi_category_Obese', 'charges']]

print("\n")
print(final_df.head())  # Display the final cleaned and processed dataframe

#========================================

# final_df <-- now this is the final data frame which we will use for machine learning models 

#========================================

