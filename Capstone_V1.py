# Import Libraries
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Set view options for info() and head()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# Import data (We need to establish a Database and call that. The excel file is too big.)
# data = pd.read_csv(r"/content/drive/MyDrive/SLADA_Project/merged_data.csv")

data1 = pd.read_csv(r"C:\Users\antho\OneDrive\SMU\Semester 6 Summer 2024\Capstone A\Datasets\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv")
data2 = pd.read_csv(r"C:\Users\antho\OneDrive\SMU\Semester 6 Summer 2024\Capstone A\Datasets\Friday-16-02-2018_TrafficForML_CICFlowMeter.csv")
data3 = pd.read_csv(r"C:\Users\antho\OneDrive\SMU\Semester 6 Summer 2024\Capstone A\Datasets\Friday-23-02-2018_TrafficForML_CICFlowMeter.csv")
data4 = pd.read_csv(r"C:\Users\antho\OneDrive\SMU\Semester 6 Summer 2024\Capstone A\Datasets\Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv")
data5 = pd.read_csv(r"C:\Users\antho\OneDrive\SMU\Semester 6 Summer 2024\Capstone A\Datasets\Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv")
data7 = pd.read_csv(r"C:\Users\antho\OneDrive\SMU\Semester 6 Summer 2024\Capstone A\Datasets\Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv")
data8 = pd.read_csv(r"C:\Users\antho\OneDrive\SMU\Semester 6 Summer 2024\Capstone A\Datasets\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv")
data9 = pd.read_csv(r"C:\Users\antho\OneDrive\SMU\Semester 6 Summer 2024\Capstone A\Datasets\Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv")
data10 = pd.read_csv(r"C:\Users\antho\OneDrive\SMU\Semester 6 Summer 2024\Capstone A\Datasets\Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv")

data6 = pd.read_csv(r"C:\Users\antho\OneDrive\SMU\Semester 6 Summer 2024\Capstone A\Datasets\Thursday-20-02-2018_TrafficForML_CICFlowMeter.csv")
data6.shape

# List of columns to drop from data6 that arent in any other datasets
columns_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP']

# Drop the specified columns
data6 = data6.drop(columns=columns_to_drop)

# df1 = pd.concat([data1, data2, data3, data4, data5, data7, data8, data9, data10], axis=0, ignore_index=True)
df1 = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10], axis=0, ignore_index=True)

data6.shape
df1.shape
# df2.shape

# check the first rows of data
df1.head()
data6.head()
df1.tail()
# df2.head()
# df2.tail()

# Drop the specified columns
df1 = df1.drop(columns=columns_to_drop)

# Count the total number of duplicate rows
num_duplicates = df1.duplicated().sum()
print(f"Total number of duplicate rows: {num_duplicates}")

# Remove all duplicates and keep only unique rows
df1 = df1.drop_duplicates(keep=False)
df1.shape

# Convert object types to categorical types
for col in df1.columns:
    if df1[col].dtype == 'object':
        df1[col] = pd.Categorical(df1[col]).codes

        
# Convert columns to numeric, coercing errors to NaN
for col in df1.columns:
    if df1[col].dtype != 'object':  # Proceed only if the column is not of object type
        df1[col] = pd.to_numeric(df1[col], errors='coerce')
        
# Replace infinite values with NaN
df1 = df1.replace([np.inf, -np.inf], np.nan)
   
# Count NaNs in each column
nan_count_per_column = df1.isna().sum()
print("NaNs per column:\n", nan_count_per_column)

# Drop rows with missing values
df1 = df1.dropna()
df1.shape

# check the data types
df1.info()

# count of missing values per column
df1.isnull().sum()

# Count of unique values in each column
df1.nunique()

# Summary stats of the data
df1.describe()

#Find the correlations for df1 with the label
correlations = df1.corr()['Label'].sort_values(ascending=False)
print(correlations)

# Find the positive and negative correlations of significant features
positive_correlations = correlations[correlations > 0.05]
negative_correlations = correlations[correlations < -0.05]

# Combine the positive and negative correlations
filtered_correlations = pd.concat([positive_correlations, negative_correlations], axis=0)
print(filtered_correlations)
filtered_correlations.shape

# Create a filtered dataset using columns that have correlations > 0.05 or < -0.05
filtered_columns = filtered_correlations.index.tolist()  # Get the list of column names
filtered_columns
filtered_dataset = df1[filtered_columns]  # Filter original dataset for these columns
filtered_dataset.shape

# Heat map of correlations
sns.heatmap(filtered_dataset.corr(),cmap='coolwarm', vmin=-1, vmax=1, xticklabels=True, yticklabels=True, annot=True, fmt=".2f")
plt.subplots(figsize=(100,100))
plt.title("Correlation Matrix Heatmap")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()

# Exclude the label column and compute the correlation matrix
correlation_matrix = filtered_dataset.drop(columns=['Label']).corr()

# Find pairs of features with correlation greater than 0.95
high_corr_pairs = []

# Iterate through the correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:  # Check if the correlation is greater than 0.9
            feature_1 = correlation_matrix.columns[i]
            feature_2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            high_corr_pairs.append((feature_1, feature_2, corr_value))

# Print features with high correlation
if high_corr_pairs:
    print("Features with a correlation greater than 0.9:")
    for feature_1, feature_2, corr_value in high_corr_pairs:
        print(f"{feature_1} and {feature_2}: {corr_value:.2f}")
else:
    print("No feature pairs with a correlation greater than 0.9.")

# Drop the highly correlated features
df1_reduced = filtered_dataset.drop(columns=["Fwd IAT Tot", "Fwd IAT Std", "Fwd Pkt Len Std", "Flow IAT Min", "Bwd IAT Tot", "Pkt Size Avg", "Pkt Len Min"])
df1_reduced.shape

# Print the correlation Matrix and the feature reduced data frame
# Heat map of correlations
sns.heatmap(df1_reduced.corr(),cmap='coolwarm', vmin=-1, vmax=1, xticklabels=True, yticklabels=True, annot=True, fmt=".2f")
plt.subplots(figsize=(10,10))
plt.title("Correlation Matrix Heatmap")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()




