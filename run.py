import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="sample_test_data.csv")
args = parser.parse_args()


df= pd.read_csv("train_data.csv")

columns_to_remove = ['uuid', 'datasetId']

condition_mapping = {
    'interruption': 1,
    'no stress': 2,
    'time pressure': 3
}

df = df.drop(columns=columns_to_remove)

df['condition'] = df['condition'].replace(condition_mapping)

test_data = pd.read_csv(args.input_file)

test_data = test_data.drop(columns=columns_to_remove)

test_data['condition'] = test_data['condition'].replace(condition_mapping)

features = ['VLF', 'VLF_PCT', 'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT', 'HF_NU',
       'TP', 'LF_HF', 'HF_LF', 'SD1', 'SD2', 'sampen', 'higuci', 'condition',
       'MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD',
       'pNN25', 'pNN50', 'KURT', 'SKEW', 'MEAN_REL_RR', 'MEDIAN_REL_RR',
       'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR',
       'KURT_REL_RR', 'SKEW_REL_RR']

target = 'HR'


X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)


degree = 3  
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(test_data[features])


model = LinearRegression()


model.fit(X_train_poly, y_train)


predictions = model.predict(X_test_poly)
results_df = pd.DataFrame({'uuid': test_data['uuid'], 'HR': predictions})

# Save the results to a CSV file
results_df.to_csv("results.csv", index=False)
