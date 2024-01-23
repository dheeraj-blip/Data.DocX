# Data.DocX
In the Jupyter notebook, we have tried multiple machine learning models like linear regression, ridge regression, lasso regression, RANSAC regression, XGBoost regression, KNN regression and many more, by using 80% of the dataset as train data, and remaining 20% as test data.<br><br>
We found out that Polynomial regression of degree 3 has the lowest RMSE (Root Mean Squared Error), compared to all the other models, hence, we are using the same for final prediction.<br><br><br>
In the 'run.py' file, we have used the 'argparse' python library, to take the path of the test dataset as command line argument. We just have to run the below code in the command prompt:
```
python run.py --input_file path\to\test_filename.csv
```
This will store the test dataset as a pandas data frame and then, this is further used in the trained model, to generate the Accurate Heart rate output, which is stored as a CSV file named 'results.csv', in the same working directory as the 'run.py' file.
<br>
## Instructions for Installations
The zip file consists run.py and train_data.csv. The test_data.csv should be added to this unzipped folder.<br><br>
**Step 1**: Unzip it. Open Command Prompt and chage your current directory to the unzipped folder.<br>
**Step 2**: Use the below command in the command prompt:
```
python run.py --input_file path\to\test_filename.csv
```
**Step 3**: The results.csv will be generated in the same folder. It will contain the Predicted Heart Rate.
## Team Members
Talasila Dheeraj : +91 9494710420<br><br>
Skanda P R : +91 8792844101<br><br>
Suhas Raj H R : +91 8073299734<br><br>
