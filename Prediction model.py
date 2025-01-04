import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
file_address = "/Users/neelesh/Downloads/Haryana data.xlsx"
haryana_data = pd.read_excel(file_address, "Sheet1", index_col=None, usecols=("I, J, Q, V"))
haryana_data = haryana_data.drop([0, 1])
haryana_data.columns = ["industry", "household_type", "mpce", "mot_cycle"]
haryana_data["householdmpce_rank"] = haryana_data["mpce"].rank(pct=True)
new_data = pd.DataFrame({"Agriculture": [1], 21: [0], "Trade": [0], "Manufacturing": [0], "Transport": [0], "Construction": [0], "Other service activities": [0], "Public administration&defence": [0], "Real estate activities": [0], "Health & social work": [0], "Mining & Quarrying": [0], "Education": [0], "Information & communication": [0], "Electricity,Gas & Water Supply": [0], "Financial & Insurance": [0], "Professional&technical activities": [0], "Accommodation &Food Services": [0], "Administrative&support services": [0], "RurSE_agri": [1], "Rur_casagri": [0], "Rur_oth": [0], "RurSE_nagri": [0], "Rur_regwage": [0], "Rur_casnagri": [0], "third": [0], "second": [0], "first": [1], "Mot": [0]})
for row in haryana_data.itertuples():
    data = {"Agriculture": [0], 21: [0], "Trade": [0], "Manufacturing": [0], "Transport": [0], "Construction": [0],
            "Other service activities": [0], "Public administration&defence": [0], "Real estate activities": [0],
            "Health & social work": [0], "Mining & Quarrying": [0], "Education": [0],
            "Information & communication": [0], "Electricity,Gas & Water Supply": [0], "Financial & Insurance": [0],
            "Professional&technical activities": [0], "Accommodation &Food Services": [0],
            "Administrative&support services": [0], "RurSE_agri": [0], "Rur_casagri": [0], "Rur_oth": [0],
            "RurSE_nagri": [0], "Rur_regwage": [0], "Rur_casnagri": [0], "third": [0], "second": [0], "first": [0],
            "Mot": [0], row[1]: [1], row[2]: [1]}
    if row[5] <= 0.5:
        data["third"] = [1]
    elif row[5] <= 0.9:
        data["second"] = [1]
    elif row[5] <= 1:
        data["first"] = [1]
    data["Mot"] = row[4]
    dataframe = pd.DataFrame(data)
    new_data = pd.concat([new_data, dataframe], ignore_index = True)
new_data = new_data.drop([0])
household_characteristics = new_data[["Agriculture", 21, "Trade", "Manufacturing", "Transport", "Construction",
            "Other service activities", "Public administration&defence", "Real estate activities",
            "Health & social work", "Mining & Quarrying", "Education",
            "Information & communication", "Electricity,Gas & Water Supply", "Financial & Insurance",
            "Professional&technical activities", "Accommodation &Food Services",
            "Administrative&support services", "RurSE_agri", "Rur_casagri", "Rur_oth",
            "RurSE_nagri", "Rur_regwage", "Rur_casnagri", "third", "second", "first"]]
mot_cyclepurchase = new_data["Mot"]
household_characteristics.columns = household_characteristics.columns.astype(str) #end of data preperation

from sklearn.model_selection import train_test_split
household_characteristicstrain, household_characteristicstest, mot_cyclepurchasetrain, mot_cyclepurchasetest = train_test_split(household_characteristics, mot_cyclepurchase, test_size=0.20) #end of splitting data into training data and testing data

from sklearn.linear_model import LogisticRegression
logisticregression = LogisticRegression()
logisticregression.fit(household_characteristicstrain, mot_cyclepurchasetrain) #end of logistic regression

from sklearn import metrics
mot_cyclepurchasepred = logisticregression.predict(household_characteristicstest)
confusion_matrix = metrics.confusion_matrix(mot_cyclepurchasetest, mot_cyclepurchasepred)
print("Prediction of whether household purchases motorcycle or not")
print(f"Prediction accuracy: {((confusion_matrix[0][0] + confusion_matrix[1][1])/(confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])) * 100}%")
probabilities = logisticregression.predict_proba(household_characteristicstest)
for i, prob in enumerate(probabilities):
    print(f"Household {i+1}: Probability household does not purchase motorcycle = {prob[0]:.4f}, Probability household purchases motorcycle = {prob[1]:.4f}") #end of model evaluation
print("\r")

file_address = "/Users/neelesh/Downloads/Haryana data.xlsx"
haryana_data = pd.read_excel(file_address, "Sheet1", index_col=None, usecols=("Q, X"))
haryana_data = haryana_data.drop([0, 1])
for row in haryana_data.itertuples():
    if row[2] == 0:
        haryana_data = haryana_data.drop(row[0])
array1 = haryana_data.to_numpy()
mpce = []
value = []
for i in array1:
    mpce.append(i[0])
    value.append(i[1])
mpce = np.reshape(mpce, [-1, 1]) #end of data preperation

from sklearn.model_selection import train_test_split
mpce_train, mpce_test, value_train, value_test = train_test_split(mpce, value, test_size=0.20)

from sklearn import linear_model as lmod
regressor = lmod.LinearRegression().fit(mpce_train, value_train)
coefficient = lmod.LinearRegression().fit(mpce_train, value_train).coef_ #end of linear regression

from sklearn.metrics import mean_squared_error, r2_score
value_predicted = regressor.predict(mpce_test)
print("For those households which purchase motorcycles predicting value of motorcycle purchased")
print(f"Mean squared error: {mean_squared_error(value_test, value_predicted):.2f}")
print(f"Coefficient of determination: {r2_score(value_test, value_predicted):.2f}") #end of model evaluation

import matplotlib.pyplot as plt
plt.scatter(mpce_test, value_test, label = "test data data points")
plt.plot(mpce_test, regressor.predict(mpce_test), "r", label = "regression line")
plt.title(f"Regression model, Mean squared error: {mean_squared_error(value_test, value_predicted):.2f}, Coefficient of determination: {r2_score(value_test, value_predicted):.2f}")
plt.legend()
plt.show() #end of plotting data