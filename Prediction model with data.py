import pandas as pd
import numpy as np
file_address = "/Users/neelesh/Downloads/a.xlsx"
logisticregression_data = pd.read_excel(file_address, "Sheet1", index_col=None, usecols=("I, J, Q, V"))
logisticregression_data = logisticregression_data.drop([0, 1])
logisticregression_data.columns = ["industry", "household_type", "mpce", "mot_cycle"]
logisticregression_data["householdmpce_rank"] = logisticregression_data["mpce"].rank(pct=True)
new_data = pd.DataFrame({"Agriculture": [1], 21: [0], "Trade": [0], "Manufacturing": [0], "Transport": [0], "Construction": [0], "Other service activities": [0], "Public administration&defence": [0], "Real estate activities": [0], "Health & social work": [0], "Mining & Quarrying": [0], "Education": [0], "Information & communication": [0], "Electricity,Gas & Water Supply": [0], "Financial & Insurance": [0], "Professional&technical activities": [0], "Accommodation &Food Services": [0], "Administrative&support services": [0], "RurSE_agri": [1], "Rur_casagri": [0], "Rur_oth": [0], "RurSE_nagri": [0], "Rur_regwage": [0], "Rur_casnagri": [0], "third": [0], "second": [0], "first": [1], "Mot": [0]})
for row in logisticregression_data.itertuples():
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
household_characteristics.columns = household_characteristics.columns.astype(str)

from sklearn.model_selection import train_test_split
household_characteristicstrain, household_characteristicstest, mot_cyclepurchasetrain, mot_cyclepurchasetest = train_test_split(household_characteristics, mot_cyclepurchase, test_size=0.20) #end of splitting data into training data and testing data


from sklearn.linear_model import LogisticRegression
logisticregression = LogisticRegression()
logisticregression.fit(household_characteristicstrain, mot_cyclepurchasetrain) #end of logistic regression

from sklearn import metrics
mot_cyclepurchasepred = logisticregression.predict(household_characteristicstest)
confusion_matrix = metrics.confusion_matrix(mot_cyclepurchasetest, mot_cyclepurchasepred)
print(f"Prediction accuracy: {((confusion_matrix[0][0] + confusion_matrix[1][1])/(confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])) * 100}%")
probabilities = logisticregression.predict_proba(household_characteristicstest)
for i, prob in enumerate(probabilities):
    print(f"Household {i+1}: Probability household does not purchase motorcycle = {prob[0]:.4f}, Probability household purchases motorcycle = {prob[1]:.4f}") #end of model evaluation

linearregression_data = pd.read_excel(file_address, "Sheet1", index_col=None, usecols=("Q, X"))
linearregression_data = linearregression_data.drop([0, 1])
mpce = []
value = []
for row in linearregression_data.itertuples():
    if row[2] == 0:
        linearregression_data = linearregression_data.drop(row[0])
for row in linearregression_data.itertuples():
    mpce.append(row[0])
    value.append(row[1])
mpce = np.reshape(mpce, [-1, 1]) #end of data preperation

from sklearn import linear_model as lmod
regressor = lmod.LinearRegression().fit(mpce, value)
coefficient = lmod.LinearRegression().fit(mpce, value).coef_ #end of linear regression

from sklearn.metrics import mean_squared_error, r2_score

value_predicted = regressor.predict(mpce)

print(f"Mean squared error: {mean_squared_error(value, value_predicted):.2f}")
print(f"Coefficient of determination: {r2_score(value, value_predicted):.2f}") #end of model evaluation

import matplotlib.pyplot as plt
plt.scatter(mpce, value, label = "training data data points")
plt.plot(mpce, regressor.predict(mpce), "r", label = "regression line")
plt.title(f"Regression model, Mean squared error: {mean_squared_error(value, value_predicted):.2f}, Coefficient of determination: {r2_score(value, value_predicted):.2f}")
plt.legend()
plt.show() #end of plotting data