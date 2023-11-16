import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

def loadData():
    file_name = 'C:/Users/psankari/Downloads/hw3-data/hw2-data/my_train.csv'
    # load the training data
    data = pd.read_csv(file_name)
    return data

def loadDevData():
    file_name = 'C:/Users/psankari/Downloads/hw3-data/hw2-data/my_dev.csv'
    # load the training data
    data = pd.read_csv(file_name)
    return data

def loadTestData():
    file_name = 'C:/Users/15418/Downloads/hw3-data/hw2-data/test.csv'
    # load the training data
    data = pd.read_csv(file_name)
    return data

def getFeatures():
    data=loadData()
    headers=data.columns
    #print(data)
    dimensions=0
    for i in range(len(headers)):
        uniqueVals=data[headers[i]].unique()
        #excluding the Ids and Sales prices as per the HW instructions
        if headers[i]=="Id" or headers[i]=='SalePrice':
            #set the number of unique values in the column to 2
            numVals=0
        else:
            numVals=len(uniqueVals)
        print(headers[i],': ', numVals)
        dimensions=dimensions+numVals
    print("number of features: ",dimensions)

def linearRegressionPart2():
    data=loadData()
    devData=loadDevData()
    headers=data.columns
    devHeaders=devData.columns
    #get targets
    targets=data['SalePrice']
    devTargets=devData['SalePrice']
    #get rid of the ID column
    data=data[headers[1:80]]
    devData=devData[devHeaders[1:80]]
    data = data.astype(str)
    devData = devData.astype(str)
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(data)
    binary_data = encoder.transform(data)
    #print(binary_data)
    dev_binary_data=encoder.transform(devData)
    #print(len(binary_data[0]))
    #trainingData=binary_data[:, 1:79]
    reg = LinearRegression().fit(binary_data, targets)
    #print(targets)
    #print(reg.predict(dev_binary_data[0].reshape(1,-1)))
    predictions=reg.predict(dev_binary_data)
    #start a count for RMSLE
    errorSum=0
    for i in range(len(devTargets)):
        target=devTargets.iloc[i]
        prediction=predictions[i]
        rmsle=(np.log(prediction+1)-np.log(target+1))**2
        errorSum=errorSum+rmsle
    finalRMSLE=(errorSum/len(devTargets))**0.5
    print('RMSLE: ', finalRMSLE)
    print((mean_squared_log_error(devTargets, predictions))**0.5)
    coefs=reg.coef_
    #empty array to store coefficients and indexes
    coefAndIndex=[]
    for i in range(len(coefs)):
        coefAndIndex.append([i, coefs[i]])
    coefAndIndex = np.array(coefAndIndex)
    sortedCoefs = (coefAndIndex[coefAndIndex[:, 1].argsort()])
    negatives=sortedCoefs[0:10]
    positives=sortedCoefs[len(sortedCoefs)-10:len(sortedCoefs)]
    features=encoder.get_feature_names_out()
    print('feature len dumb: ', len(features))
    #for i in range(len(positives)):
    #    print("positive: ", features[int(positives[i][0])])
    #for i in range(len(negatives)):
    #    print("negative: ", features[int(negatives[i][0])])
    #print('intercept: ',reg.intercept_)

def linRegOnTest():
    data=loadData()
    headers=data.columns
    #get targets
    targets=data['SalePrice']
    #get rid of the ID column
    data=data[headers[1:80]]
    testData = testData[headers[1:80]]
    data = data.astype(str)
    testData=testData.astype(str)
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(data)
    binary_data = encoder.transform(data)
    test_binary_data=encoder.transform(testData)
    reg = LinearRegression().fit(binary_data, targets)
    predictions=reg.predict(test_binary_data)
    results=['Id'+' ,'+ 'SalePrice\n']
    for i in range(len(testData)):
        id=IDs[i]
        prediction=predictions[i]
        results.append(str(id)+' , '+str(prediction)+'\n')
    print(results)
    with open('results.csv', 'w') as f:
        f.writelines(results)

def smartBinarization():
    devData=loadDevData()
    data=loadData()
    headers=data.columns
    #get targets
    targets=data['SalePrice']
    devTargets=devData['SalePrice']
    #print(devTargets)
    #get rid of ID and target column
    data=data[headers[1:80]]
    devData=devData[headers[1:80]]
    #changing the values in LotFrontage and GarageYrBlt and MasVnrArea
    data.loc[np.isnan(data["LotFrontage"]) == True, "LotFrontage"] = 0
    devData.loc[np.isnan(devData["LotFrontage"]) == True, "LotFrontage"] = 0
    data.loc[np.isnan(data["GarageYrBlt"]) == True, "GarageYrBlt"] = 1900
    devData.loc[np.isnan(devData["GarageYrBlt"]) == True, "GarageYrBlt"] = 1900
    data.loc[np.isnan(data["MasVnrArea"]) == True, "MasVnrArea"] = 0
    devData.loc[np.isnan(devData["MasVnrArea"]) == True, "MasVnrArea"] = 0
    #commented out, but allows checking that the nans have been changed
    #for i in range(len(data["LotFrontage"])):
    #    print(str(data['LotFrontage'][i])+' new')
    numeric=[]
    categorical=[]
    for i in range(1,80):
        #print(headers[i],type(data[headers[i]][0]).__name__)
        if type(data[headers[i]][0]).__name__=='int64' or type(data[headers[i]][0]).__name__=='float64':
            numeric.append(headers[i])
        else:
            categorical.append(headers[i])
    devData=devData.astype(str)
    data = data.astype(str)
    #print the categories to make sure they make sense
    print("numeric: ", numeric)
    print("categorical: ", categorical)
    num_processor = MinMaxScaler(feature_range=(0, 1))
    cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')
    preprocessor = ColumnTransformer([('num', num_processor, numeric), ('cat', cat_processor,categorical)])
    preprocessor.fit(data)
    features = preprocessor.get_feature_names_out()
    #print(features)
    binary_data=preprocessor.transform(data)
    dev_binary_data=preprocessor.transform(devData)
    #commented out, but lets you look at the binarized data side by side
    #for i in range(len(dev_binary_data[0])):
    #    print("train: ",binary_data[0][i], " dev:",dev_binary_data[0][i])
    reg = LinearRegression().fit(binary_data, targets)
    predictions=reg.predict(dev_binary_data)
    print(predictions)
    #look at most negative features to figure this out
    print(devData.iloc[34])
    #start a count for RMSLE
    errorSum=0
    for i in range(len(devTargets)):
        target=devTargets.iloc[i]
        prediction=predictions[i]
        #compare the targets and predictions
        #print(target, prediction)
        if prediction>0:
            rmsle=(np.log(prediction+1)-np.log(target+1))**2
            errorSum=errorSum+rmsle
    finalRMSLE=(errorSum/len(devTargets))**0.5
    print('RMSLE: ', finalRMSLE)
    coefs=reg.coef_
    #empty array to store coefficients and indexes
    coefAndIndex=[]
    for i in range(len(coefs)):
        coefAndIndex.append([i, coefs[i]])
    coefAndIndex = np.array(coefAndIndex)
    sortedCoefs = (coefAndIndex[coefAndIndex[:, 1].argsort()])
    negatives=sortedCoefs[0:10]
    positives=sortedCoefs[len(sortedCoefs)-10:len(sortedCoefs)]
    features=preprocessor.get_feature_names_out()
    #print(features)
    print('feature length smart binarization: ',len(features))
    #for i in range(len(positives)):
        #print("positive: ", features[int(positives[i][0])])
    #for i in range(len(negatives)):
        #print("negative: ", features[int(negatives[i][0])])
    #print('intercept: ',reg.intercept_)


#linearRegressionPart2()
smartBinarization()