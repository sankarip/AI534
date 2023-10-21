#import pandas
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
#part 1
#add the file name
file_name='C:/Users/15418/Downloads/hw1-data/hw1-data/income.train.txt.5k'
#load the training data
headers=["age", "sector", "education", "marital-status", "occupation", "race", "sex", "hours-per-week", "country-of-origin", "target"]
data=pd.read_csv(file_name,names=headers)
#print(data.iloc[1])
#get the income column
income=data["target"]
#start counting the number of incomes more than or equal to 50k
count=0
for i in range(len(income)):
    if str(income[i])==' >50K':
        count=count+1
#calculate training data percent
trainingPercent=count/len(income)
print("training data percentage positive: ",trainingPercent*100)
#load the dev data
file_name='C:/Users/15418/Downloads/hw1-data/hw1-data/income.dev.txt'
devData=pd.read_csv(file_name,names=headers)
#get income and calculate percent positive, same as above
income=devData["target"]
count=0
for i in range(len(income)):
    if str(income[i])==' >50K':
        count=count+1
#calculate training data percent
devPercent=count/len(income)
print("dev data percentage positive: ",devPercent*100)
#get ages from training data
ages=data["age"]
#get max and min
minAge=min(ages)
maxAge=max(ages)
print("min age: ", minAge)
print("max age: ", maxAge)
#get hours worked
hoursWorked=data["hours-per-week"]
#get max and min
maxHours=max(hoursWorked)
minHours=min(hoursWorked)
print("most hours worked: ",maxHours)
print("least hours worked: ", minHours)
#getting unique values for each category
dimensions=0
for i in range(len(headers)):
    uniqueVals=data[headers[i]].unique()
    if headers[i]=="age" or headers[i]=="hours-per-week":
        #set the number of unqiue values in the column to 2
        numVals=1
    else:
        numVals=len(uniqueVals)
    dimensions=dimensions+numVals
#offset to get rid of 'target' column
dimensions =dimensions-2
print("dimensions: ", dimensions)

#part 2
toyFilePath='C:/Users/15418/Downloads/hw1-data/hw1-data/toy.txt'
toyData = pd.read_csv(toyFilePath, sep=", ", names=["age", "sector"])
encoded_data = pd.get_dummies(toyData, columns=["age", "sector"])
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
#fit to first dataset
encoder.fit(toyData)
binary_data = encoder.transform(toyData)
#binarizing full dataset
encoder.fit(data)
binary_data = encoder.transform(data)
dev_binary_data=encoder.transform(devData)
print("number of dimensions: ", len(binary_data[0]))

#X = [[0], [1], [2], [3]]
#y = [0, 0, 1, 1]
#neigh = KNeighborsClassifier(n_neighbors=1)
#neigh.fit(X, y)
#print(neigh.predict([[1.5]]))


neigh = KNeighborsClassifier(n_neighbors=1)
#neigh.fit(binary_data[:, 0:229], binary_data[:, 230])
x=len(binary_data)
neigh.fit(binary_data[0:x, 0:229], binary_data[0:x, 231])
#print(encoder.get_feature_names_out())

#get indices of the ones that miss and train on just them

# correctCount = 0
# misses=[]
# for i in range(0,x):#range(len(binary_data)):
#     prediction=neigh.predict([binary_data[i,0:229]])
#     #print(neigh.predict_proba([binary_data[i, 0:229]]))
#     if binary_data[i,231]==prediction:
#         correctCount=correctCount+1
#     else:
#         misses.append(i)
# print(misses)


#testing=data[(data['sector']==" Private") & (data['age']==33) & (data['race']==" White") & (data['hours-per-week']==40) & (data['education']==' HS-grad') & (data['sex']== ' Female') & (data['marital-status']==' Married-civ-spouse') & (data['occupation']== ' Craft-repair')]
#for i in range(len(testing)):
 #   print(testing.iloc[i])
#missed230=[314,591,807,1274,1791,1888,1890,1953,1962,2011,2142,2157,2178,2435,2454,2519,2778,2842,2848,2887,2953,3019,3026,3181,3232,3234,3260,3299,3327,3347,3373,3417,3434,3496,3520,3540,3589,3601,3688,3802,3863,3949,3966,4039,4058,4080,4155,4162,4177]

#percentCorrect = correctCount / x#len(binary_data)

#print(percentCorrect)


# for k in range(1,101,2):
#     neigh = KNeighborsClassifier(n_neighbors=k)
#     neigh.fit(binary_data[:, 0:229], binary_data[:, 231])
#     correctCount = 0
#     devCorrectCount = 0
#     for i in range(len(binary_data)):
#         prediction=neigh.predict([binary_data[i,0:229]])
#         if binary_data[i,231]==prediction:
#             correctCount=correctCount+1
#     for i in range(len(dev_binary_data)):
#         devPrediction=neigh.predict([dev_binary_data[i,0:229]])
#         if dev_binary_data[i,231]==devPrediction:
#             devCorrectCount=devCorrectCount+1
#     devPercentCorrect=devCorrectCount/len(dev_binary_data)
#     percentCorrect=correctCount/len(binary_data)
#     print("k=",k," training error rate: ", round((1-percentCorrect)*100,1), " (+:", round(percentCorrect*100,1), ")", " dev error rate: ", round((1-devPercentCorrect)*100,1), " (+:", round(devPercentCorrect*100,1), ")")

def problem3q1(data,devData):
    num_processor = 'passthrough'  # i.e.,notransformation
    cat_processor=OneHotEncoder(sparse=False,handle_unknown='ignore')
    preprocessor=ColumnTransformer([('num',num_processor,['age','hours-per-week']), ('cat',cat_processor,['sector',"education", "marital-status", "occupation", "race", "sex",  "country-of-origin", "target"]) ])
    preprocessor.fit(data)
    processed_data=preprocessor.transform(data)
    dev_processed_data=preprocessor.transform(devData)
    for k in range(1,101,2):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(processed_data[:, 0:91], processed_data[:, 93])
        correctCount = 0
        devCorrectCount = 0
        for i in range(len(processed_data)):
            prediction=neigh.predict([processed_data[i,0:91]])
            if processed_data[i,93]==prediction:
                correctCount=correctCount+1
        for i in range(len(dev_processed_data)):
            devPrediction=neigh.predict([dev_processed_data[i,0:91]])
            if dev_processed_data[i,93]==devPrediction:
                devCorrectCount=devCorrectCount+1
        devPercentCorrect=devCorrectCount/len(dev_processed_data)
        percentCorrect=correctCount/len(binary_data)
        print("k=",k," training error rate: ", round((1-percentCorrect)*100,1), " (+:", round(percentCorrect*100,1), ")", " dev error rate: ", round((1-devPercentCorrect)*100,1), " (+:", round(devPercentCorrect*100,1), ")")

def problem3q2(data,devData):
    num_processor = MinMaxScaler(feature_range=(0, 2))
    cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')
    preprocessor=ColumnTransformer([('num',num_processor,['age','hours-per-week']), ('cat',cat_processor,['sector',"education", "marital-status", "occupation", "race", "sex",  "country-of-origin", "target"]) ])
    preprocessor.fit(data)
    processed_data=preprocessor.transform(data)
    dev_processed_data=preprocessor.transform(devData)
    for k in range(1,101,2):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(processed_data[:, 0:91], processed_data[:, 93])
        correctCount = 0
        devCorrectCount = 0
        for i in range(len(processed_data)):
            prediction=neigh.predict([processed_data[i,0:91]])
            if processed_data[i,93]==prediction:
                correctCount=correctCount+1
        for i in range(len(dev_processed_data)):
            devPrediction=neigh.predict([dev_processed_data[i,0:91]])
            if dev_processed_data[i,93]==devPrediction:
                devCorrectCount=devCorrectCount+1
        devPercentCorrect=devCorrectCount/len(dev_processed_data)
        percentCorrect=correctCount/len(binary_data)
        print("k=",k," training error rate: ", round((1-percentCorrect)*100,1), " (+:", round(percentCorrect*100,1), ")", " dev error rate: ", round((1-devPercentCorrect)*100,1), " (+:", round(devPercentCorrect*100,1), ")")

def personalKNN(trainData, k, devData):
    num_processor = MinMaxScaler(feature_range=(0, 2))
    cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')
    preprocessor = ColumnTransformer([('num', num_processor, ['age', 'hours-per-week']), ('cat', cat_processor,['sector', "education","marital-status","occupation", "race", "sex","country-of-origin"])])
    preprocessor.fit(trainData)
    dev_processed_data = preprocessor.transform(devData)
    processed_data = preprocessor.transform(trainData)
    A=np.array(processed_data[:,0:len(processed_data[0])])
    p=np.array(dev_processed_data[:,0:len(processed_data[0])][0])
    distances=A-p
    eucDistances=np.linalg.norm(distances, axis=1)
    manDistances=np.sum(np.absolute(distances), axis=1)
    #I am sorry I don't know how to do this without sorting
    #add the indice of each value to new array
    newVals=[]
    for i in range(len(manDistances)):
        newVals.append([i,manDistances[i]])
    newVals=np.array(newVals)
    sortedMan=(newVals[newVals[:, 1].argsort()])
    nearestMan=sortedMan[0:k]
    print("Manhattan Indices and Distances: ", nearestMan)
    for i in range(len(nearestMan)):
        #print(data.iloc[int(nearestMan[i,0])])
        #print(nearestMan[i])
        #print(int(nearestMan[i,0]))
        pass
    #do for euc

    ####for some reason euc and man distances are ending up the same
    newValseuc=[]
    for i in range(len(eucDistances)):
        newValseuc.append([i,eucDistances[i]])
    newValseuc=np.array(newValseuc)
    sortedEuc=(newValseuc[newValseuc[:, 1].argsort()])
    nearestEuc=sortedEuc[0:k]
    print("Euclidean Indices and Distances: ", nearestEuc)
    for i in range(len(nearestEuc)):
        #print(data.iloc[int(nearestEuc[i,0])])
        #print(nearestEuc[i])
        #print(int(nearestEuc[i,0]))
        pass

#num_processor = MinMaxScaler(feature_range=(0, 2))
#cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')
#preprocessor = ColumnTransformer([('num', num_processor, ['age', 'hours-per-week']), ('cat', cat_processor,['sector', "education","marital-status","occupation", "race", "sex","country-of-origin","target"])])



#personalKNN(data,3,processed_data[0,0:91])
personalKNN(data,3,devData)


