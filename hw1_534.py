#import pandas
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
#part 1
#add the file name
file_name='C:/Users/15418/Downloads/hw1-data/hw1-data/income.train.txt.5k'
#load the training data
headers=["age", "sector", "education", "marital-status", "occupation", "race", "sex", "hours-per-week", "country-of-origin", "target"]
data=pd.read_csv(file_name,names=headers)
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
x=300
neigh.fit(binary_data[0:x, 0:229], binary_data[0:x, 230])
#print(encoder.get_feature_names_out())

#get indices of the ones that miss and train on just them

correctCount = 0
for i in range(0,x):#range(len(binary_data)):
    prediction=neigh.predict([binary_data[i,0:229]])
    #print(neigh.predict_proba([binary_data[i, 0:229]]))
    if binary_data[i,230]==prediction:
        correctCount=correctCount+1

percentCorrect = correctCount / x#len(binary_data)

print(percentCorrect)


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





