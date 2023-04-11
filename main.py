import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data') #get the file
df.replace('?', -99999, inplace=True) #modifies the dataframe to get rid of question marks (missing features), and to instead place an outlier
#in real world dataframes there is often a lot missing so you don't want to sacrifice that

#we want to get rid of useless data as well as outliers
df.drop(['id'], 1, inplace=True) #dropping the id column, brackets are needed to specify

X = np.array(df.drop(['class'], 1)) #everything except for the class column
y = np.array(df['class']) #literally only the class column

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2) #takes the Xs, and the Ys, splits it up so 80% are the training, and 20% is testing, creates lists for all
clf = neighbors.KNeighborsClassifier() #creates a k neighbors classifier model
clf.fit(X_train, y_train) #fits the training data in the model

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 2, 2, 1, 2, 3, 2, 1]]) #lists of two rando examples
example_measures = example_measures.reshape(len(example_measures), -1) #automatically predict on any number of examples, reshape the array

prediction = clf.predict(example_measures)
print(prediction)

