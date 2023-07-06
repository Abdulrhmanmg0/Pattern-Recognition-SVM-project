# upload file
from google.colab import files
uploaded = files.upload()

# import all libraries
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score

# read data
df = pd.read_csv('/content/cell_samples (1).csv')

# data information
print(df.head())
print(df.describe())
print(df.isnull().sum())

df.dtypes

df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
df['BareNuc'] = df['BareNuc'].astype('int')
df.dtypes

# split the data into features and target
feature_df = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)

df['Class'] = df['Class'].astype('int')
y = np.asarray(df['Class'])

# discover data set with ploting

fig, axs = plt.subplots(3, 3)
fig.suptitle('Histogram for each feature')
c = 0
for i in range(3):
  for j in range(3):
    axs[i, j].hist(X[:,c],edgecolor='black')
    axs[i, j].set_title(df.columns[c+1])
    c+=1

for ax in axs.flat:
    ax.label_outer()

fig, axs = plt.subplots(3, 3)
fig.suptitle('Scatter plot for each feature respect to the target 2 , 4')
c = 0
for i in range(3):
  for j in range(3):
    axs[i, j].scatter(X[:,c],y,edgecolor='black')
    axs[i, j].set_title(df.columns[c+1])
    c+=1

for ax in axs.flat:
    ax.label_outer()

fig, axs = plt.subplots(3, 3)
fig.suptitle('Boxplot plot for each feature')
c = 0
for i in range(3):
  for j in range(3):
    axs[i, j].boxplot(X[:,c])
    axs[i, j].set_title(df.columns[c+1])
    c+=1

for ax in axs.flat:
    ax.label_outer()

Corr = feature_df.corr()
plt.matshow(Corr)
plt.title('Correlation Matrix', fontsize=16)
plt.colorbar()
plt.show()

sns.pairplot(df, hue='Class')
plt.show()

g = sns.pairplot(feature_df, diag_kind="kde")
g.map_lower(sns.kdeplot, levels=4, color=".2")

# split into training and testing
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=32)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# fit SVM model with our training set and evaluate it with testing set
model = SVC(kernel="sigmoid", C=100)
model.fit(X_train,y_train)
print("Accuracy in Training : ",model.score(X_train,y_train))
print("Accuracy in Testing : ",model.score(X_test,y_test))

# Hyperparameter tuning using GridSearchCV
param_grid = {'C':[0.01,0.1,0,1,10,100],
              'kernel':['rbf','poly','sigmoid','linear'],
              'degree':[1,2,3,4,5]}
grid = GridSearchCV(SVC(),param_grid)
grid.fit(X_train,y_train)

print(grid.best_params_)
print(grid.score(X_test,y_test))

yhat = grid.predict(X_test)

# Evaluation metrecis
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix unnormalized')

cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= True,  title='Confusion matrix normalized')

jaccard_score(y_test, yhat,pos_label=2)