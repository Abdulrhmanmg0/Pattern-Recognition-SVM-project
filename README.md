# ***Pattern-Recognition-SVM-project***
 
 <p>For this Project on<em>Pattern-Recognition Course</em>, we were able to select any machine learning algorithm regardless of whether it was covered in the course or not, so I decided to select SVM ( <em>support vector machine</em> ) which wasn't included in our course material.<br>I will explain everything I learned by myself in this project :</p>

## How SVM work ?
SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the data are not otherwise linearly separable.

for example:
when the data isn't linearly separable we use kernels to transform the data into a form that will make them separated .

## What are the Major Kernels ?
### 1- Radial Basis Function (RBF) :
known as Gaussian Kernel Radial Basis Function.

<img align="left" src=https://github.com/Abdulrhmanmg0/OOP2-Project-Java/assets/93158698/e45db40a-98de-4b66-830c-9db77f5f16c8 height = 100>
<br><br><br><br><br>

### 2- Sigmoid Function.

<img align="left" src=https://github.com/Abdulrhmanmg0/OOP2-Project-Java/assets/93158698/ddfd023a-eba3-4dfe-adc5-5a4e025c4a40 height = 100>
<br><br><br><br><br>

### 3- Polynomial Function.

<img align="left" src=https://github.com/Abdulrhmanmg0/OOP2-Project-Java/assets/93158698/610a4e01-7f9a-4e0a-a6cc-757c244fd934 height = 100 width=120>
<br><br><br><br><br>

### 4- Linear.

## What are the parameters of SVM ?
their are many parameters in SVM but I will only talk about what I have used.
<br>

### 1- C 
Regularization parameter indicate the penalty by default (1).

### 2- kernel
Specifies the kernel type ( linear , poly , rbf , sigmoid ) by default ( rbf ).

### 3- degree
Degree of the polynomial kernel function , It will be ignored in any other function other than poly by default (3).

## How to choose the best parameters ?
by using a process called hyperparameter tuning, their is many methods for hyperparameter tuning. 
<br>
I used the GridSearchCV. <br>
GridSearchCV : is an exhaustive search over specified parameter values which can specified by the user using a dictionary .
<br>
for example : 

```
param_grid={
    'C':[0.01,0.1,1,10,100],
    'kernal':['rbf','poly','sigmoid','linear'],
    'degree':[1,2,3,4,5,6]
    }
```
we specified the parameter and the values to be tested after running the GridSearchCV we can return the best parameters to use it .

## How to evaluate ? 
Evaluation metrics are quantitative measures used to assess the performance of machine learning model , Its important to use more than one metrics because the model might perform a good performance in one metrics and poorly in the other metrics.
<br>
I used the following metrics : <br>

## 1- Precision.
## 2- Recall.
## 3- F1-score.
## 4- Jaccard.
