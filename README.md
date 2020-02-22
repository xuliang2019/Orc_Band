# Orc Band!

<div align=center> <img src="https://github.com/xuliang2019/Orc_Band/blob/master/docs/Logo.jpg" width="400"> </div>

## Project Introduction
Our overall goal of the project is to predict the bandgap of organic semiconductors with Machine Leaning Methods. The dataset that we use is Harvard Clean Energy Project Database. To achieve this goal, Our tasks are:
1. Calculate molecular descriptors for organic semiconductor from SMILES strings;
2. Determine predictors for mathine learning mehthod by LASSO regression;
3. Screen and optimize regression model;
4. Build a wrapping function that help user to use our model.


### Calculation by RDkit
[RDkit](https://www.rdkit.org/) is a very useful and opensource package which can be download very easily. By using the map calculation in the package, we can easily get thousands of descriptors from the SMILES strings. And use several methods to screen the predictors.

### Machine Learning Model
For all the regression models we choosed, 75% of the data are used to train the model and 25% are used to test the model. By choosing the model, we randomly choose a couple of small size of data to run it several times and calculate the average statistic data.
#### Multiple Linear Regression
Import Linear Regression by using
```
from sklearn.linear_model import LinearRegression
```
The score of this model is 0.59.
#### Polynominal Regression
Import Polynominal Regression by using
```
from sklearn.preprocessing import PolynomialFeatures
```
The score of this model is 0.54.
#### Random Forest regression
Import Random Forest Regression by using
```
from sklearn.ensemble import RandomForestRegressor
```
The score of this model is 0.67.
#### Neural Network
Import Keras by using
```
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
```
The score of this model is 0.57.

*Note: Tensor Flow Needed!

#### Prediction Model
The Scatter Figure of the Predicted Bandgap for 4 Models are as follows.
<div align=center>
<img src="https://github.com/xuliang2019/Orc_Band/blob/master/docs/Model_Comparison.png" width="700">
</div>
The calculated statistic data are as follows:

<br>

| Error | Multiple Linear | Random Forest | Polynomial  | Neural Network |
| :---: | :-------------: | :-----------: | :---------: | :------------: |
| MSE   | 0.0450     | 0.0357   | 0.0503 | 0.1732    |
| MAE   | 0.1659     | 0.1425   | 0.1728 | 0.3349    |
| MAPE  | 0.0928     | 0.0792   | 0.0959 | 0.1883    |
| $R^2$ | 0.5933     | 0.6772   | 0.5458 | 0.5665    |
| Kfold | 0.5906     | 0.6819   | 0.5906 | -0.0620   |


According to the figure and the table above, we choose the Random Forest Regression as our Prediction Model. And by optimizing it, we have a really good model, which has $R^2=0.80$. (For whole data set)

<div align=center>
<img src="https://github.com/xuliang2019/Orc_Band/blob/master/docs/Optimized_Random_Forest.png" width="300">
</div>

## Installation
### 1.Rdkit

    $ conda create -c rdkit -n my-rdkit-env rdkit

This is recommended and we install this in this way.
Here is the link for [rdkit](https://www.rdkit.org/docs/Install.html)
Note: Rdkit package muse be installed in your computer, and you may need to download it manually by yourself

### 2.Orcband

    1. $ pip install git+https://github.com/xuliang2019/Orc_Band.git

### 2.Orcband Instruction   
  
<div align=center> <img src="https://github.com/xuliang2019/Orc_Band/blob/master/docs/orc_band.gif" width="400"> </div>
  *Note: The data we use is confidential, so you may be not able to run our jupyter notebook directly in each file.
