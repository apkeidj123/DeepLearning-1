# DeepLearning-1
Regression and Classification

1. Linear Regression
* Split train.csv into training set (80%) and test set (20%).Both the training and test set should be normalized by subtracting the (column-wise) means of training set from them and then divided by the (column-wise) standard deviations of the training set.
* Implement a linear regression model without the bias term to predict G3. Use pseudo-inverse to obtain the weights. Record the root mean squared error (RMSE) of the test set.
* Regularization is often adopted to avoid over-fitting. Implement a regularized linear regression model without the bias term where lambda = 1.0. 
* Implement a Bayesian linear regression model with the bias term.
* Plot the ground truth (real G3) versus all predicted values generated.
![image](https://github.com/apkeidj123/DeepLearning-1/blob/master/PIC/2019-03-28_231703.png)
![image](https://github.com/apkeidj123/DeepLearning-1/blob/master/PIC/2019-03-28_231740.png)
![image](https://github.com/apkeidj123/DeepLearning-1/blob/master/PIC/2019-03-28_231914.png)

2. Classification
* Create a new column to indicate whether G3 is greater or equal to 10 (1 if this event is true; 0 if this event is not true) to serve as the labels for classification. Implement a linear regression model with regularization (lambda = 1:0) and the bias term to predict the labels. Record the classification results when thresholds are set to 0.1, 0.5, and 0.9. Note that samples with model activations greater than the threshold are classified as class 1, and class 0 otherwise.
* Repeat but use logistic regression.
* Plot confusion matrices
![image](https://github.com/apkeidj123/DeepLearning-1/blob/master/PIC/2019-03-28_233100.png)
![image](https://github.com/apkeidj123/DeepLearning-1/blob/master/PIC/2019-03-28_233110.png)
