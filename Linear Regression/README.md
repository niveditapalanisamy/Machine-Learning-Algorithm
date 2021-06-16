# Linear Regression #
Linear Regression is a supervised learning algorithm, which works best for continuous data. Linear Regression is most common and one of the
powerful algorithm. <br/>
* Suppose If we have a Scatter data,<br/>
* We will fit the straight line and make predictions.<br/>
* So, above we fitted the straight line on our inputs and we are predicting the output using x and y.<br/>
<img src="https://www.w3schools.com/python/img_matplotlib_scatter.png" width="400"><img src="https://www.w3schools.com/python/img_linear_regression.png" width="400">
## Linear Regression from scratch ##
Linear Regression is implemented without using scikit learn library where we will be using the following functions to predict the accuracy of the given model
  *  `Hypothesis fuction` ---> A hypothesis is a function that best describes the target in supervised machine learning. The hypothesis that an algorithm would come up depends upon the data and also depends upon the restrictions and bias that we have imposed on the data. 
  *  `Cost Function` ---> It is a function that measures the performance of a Machine Learning model for given data. Cost Function quantifies the error between predicted values and expected values and presents it in the form of a single real number.The goal is to then find a set of weights and biases that minimizes the cost.So, we have choose our Θ0 and Θ1 but who gaurantees that our parameters is the best parameter, so for checking the accuracy of our hypothesis or our parameters, we have cost function. Cost function is taking out the distance between actual and predicted values by subtracting h(xi) − yi and at last we are squaring them up. 
  *  `Gradient Descent` ---> Gradient Descent runs iteratively to find the optimal values of the parameters corresponding to the minimum value of the given cost function.It is basically used for updating the parameters of the learning model. 
  *  `Learning rate` ---> If you choose very small learning rate, it will be very very slow and it will never converge to the local minimum. If you choose very large learning rate, your model might diverge and never converge to the local minimum. 
### Basic Formulas: 
 * *Hypothesis Function :* h(x)=Θ0∗x0+Θ1∗x1+Θ2∗x2 +Θ3∗x3+Θn∗xn
 *  *Cost Function :* MSE(θ0, θ1, θ2, θ3, . . . , θn) =1/m +mΣi=1(Θ^T.xi − yi)2
 *  *Gradient Descent :* ∂/∂Θj J(Θ) =2/m +mΣi=1(ΘTxi − yi)2
