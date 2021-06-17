# Logistic Regression
Logistic Regression is a classification algorithms, which will tell you the probability that an instance belong to this class or not with some threshold. If our probability is greater than 0.5 or 50 % then we will say this belongs to the class, means positive_label 1 ot if lesser we denote negative class or vice versa 0.
Logistic regression is used for solving the classification problems.
  * In Logistic regression, instead of fitting a regression line, we fit an "S" shaped logistic function, which predicts two maximum values (0 or 1)
  * In Linear regression , the predicted value Y exceeds 0 and 1 range, whereas in Logistic Regression ,the predicted value Y lies with in 0 and 1 range. 
 
  <img src="https://miro.medium.com/max/4640/1*dm6ZaX5fuSmuVvM4Ds-vcg.jpeg" width="500">
  
## Logistic Regression from Scratch
Logistic Regression is implemented without using scikit learn library and the dataset we have used is breast cancer ,a dataset taken from scikit learn library where we will be using the following functions to predict the accuracy of the given model
  * `Hypothesis Function` --->  It does the same work that as in Linear regression ,Logistic regression computes the features weights and the bias term and multiply with the respected features and sum them up.
  * `Cost Function` --->  The idea of cost function is that we count the sum of the metric distances between our hypothesis and real labels on the training data. The more optimized our parameters are, the less is the distance between the true value and hypothesis.
    * when the model predicts 1 and the label  y  is also 1, the loss for that training example is 0.
    * Similarly, when the model predicts 0 and the actual label is also 0, the loss for that training example is 0.
    * However, when the model prediction is close to 1and the label is 0, the second term of the log loss becomes a large. 
  * `Gradient Descent` ---> To update our weights  Θ  we apply our gradient descent, to improve our weights at every iteration. We take out the partial derviative of our cost function, or in other words, how much the cost function will change if we change  Θ  little bit.
### Basic Formulas:
* *hypothesis function :* h(x)=Θ0∗x0+Θ1∗x1+Θ2∗x2 +Θ3∗x3+Θn∗xn \
* *Cost Function :* J(θ)=−1/m(m∑i=1y(i)log(h(z(θ)(i)))+(1−y(i))log(1−h(z(θ)(i))))
    * m  is the number of training examples
    * y(i) is the actual label of the i-th training example.
    * h(z(θ)(i)) is the model's prediction for the i-th training example. 
* *Gradient Descent :* ∇θjJ(θ)=1/m(m∑i=1((x(i)−y(i))xj)
