This project is based on the detection of fraudulent bank transfers. After an exhaustive exploratory analysis of the data, we will apply different predefined machine learning algorithms and we will see their efficiency in our classification problem. Once they have been used and their theoretical basis explained, we will compare them and draw conclusions from each of them.

Each observation of our data is a bank transfer, so most of them will be non-fraudulent, which gives us unbalanced data and therefore we will perform a couple of different samples for each algorithm. Once applied, we will see the percentage of failure in each class of the response variable and in each type of sampling, which will give us an overall view of their effectiveness.

Due to the large number of observations available to us, apart from the algorithms mentioned above, we have considered it appropriate to complete our study by making use of deep learning. We will create a neural network that we will define as a sequential model, we will see if it is useful in this problem and we will compare the results with those obtained previously.


DESCRIPTION OF THE DATA SET AND ITS VARIABLES

This data has 594.643 rows, each of them representing a bank transaction. First, let’s see which are our variables. In this data set we have 10 variables, two of them are numeric and the rest are treated as factors.

•	Step: This variable represents the day on which the transaction begins. It has 180 steps, so the simulation is across 6 months.

•	Customer: The customer ID. The total number of customers is 4.112, so that each of them carries out many transactions. Only 1.483 of the customers have experienced fraud at least once.

•	zipCodeOrigin: The zip code.

•	Merchant: The merchant’s ID. Here we have 50 different merchants. Only 30 of them committed fraud.

•	zipMerchant: The merchant’s zip code.

•	Age: The customer’s age. We have seven levels in this variable, each of them (except the last one) representing an age range.

–	0 : ≤ 18
–	1 : 19 − 25
–	2 : 26 − 35
–	3 : 36 − 45
–	4 : 46 − 55
–	5 : 56 − 65
–	6 : > 65
–	U: Unknown

•	Gender: The gender of the customer.

–	E: Enterprise
–	F: Female
–	M: Male
–	U: Unknown

•	Category: The category of the purchase. We have 15 types of transactions:

–	Bars and Restaurants
–	Contents
–	Fashion
–	Food
–	Health
–	Home
–	Hotel services
–	Hyper
–	Leisure
–	Sport and Toys
–	Tech
–	Transportation
–	Travel
–	Wellness and Beauty
–	Other Services

•	Amount: The amount of money used in the transaction.

•	Fraud: The response variable which shows if the purchase is fraudulent or not.

–	1: Fraudulent
–	0: Non- fraudulent

Data Analysis

First, we realized that the data are unbalanced, as can be seen in the plot bellow. Only 1.21% (7.200) of the transactions are fraudulent, so we are going to have to employ sampling techniques to balance it out and apply machine learning algorithms correctly.


Fig.1 Fraud Countplot
 
 ![image](https://github.com/user-attachments/assets/f3d6bec3-f954-49c3-a97a-650bbdee0e7a)


The two techniques we will use later are Oversampling and Undersampling. Both have advantages and disadvantages. Undersampling reduces the number of observations of the majority class, i.e. it randomly selects 1.21% of non-fraudulent observations, so a lot of information is lost. On the other hand, Oversampling generates new fraudulent observations based on the data we already have. By applying this we do not lose information, but we nevertheless run the risk of overfitting the model, i.e. it works well on our data but not on different data, and overestimating performance.
Since both methods have disadvantages, we will first test our methods by doing only Undersampling, and then we will sample by combining Oversampling and Undersampling using a pipeline. For Oversampling we will use a technique called SMOTE (Synthetic Minority Over-sampling Technique). This technique will create new observations of the minority class using the neighbouring instances, so the new dummy transactions will not be exact copies but will be like the ones we already have.
Let us now look at the categories most affected by fraud. After looking at the table with the mean amount and the percentage of fraud, leisure and travel are the categories that suffer most from fraud. It is no coincidence that these are the categories where the most money is spent on average in transactions, the scammers choose the ones where people spend the most money on average. On the other hand, we can see that there are three categories in which no fraud is committed: Contents, Food and Transportation. Again, these three categories are three of the four where the least money is spent, and it seems that scammers do not find it useful to take the risk of committing fraud for so little reward.


TABLE 1. MEAN FEATURE VALUES PER CATEGORY

Category	              Amount	Fraud %
Bars and Restaurants	  43.46	  1.88%
Contents	              44.55	  0.00%
Fashion	                65.67	  1.80%
Food	                  37.07	  0.00%
Health	                135.62	10.51%
Home	                  165.67	15.21%
Hotel services	        205.614	31.42%
Hyper	                  45.97	  4.59%
Leisure	                288.91	94.99%
Sport and Toys	        215.72	49.53%
Tech	                  120.95	6.67%
Transportation	        26.96	  0.00%
Travel	                2250.41	79.40%
Wellness and Beauty	    65.51	  4.76%
Other Services	        135.88	25.00%


Fig 2. Boxplot for the amount spend in each category

![image](https://github.com/user-attachments/assets/35469987-38bf-4f82-a840-ebf4c275b1ca)
 
We are not going to include it here, but if we do the same table only considering the average amount when transactions are fruitful, we will see that the amount of money spent on such transactions on average per category is almost four times higher than the average amount per category including all transactions.
To continue, we will now look at which sector of the population is most affected by fraud, first by gender and then by the age of the individual. It seems that the sectors most affected are women and people aged 18 and under. Again, these two sectors of the population are the ones that spend the most money on banking transactions. We can also see how the percentage of fraud and the mean amount of money in customers with unknown age is the same as that of enterprises, so it is clear that the customers without age are precisely the enterprises.

TABLE 2. MEAN FEATURE VALUES PER AGE

Age	    Amount	Fraud %
≤ 18	  46.60	  1.96%
19 − 25	37.53  	1.19%
26 − 35	38.36	  1.25%
36 − 45	37.88	  1.19%
46 − 55	38.20	  1.29%
56 − 65	36.43	  1.10%
> 65	  36.87	  0.97%
Unknown	36.63	  0.59%

TABLE 3. MEAN FEATURE VALUES PER GENDER

Gender	    Amount	 Fraud %
Enterprise	36.63	   0.59%
Female	    39.21	   1.47%
Male	      36.31	   0.91%
Unknown	    31.51	   0.00%

Finally, by doing an scatter plot, we can see that the time at which the simulation started does not influence the frequency of frauds at all. Another thing to note is that most of the frauds are small amounts, almost all of them below 1.000. Specifically, only 607 of the 7.200 frauds have the amount above 1.000 dollars.
 
Fig 3. Scatter plot amount fraud

![image](https://github.com/user-attachments/assets/d2767395-368b-41ec-9cf8-87fd150ba0b5)

DATA PRE-PROCESSING

Since our data does not have many variables, the pre-processing of the data will be short and simple. The first thing to do is to eliminate variables with zero variance, i.e. variables that take a single value. In our case, the only features that take a single value are zipMerchant and zipCodeOrigin, so we will remove them before applying the models. This value is ’28007’. Another good thing about these data is that it doesn’t have missing values, so we don’t need to deal with it.
Next we will convert the categorical variables to numerical variables using a very simple code snipped. With this, our categorical variables will take values from one to the number of levels they each have, seven in the case of age, 4 in the case of gender and so on. Ideally we would convert these variables into dummies as they have no relation to size. That is, one customer is not bigger than another. But since we have a lot of observations, these variables would grow to be very large and it would take us forever to train the models. Moreover, as we will see in the following chapters, the fact that we do not convert them into dummies does not affect our results, which turn out to be really good.
Finally, to finish with our pre-process, we define our independent variable, X, by removing the fraud column, and the response variable, y, which will be only the column referring to fraud.

MACHINE LEARNING ALGORITHMS

In this chapter we will try to explain how the algorithms we are going to use work, how we apply them and the results we obtain from them. As we will see, each of them has a different theoretical basis and it is interesting to learn how they work before using them, in order to understand why some of them work better than others for our data. As we said before, we will apply each of the methods twice, the first time doing Undersampling of the data, and the second time doing a combination of Oversampling and Undersampling. We have done this combination as it turns out to be the most useful for unbalanced data. It may be that better estimates can be obtained by doing only Undersampling, but we cannot afford to lose so much information. We must create concise and realistic models for fraud detection that are not only useful for our data but can be applied to other data sets, to help people who have unfortunately fallen into the clutches of unscrupulous scammers.

  K-NN: K Nearest Neighbors
The first algorithm we are going to apply is the well-known K-NN (K Nearest Neighbors). This method is a non-parametric supervised learning algorithm, which uses the proximity between data points to make predictions about the grouping of individual data points. Although it can be applied to regression problems, its most widespread use is in classification problems since similar points can be found close to each other. The basic idea underlying this paradigm is that a new case will be classified in the most frequent class to which its K nearest neighbours belong. The paradigm is therefore based on a very simple and intuitive idea, which together with its easy implementation makes it a very widespread classification algorithm. The only drawback is that it uses the entire data set to find each point and therefore requires a lot of memory and processing resources. Therefore, it tends to work better on small data sets and without a huge number of variables. Let’s see what its theoretical basis is, the notation we are going to use to explain it is the following:


		                   X1	    ...	 Xj	   ...	Xn	    C

(x1,c1) ∈ D  	1	     x11	   ...	 x1j	  ...	x1n	   c1
           	  ...	   ...		  ...		...	  ... ...    ...
(xi,ci) ∈ D	  i	     xi1	   ...	 xij	  ...	xin	   ci
           	  ...	   ...		  ...		...	  ... ...    ...
(xN,cN) ∈ D	  N	     xN1	   ...	 xN j	 ...	xNn	   cN

x	            N + 1	 xN+1,1	...	xN+1,j	...	XN+1,n	?

•	D indicates a file of N cases, each of which is characterised by n predictor variables X1,..., Xn, and a variable to predict, the class C.

•	The N cases are denoted as follows:
(x1,c1),...,(xN,cN) where xi = (︁xi,1,..., xi,n)︁ ∀ i = 1,..., N ci ∈ {︂c1,...,cm}︂ ∀ i = 1,..., N
c1,...,cm denote the m possible values for the class C.

•	The new case to classified is denoted as x = (x1,..., xn)

Once we have defined all this, the algorithm process is straightforward. First we calculate the distances from all the cases already classified to the new case, x, which we want to classify. Once we have selected the K cases already classified, DxK closest to the new case x, we assign the value of the most frequent class C among the K objects, DxK.

Here it is shown in a more visible way:
	Input → D = {(x1,c1),...,(xN,cN)}	Case to classify → x = (x1,..., xn)

For all object already classified (xi,ci) → We calculate di = d (xi,x)
we sort di in ascending order, keep the K cases DxK already classified that are the closest to x, and assign to x the most frequent class DxK.

The following figure shows 24 cases already classified into two possible values (m = 2). The predictor variables are X1 and X2 and K = 3 has been selected. Of the 3 cases already classified that are closest to x (represented by •), two of them belong to the class ◦, so K-NN predicts the class ◦ for the new case. If we had chosen K = 1, the classifier would have assigned x the class +, which is the closest.

Fig 4. Example of application of the K-NN algorithm

![image](https://github.com/user-attachments/assets/1b46b255-42b3-480d-8bd6-517348395cfa)


As can be seen, there are two important things to select before running the algorithm on our data:
•	The metric we will use to calculate the distance between the query point, x, and the other data points.
•	The value of K: How many neighbours we will check to determine the classification of the query point.

Having explained how this wonderful machine learning method works, let’s see what results can be obtained by applying it to our data on bank frauds.


K-NN with Undersampling

After doing the preprocessing explained in chapter 3, our next step is to do an Undersampling of our data. We set any seed and reduce the majority class (0) to 7.200, which is the number of frauds in our data, using the function RandomUnderSampler from the package imblearn.under_sampling. After doing this, we split our independent variable and the response variable into train and test using the train_test_split function of the sklearn.model_selection package.

We now have our balanced response variable and can start with the model. First, we would like to define the concept of cross-validation, as we will use it throughout the project. Cross-validation is a technique usually used in machine learning to assess the variability of a data set and the reliability of any model trained on it. This is how it works:
  1.	Cross-validation randomly divides the training data into folds, let’s say 5.
  2.	The component reserves the data from fold 1 for use in validation and uses the remaining folds to train a model. It generates five models, trains each 		model      on four fifths of the data, and tests each model on the remaining fifth.
  3.	During testing of the component for each fold, the module evaluates various accuracy statistics. The statistics used by the component depend on the type 	of       model being evaluated.
  4.	When the compilation and evaluation process is completed for all folds, the crossvalidation model generates a set of yield metrics and scored results for         all data.

For this case of K-NN using Undersampling, we will consider a stratified crossvalidation of 10 folds in 5 repeats, using the function RepeatedStratifiedKFold from the sklearn.model_selection package. The fact that it is stratified ensures that each fold of the data has the same proportion of observations with a given label.

Having defined the cross-validation, let’s see what the value of K is going to be. Apart from the choice of the K, we have also found it interesting to find out which are the most influential variables. To do these two things at the same time, we will use a pipeline in which we will include the functions that will select the variables, SelectKBest(f_regression) (both of them from the package sklearn.feature_selection), and the one that will get the optimal number of neighbours, KNeighborsRegressor from the package sklearn.neighbors. Then, using a Grid Search, we will evaluate everything in our training set and obtain the desired values.
These are:
  •	K = 5
  •	Features = 4: Gender, Merchant, Category and Amount.

We now have the K, all that remains is to run the model and evaluate it on the test data. Once this is done we get an accuracy of 0.922 with a standard deviation of 0.007. The classification report of the model, the confusion matrix and the ROC-AUC curve are shown below.

			Precision	Recall	f1-Score	Support
0			0.89		0.97	0.93		1455
1			0.96		0.88	0.92		1425
Accuracy					0.92		2880
Macro Average		0.93		0.92	0.92		2880
Weighted Average	0.93		0.92	0.92		2880

Fig.  ROC-AUC Curve K-NN (Undersampling)		Fig. Confusion Matrix K-NN (Undersampling)


![image](https://github.com/user-attachments/assets/6e418bb8-f184-4cda-a886-b75044715859)


K-NN with Over-and-Undersampling

Having seen how K-NN works with Undersampling, let’s now see if the results are improved by applying a combination of Over-and-Undersampling (in this case, for all algorithms, the division into train and test shall be done before sampling). To do this, we will use the Pipeline function from the imblearn.pipeline package. For the Undersampling we will use the aforementioned function, and the Oversampling will be done with the SMOTE function from the imblearn.over_sampling package. Within the pipeline we will include these two sampling and the K-NN model. This works as follows:

1.	Apply SMOTE to give give the minority class 5% of the size of the majority class.
2.	Using RandomUnderSampler we reduce the majority class to 50% more than the minority class.
 
In this case, due to the large number of observations we have, we will use a crossvalidation of 5 folds and 3 repeats. After running the Pipeline, we get an accuracy of 0.942 with a standard deviation of 0.004 (a bit better than in the Undersampling section).

As before, we show the classification report of the model, the confusion matrix and the ROC-AUC curve.

TABLE CLASSIFICATION REPORT FOR K-NEAREST NEIGHBOURS (OVER-AND-UNDERSAMPLING)

			Precision	Recall	f1-Score	Support
0			1.00		0.97	0.98		117461
1			0.26		0.87	0.40		1468
Accuracy					0.97		118929
Macro Average		0.63		0.92	0.69		118929
Weighted Average	0.99		0.97	0.98		118929


Fig. ROC-AUC Curve K-NN	(Over-and-Undersampling)	Fig. Confusion Matrix K-NN (Over-and-Undersampling)

![image](https://github.com/user-attachments/assets/6cf2f864-8a61-4dcc-bbf3-17654163380f)

Once we have the two results, let’s compare them. First, the metric we must look at is recall. As you can see in the tables, the recall is the same for the majority class and almost the same for the minority class. The objective of this work was fraud detection, so what we are really interested in is the percentage of failure in the ones, that is, what percentage of the total number of frauds that are classified are predicted as zeros. The following table shows the percentage of failure in both zeros and ones and both Undersampling and Over-and-Undersampling.

TABLE PERCENTAGE OF FAILURE IN K-NN

Class	Undersampling	Over-and-Undersampling
0	3.44%		3.10%
1	11.86%		12.87%

The failure rate for ones is better in Undersampling, but the failure rate for zeros is somewhat worse. Anyway, since what we are interested in are the ones, and the difference in the failure rate for the zeros is minimal, we can conclude that in this case by Undersampling we obtain more favourable results.

Decision Tree Classifier

The next algorithm we are going to apply is Decision Trees. These are predictive models made up of binary rules that distribute the observations according to their attributes and thus predict the value of the response variable. Statistical and machine learning methods based on trees encompass a set of supervised non-parametric techniques that segment the space of predictors into simple regions, within which it is easier to handle interactions.
These are their main advantages.

•	Trees are easy to interpret even when the relationships between predictors are complex.
•	Trees can, in theory, handle both numerical and categorical predictors without having to create dummy variables.
•	They generally require much less data cleaning and pre-processing compared to other statistical learning methods (i.e., they do not require 			standardization).
•	They can select predictors automatically.
•	They can be applied to both regression and classification problems.

In this paper we will deal with classification trees, which are the subtype of decision trees that we apply when the response variable is categorical. In general terms, in the training of a classification tree, the observations are distributed through nodes generating the structure of the tree until a terminal node is reached. When a new observation is to be predicted, the tree is traversed according to the value of its predictors until one of the terminal nodes is reached. The training process in a classification tree is divided into two stages.

•	Successive partitioning of the predictor space generating terminal nodes (nonoverlapping regions). If these regions are limited to multi-dimensional 		rectangular regions, the construction process is simplified.
•	Prediction of the response variable in each region.

Despite the simplicity with which the process of constructing a tree can be summarized, it is necessary to establish a methodology to create the regions R1,R2,R3,...Rj or, equivalently, to decide where the divisions are introduced: in which predictors and in which values of the predictors. The method that is used to construct a classification tree is recursive binary splitting. This solution follows the same idea as stepwise predictor selection in multiple linear regression, it does not evaluate all possible regions but achieves a good computational-outcome balance.

The aim of this method is to find in each iteration the predictor Xj and the cut-off point s such that, if the observations are distributed in the regions {X|Xj < s} and {X|Xj ≥ s}, the greatest possible reduction of the method we use to find the most homogeneous nodes possible is achieved. The most used methods for the selection of optimal splits are the following:

•	Gini Index: It is considered a measure of node purity.
		K
		∑︂
	Gm =	    pˆmk (︁1 − pˆmk)︁
		k=1

When pˆmk is near a 0 or a 1 (the node contains mostly observations of one class), the term pˆmk (︁1 − pˆmk)︁ is very small. As a consequence, the higher 	the purity of the node, the lower the value of the Gini Index.

•	Classificatioon Error Rate: It is defined as the proportion of observations that do not belong to the most common class in the node.
	
 Em = 1 − maxk (︁pˆmk)︁

where pˆmk represents the proportion of observations of the m node that belongs to the k class.

•	Cross Entropy: Entropy is another way of quantifying the disorder of a system. In the case of nodes, disorder corresponds to impurity. If a node is pure, 	containing only observations of one class, its entropy is zero. Conversely, if the frequency of each class is the same, the entropy value reaches the 		maximum value of 1.

		K
		∑︂
	D = −		pˆmk log(︁pˆmk)︁
		k=1

•	Chi-Square: This approach consists of identifying whether there is a significant difference between the child nodes and the parent node, i.e. whether 		there is evidence that the splitting achieves an improvement. The higher the χ2 statistic, the greater the statistical evidence that there is a 		difference.
	
	χ2 = ∑︂ (observed k − expected k)︁2
 		_________________________
			expected k
		k

Having explained the different methods for the selection of optimal splits, we will continue by explaining the recursive binary splitting method. The algorithm used is the following:

1.	The process starts at the top of the tree, where all observations belong to the same region.
2.	All possible cut-off points s are identified for each of the predictors (X1, X2,..., Xp). In the case of qualitative predictors, the possible cut-off 		points are each of their levels. For continuous predictors, their values are ordered from lowest to highest, the intermediate point between each pair of 	values is used as the cut-off point.
3.	The total of one of the methods explained above (whichever we choose) that is achieved with each possible division identified in step 1 is calculated.
4.	We select the predictor Xj and the cut-off point S that gives rise to the most homogeneous splits possible. If there are two or more splits that achieve 	the same improvement, the choice between them is random.
5.	Steps 1 to 4 are repeated iteratively for each of the regions that have been created in the previous iteration until some stop rule is reached. Some of 	the most used is that no region contains a minimum of n observations, that the tree has a maximum of terminal nodes, or that the incorporation of the 		node reduces the error by at least a minimum %.

Decision Tree with Undersampling

Now that we have explained how Decision Trees work, let’s apply them to our data. We will do this using the function DecisionTreeClassifier from the sklearn.tree package. We can specify many parameters in this function, but in our case we will only need the following:

•	criteria = "gini": The method we use for selecting optimal splits.
•	random_state = 42: Controls the randomness of the estimator.
•	max_depth = None : The maximum depth of the tree. In this case (None) the nodes are expanded until we have all pure leaves, or all leaves contain less 		than min_samples_split (the minimum number of samples required to split an internal node) samples. We will use the default value of this parameter which 	is 2.
•	min_samples_leaf = 5 : The minimum number of samples required to be in a leaf node.

On the other hand, we are going to change the cross-validation for this algorithm, considering now 5 folds in 3 repeats using the same function as for K-NN
(RepeatedStratifiedKFold). Once this is done, we get an accuracy of 0.968 with a standard deviation of 0.003. The classification report of the model, the confusion matrix and the ROC-AUC curve are shown below.

TABLE CLASSIFICATION REPORT FOR DECISSION TREE CLASSIFIER (UNDERSAMPLING)

			Precision	Recall	f1-Score	Support
0			0.97		0.97	0.97		1455
1			0.97		0.97	0.97		1425
Accuracy					0.97		2880
Macro Average		0.97		0.97	0.97		2880
Weighted Average	0.97		0.97	0.97		2880

Fig. ROC-AUC Curve Decission Tree Classifier (Undersampling)  Fig. Confusion Matrix Decission Tree Classifier (Undersampling)

![image](https://github.com/user-attachments/assets/3eb8ecce-7909-46ee-988e-e976e950e737)
		
As can be seen, the results are extremely good. The failure rate is very small in both classes, with the 1 failing the least. It can also be seen from the table above that we have a very high recall and accuracy for both classes.

TABLE PERCENTAGE OF FAILURE IN DECISION TREE

Class	Undersampling
0	3.44%
1	2.67%

Due to the high number of observations when doing Over-and-Undersampling, Python is not able to create a tree. It only creates one when we set the minimum number of samples required to be in a leaf node and to split an internal node too high. When it does, it predicts all zeros as zeros, but fails to predict any one as one. This is why we have not considered it appropriate to include it in this project.

Random Forest Classifier

A Random Forest model consists of an ensemble of individual decision trees, each trained on a random sample drawn from the original training data by bootstrapping. This implies that each tree is trained on slightly different data. In each individual tree, the observations are distributed along nodes generating the tree structure until a terminal node is reached. The prediction of a new observation is obtained by aggregating the predictions of all the individual trees that make up the model.

To understand how Random Forest models work, it is first necessary to understand the concepts of ensemble and bagging.

In general, small trees (few branches) have low variance but fail to represent the relationship between variables well, i.e. they have high bias. In contrast, large trees are very close to the training data, so they have very low bias but high variance. One way to solve this problem is ensemble methods.
Ensemble methods combine multiple models into a new model with the aim of achieving a balance between bias and variance, thus achieving better predictions than any of the original individual models. Two of the most commonly used types of ensemble are:

•	Boosting: Multiple simple models, called weak learners, are fitted sequentially so that each model learns from the errors of the previous one. As a final 	value, the mean of all predictions (continuous variables) or the most frequent class (qualitative variables) is taken.
•	Bagging: Multiple models are fitted, each with a different subset of the training data. To predict, all models that make up the aggregate participate by 	contributing their prediction. As a final value, as in boosting, the mean of all predictions (continuous variables) or the most frequent class 			(qualitative variables) is taken. Random Forest models fall into this category.

In the case of Decision Trees, given their low bias and high variance nature, bagging has proven to have very good results. The way to apply it is:

1.	Generate B pseudo-training sets by bootstrapping from the original training sample.
2.	Train a tree with each of the B samples from step 1. Each tree is created with almost no constraints and is not pruned, so it has high variance but 		little bias. In most cases, the only stopping rule is the minimum number of observations that the terminal nodes must have. The optimal value of this 		hyperparameter can be obtained by comparing the out-of-bag error or by cross-validation.
3.	For each new observation, obtain the prediction of each of the B trees. The final prediction value is obtained as the mean of the B predictions in the 		case of quantitative variables and as the most frequent predicted class for qualitative variables.
	In the bagging process, the number of trees created is not a critical hyperparameter in that, no matter how much the number is increased, the risk of 		overfitting is not increased. Once a certain number of trees is reached, the test error reduction stabilizes. However, each tree takes up memory, so it 	is not advisable to store more than necessary.

The Random Forest algorithm is a modification of the bagging process that achieves improved results by further decorrelating the trees generated in the process.

If we have a data set with a predictor that is very influential over the others, all the trees created in the bagging process will be dominated by the same predictor and will be very similar to each other. Therefore, due to the correlation between the trees, this process will hardly decrease the variance or improve the model. Random Forest avoids this problem by making a random selection of m predictors before evaluating each split. In this way, an average of (p − m)/p (where p is the number of predictors) splits will not consider the influential predictor, allowing other predictors to be selected. The difference in the result will depend on the value m chosen. If m = p the Random Forest and bagging results are equivalent. As said before, the best way to find the optimal value of m is to evaluate the out-of-bag error or to resort to cross-validation.

Random Forest Classifier with Undersampling

In order to apply this algorithm we are going to use the RandomForestClassifier function that we can find in the sklearn.ensemble package. The parameters we are going to specify in this function are the following. In all others we will use its default value.

•	n_estimators = 300: The number of trees in the forest.
•	max_depth = 8: The maximum depth of the tree
•	random_state = 42: Controls the randomness of the bootstrapping of the samples used building the trees and the sampling of the features.
•	verbose = 1: Controls the verbosity when fitting and predicting.
•	class_weight = balanced: All the classes are supposed to have weight one since it is not given in our data. In this case, the balanced mode adjust 		weights inversely proportional to class frequencies in the input data.
•	n_jobs = −1: The number of jobs to run in parallel, −1 indicates that we use all processors.
	As in the case of decision trees, we will use a 5 fold cross validation on 3 repeats. Once the method is applied, we obtain an accuracy of 0.971 with a 	standard deviation of 0.002. Let’s see how the classification report, the confusion matrix and the ROC-AUC curve look like.

TABLE CLASSIFICATION REPORT FOR RANDOM FOREST CLASSIFIER (UNDERSAMPLING)

			Precision	Recall	f1-Score	Support
0			0.99		0.96	0.97		1455
1			0.96		0.99	0.97		1425
Accuracy					0.97		2880
Macro Average		0.97		0.97	0.97		2880
Weighted Average	0.97		0.97	0.97		2880

Fig. ROC-AUC Curve Random Forest Classifier (Undersampling)   Fig. Confusion Matrix Random Forest Classifier (Undersampling)

![image](https://github.com/user-attachments/assets/5072feab-376c-419a-88b8-f8d5c2a90048)


Random Forest Classifier with Over-and-Undersampling

As in the K-NN section, we will use the Pipeline function to combine both Over and Undersampling. For this model we set this:

1.	Apply SMOTE to give the minority class 6% of the size of the majority class.
2.	Using RandomUnderSampler we reduced the majority class to 50% more than the minority class.

The cross validation strategy is the same as in the Undersampling case and we change the n_estimators to 100 which is its default value. After applying it we obtain an accuracy of 0.995 with an standard deviation of 0.0004. The results when applying the method in the test data are shown below.

TABLE CLASSIFICATION REPORT FOR RANDOM FOREST CLASSIFIER (OVER-AND-UNDERSAMPLING)

			Precision	Recall	f1-Score	Support
0			1.00		0.96	0.98		117461
1			0.24		0.98	0.38		1468
Accuracy					0.96		118929
Macro Average		0.62		0.97	0.68		118929
Weighted Average	0.99		0.96	0.97		118929

Fig. ROC-AUC Curve Random Forest Classifier(Over-and-Undersampling)        Fig. Confusion Matrix Random Forest Classifier (Over-and-Undersampling)

![image](https://github.com/user-attachments/assets/80b7fb09-bc68-4cb3-b532-19de55b3bd58)


Now that we have the two results using different types of sampling, let’s compare them. As we did for the two previous algorithms, the following table shows the percentage of failures in each class.

TABLE PERCENTAGE OF FAILURE IN RANDOM FOREST CLASSIFIER

Class	Undersampling	Over-and-Undersampling
0	3.99%		3.91%
1	1.12%		1.70%

All the results are really good. You can see how the recall is extremely good for the ones using both types of sampling, and the ROC-AUC curve is hardly noticeable in the graph, which is an indication that the models work very well. On the other hand, if we look at the failure rate table, we can see that the failure rate is very low for both samples and for both zeros and ones. It seems that Undersampling works somewhat better for ones and somewhat worse for zeros, but since using Over-and-Undersampling the number of observations is quite high, we would stick with this type of sampling for this model. Moreover, the loss of information produced by Undersampling is an additional incentive to use Over-and-Undersampling.

In Summary

Once we have run all our algorithms, we summarize the results in this table. Let’s see what interesting information we can extract from it.

TABLE PERCENTAGE OF FAILURE OF ALL ALGORITHMS

Sampling		Undersampling		Over-and-Undersampling		Mean	
Class			0	1	Total	0	1	Total		0	1	Total
KNN			3.44%	11.86%	7.65%	3.10%	12.87%	7.99%		3.27%	12.37%	7.82%
Decission Tree		3.44%	2.67%	3.06%					3.44%	2.67%	3.06%
Random Forest		3.99%	1.12%	2.56%	3.91%	1.70%	2.81%		3.95%	1.41%	2.68%
Mean			4.00%	3.95%	3.98%	3.60%	5.78%	4.69%		3.82%	4.48%	4.15%

First, the last 4.15% box shows the total failure rate, taking into account all methods, the two types of sampling and the two classes. This percentage is 4.15%, which is really good, we have only failed a little more than 1 out of 25 predictions. Let’s see now which method works best.

A priori, we have the lowest failure rate using Random Forest and it is only 2.68%. Looking at the table, although the failure rate on zeros is not the best, the percentage detecting frauds is the best of all, with only 1.41% of the ones predicted as zeros. Despite this being the best method now, special mention must be made of the Decission Tree. It is also true that in Decission Tree we have considered only Undersampling and this greatly improves the failure rate with respect to the others.

As to which type of sampling is better, we have the failure rate using Undersampling and in cyan using Overand-Undersampling. As expected, the percentage is better using Undersampling, but not much better, they only differ by 0.71%. Considering that the number of observations to be classified using Over-and-Undersampling is about 80 times higher than those classified using Undersampling and knowing beforehand the huge loss of information that using Undersampling alone entails, we can conclude that the second type of sampling has worked very well, giving us really encouraging results. Random Forest is again the best performing algorithm for both sampling.

At this point, we can say that the failure rate is good, but it could have been better if K-NN had worked in a more optimal way. This algorithm clearly fails in predicting frauds, having a failure rate in both samples higher than 10%, which is not bad at all, but it is certainly much worse than in the other algorithms, where the highest failure rate in predicting ones is 2.76% in XGBoost using Over-and-Undersampling.

Deep Learning

TO BE CONTINUED...
