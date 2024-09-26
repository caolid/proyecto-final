Este trabajo de fin de curso se basa en la detección de transferencias bancarias fraudulentas. Tras un exhaustivo análisis exploratorio de los datos, apliqué diferentes algoritmos de machine learning
y analicé su eficacia en el problema de clasificación.

Cada observación de los datos es una transferencia bancaria, por lo que la mayoría de ellas son no fraudulentas, 
lo que nos da unos datos desequilibrados y por ello tuve que realizar un par de técnicas de muestreo para cada algoritmo. 

Debido al gran número de observaciones, además de los algoritmos
mencionados anteriormente, consideré oportuno completar el estudio
haciendo uso del deep learning. Creé una red neuronal sencilla que definí como un modelo secuencial.

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

