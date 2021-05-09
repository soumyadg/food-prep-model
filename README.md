
# Data Science Exercise - Food Preparation Time Model

## Problem Definition
Deliveroo is committed to providing a delivery experience that delights our customers while still being incredibly efficient. Thus it is critical that we have the best possible model of how long it takes for a food order to be prepared. This allows us to ensure that a rider arrives at a restaurant to pick up an order exactly when the food is ready.
The aim of this exercise is to use historical data to predict the food preparation time for each order.

## Modelling Exercise
1. Use any tool or language you would prefer.
2. Perform any cleaning, exploration and/or visualisation on the provided data (orders.csv and restaurants.csv)
3. Build a model that estimates the time it takes for a restaurant to prepare the food for an order.
4. Evaluate the performance of this model. If you were a data scientist at the Deliveroo working on this (i.e. if this was your day job and not just part of an interview process!), explain your next steps with this model considering its potential applications within the food delivery system.

Feel free to include a list of additional ideas you would explore if you had more time to work on this problem. You will have the opportunity at an onsite interview to explain your approach and answer further questions on it. We are looking for sensible approach that does the basics well, not a showcase of novel techniques. The text is as important as the code, it's important that you guide us through your thought process. What are the key take-aways from your EDA? Why some feature engineering makes sense? Why did you pick that algorithm to solve the task? What are the business implications of the results of your modelling?

Note: All timestamps are given to you in the local timezone. You have no need to try any timezone conversions.

Please return your code and explanation of your approach. A jupyter notebook is a good format to merge both in the same file, but feel free to use what you find more useful. Please include any required instructions for how to run your code. Also, if you decide to submit a notebook please include a pdf or html version to make its reading a bit easier.


## Appendix: Data Schema and Description

Filename : **orders.csv**

| Column Name       | Type    | Description                   |
|-------------------|---------|-------------------------------|
| order_acknowledged_at | String (timestamp) | Timestamp (local timezone) indicating when the order is acknowledged by the restaurant. |
| order_ready_at    | String (timestamp) | Timestamp (local timezone) indicating when the food is ready. |
| order_value_gbp       | Float   | Value of the order in GBP.|
| restaurant_id     | Integer | Unique restaurant identifier. |
| number_of_items   | Integer | Number of items in the order. |
| prep_time_seconds   | Integer | (order_ready_at - order_acknowledged_at) in seconds. This is the food preparation time and what you should model. |

Filename : **restaurants.csv**

| Column Name       | Type    | Description                   |
|-------------------|---------|-------------------------------|
| restaurant_id     | Integer | Unique restaurant identifier  |
| country           | String  | Country where the restaurant is |
| city              | String  | City where the restaurant is |
| type_of_food      | String  | Type of food prepared by the restaurant |




## Solution

****CONTEXT****

Whenever we think of Deliveroo, we think of great food delivered to our doorsteps as fast as possible, usually within half an hour. But the real story lies behind the scenes. When Deliveroo started in 2013, it had a completely different operating model to what it has now. While the outcome was exactly the same, food being delivered to your home, the mode of operation was completely different. The app let the customer order food. After the order was acknowledged by the restaurant, the restaurant took their time to prepare it and then the restaurant would call up a delivery partner who would then arrive and pick up the food and deliver to the customer. Fast forward to 2021- now Deliveroo has completely automated the process increasing its efficiency by multiple folds. The food doesn’t have to wait longer at the counter after being prepared. The customer doesn’t have to wait for long before she/he can get the food delivered and the rider partner reaches the restaurant exactly (almost) when the food is ready to be picked up.
Deliveroo operates one of the most complicated marketplaces with three different aspects – the riders, the restaurants and the customers. While millions of orders are placed through the day, in over 13 countries and 200 cities worldwide, the secret to the solution are the data and the intense machine learning algorithms that run in the back, matching riders and restaurants allowing orders to reach the customer doorsteps on time.

****DATA****
The data that was provided for this take home comprised of Deliveroo orders table and a restaurant table from 22 different cities from 4 different countries – UK, Germany, Ireland and France. While the restaurant table has the ‘restaurants_id’ as the primary key with other columns being country, city and type of food, the order data table has 6 columns - order acknowledge time, order ready time, value of the order in gbp, restaurant id, number of items ordered and preparation time in seconds.

****DATA CLEANING****
Data Cleaning refers to the process of correcting bits and parts of data to ensure we achieve high data-integrity. For this data multiple steps were undertaken, such as importing and joining the raw csv files on the primary key ‘restaurant_id’, Converting order time and order ready time columns to data-time object for further analysis etc as shown in the jupyter notebook file.
Machine Learning Take Home – Deliveroo used case.
 
****EXPLORATORY DATA ANALYSIS (EDA)****
One of the most important things for any machine learning analysis is the Exploratory Data Analysis. EDA refers to the most critical process of machine learning – to gauge the data, perform initial investigations, discover trends, spot anomalies and to check patterns using statistical summaries and graphical representations. Similarly for this data, an intensive EDA was performed to gain insights before getting our hands dirty with it.

All the EDA that has been performed has been enumerated below with their respective observations and the business perspective associated with it:

**Analysis 1:** Order counts by the hour.  
**Observation:** Most orders are placed between 6:00 pm to 9:00 pm.  
**Business perspective:** It is imperative to realize when DELIVEROO receives most requests for food delivery as they would like more delivery partners on the road to serve its customers better and faster during these peak hours.


**Analysis 2:** Count of orders based on the countries.  
**Observation:** It can be seen that UK has the lion's share of Deliveroo orders.  
**Business perspective:** UK would need more riders than any other countries.

**Analysis 3:** Distribution of transactions from each city.  
**Observation:** 78% of the transactions are only from London.  
**Business perspective:** London would require more riders than any other cities.

**Analysis 4:** Distribution of cuisine that has been ordered.  
**Observation 4:** Italian, Burgers, Thai, American and Japanese has been ordered the most, accounts for almost 50% of the total count of orders.
**Business perspective:** More drivers around these types of restaurants could be highly beneficial for Deliveroo, would facilitate faster food pick-up and drop-off.

**Analysis 5:** Mean order-values in GBP based on the cities from different countries.  
**Observation:** Berlin has the least average order value while London has the most. One of the reasons for this is the fact that orders in Berlin are charged in Euros which has lower exchange rates than in London. So, for similar food, customers are supposed to pay less in other countries than in the UK. Also, cities in UK such as London, Manchester, Liverpool are more expensive than other metropolitan cities from other countries.  
**Business perspective:** Cities in the UK would provide more revenue to Deliveroo.

**Analysis 6:** Distribution of Order values.  
**Observation:** The majority of the order values ranges between 10-40 GBP.  
**Business perspective:** Deliveroo understands the spending pattern of customers.

**Analysis 7:** Preparation time of food in hours based on the price range of restaurants.  
**Observation:** We see that the more expensive the restaurant is, the more time it takes to prepare the food.  
**Business perspective:** Deliveroo could dispatch more riders towards restaurants which are in-expensive like ready-made or fast-food restaurants.

**Analysis 8:** Created a bucket based on the hour of order.  
**Observation:** The maximum number of orders are placed in the Evening.  
**Business perspective:** Deliveroo should dispatch maximum riders in the evening.

**Analysis 9:** Average food preparation time based on the shifts in the day.  
**Observation:** Even though we see that majority of the orders are placed in the evening, orders placed in the morning take more time to prepare. This might be because of less chefs available in the morning.  
**Business perspective:** Deliveroo can convey a message to the customers ordering food during the morning shift that the food might take longer to reach them. They can incorporate that on their app by increasing the estimated time of arrival (ETA) of the order.

**Analysis 10:** Counts of orders based on the day of the week.  
**Observation:** Saturdays and Sundays have the highest order counts in a week.  
**Business perspective:** Deliveroo should employ maximum riders during the weekend.

**Analysis 11:** Average food preparation time based on the day of the week.  
**Observation:** Sundays and Saturdays have the least food preparation time, probably restaurants hire extra chefs to mitigate the extra demand on the weekends.  
**Business perspective:** As food prepared on weekends take less time, Deliveroo can dispatch maximum rider partners during the weekends. Weekends bring in more business for the restaurants and more revenue for the riders and Deliveroo.

**Analysis 12:** Heatmap to check out the correlation between different features of the dataset. 
**Observation:** The order value in pounds has a correlation with price per item. Let us consider a small example to explain this: A customer orders a pizza which costs £10. So, the price per item for such an order is 10. Now if another customer orders the same pizza and adds a bottle of Coca-Cola for £2, the price per item goes to £6. This correlation can be explained from the heatmap.  

**Analysis 12:** Heatmap to check out the correlation between different features of the dataset.  
**Observation:** The order value in pounds has a correlation with price per item.
Let us consider a small example to explain this: A customer orders a pizza which costs £10. So, the price per item for such an order is 10. Now if another customer orders the same pizza and adds a bottle of Coca-Cola for £2, the price per item goes to £6. This correlation can be explained from the heatmap.
Another strong correlation can be seen between order value and number of items ordered which is very intuitive.  
**Business perspective:** Deliveroo can use highly correlated features for machine learning.

**Analysis 13:** Distribution of food preparation time  
**Observation:** Majority of the orders are being prepared within an hour.  
**Business perspective:** Deliveroo can dispatch riders in a way so that can reach the restaurant within 60 minutes of the order being acknowledged. This time should be calculated keeping in mind the time it would take the driver partner to reach the restaurant.

**Analysis 14:** Average time for food preparation time based on cities.  
**Observation:** Restaurants in outskirts of metropolitan cities take longer time to prepare food.  
**Business perspective:** While restaurants outside the metropolitan cities would take more time to be ready, Deliveroo can suffice the demand with lesser drivers, prevents overusing resources.
 
**Analysis 15:** Average food preparation time based on the countries.  
**Observation:** Average food preparation time in Ireland is highest among the other countries.  
**Business perspective**: Ireland seems to take more time to prepare food, so lesser drivers in Ireland would suffice the demand with respect to other France.

**Analysis 16:** Time required to prepare different types of food.  
**Observation:** As it appears from the data that Juices require the most time to prepare. This looks like a bit of anomaly. This is because of few juice orders that took a long time to prepare as evident from the data.  
**Business perspective:** This analysis lets Deliveroo to understand which type of food needs more time to prepare and use it to update their machine learning model. Better estimation improves the food delivery process that might lead to higher rider tips – an added incentive is always amazing.

## MACHINE LEARNING

Now that the data cleaning and EDA has been done, the next stage would be using the data to train a machine learning model. From a high level, it’s evident that the rider dispatch system acts as a brain of Deliveroo. The rider dispatch algorithm depends on multiple factors such as the distance of the pickup point from the riders’ current location, the distance of the drop off point from the rider’s current location, time of the food preparation etc. With the data from 2015 given for this exercise, if we could predict the time a restaurant would take to prepare the food, it would let Deliveroo dispatch riders to the restaurant on time expediting the pickup and delivery process of the order. This acts as the primary motivation for this machine learning analysis.

**ONE HOT ENCODING**  
One Hot Encoding is generally used to deal with categorical variables in the data. The word encoding refers to representing each piece of data so that a machine learning algorithm can understand. For this exercise, three categorical features – Time Bucket, Price and Country has been one hot encoded. The reason for including only these three categorical variables is that they represent only 4 types of distinct variables each. This leads to appending 12 extra dimensions to the dataset. Other variable such as type of food has not been encoded as it would increase the dimension of the data set by 83 columns (there are 83 types of ‘type_of_food’) and might affect lead to issues like parallelism and multi-collinearity in high dimensions.

## MODELING

**Algorithms:**. 
While the objective of the modeling exercise is to predict food preparation time, we will be using regression-based analysis techniques. Most popular ones are Linear Regression, Polynomial Regression among others. In this analysis, few of such machine learning techniques have been used.

**Training and Test:**. 
The data has been split into train and test datasets with 60% data used for training and 40% used for testing purpose.
Machine Learning Take Home – Deliveroo used case. Submitted by Soumya Dasgupta
 
**METRICS**. 
The most common metrics used for the regression analysis are Mean Absolute Error, Mean Squared Error and Root Mean Squared Error. These metrics have been used to determine the efficacy of the machine learning techniques used in the analysis.
A scatterplot of test-data and predicted data for each of the Machine Learning algorithm has been shown as well. A good prediction would have points lying on the y=x line.
  
![](/Images/1.png)
  
  
From the table it can be seen that Mean Squared Error and Root Mean Squared Error is least for the Polynomial Regression machine learning technique and hence this would be used for the prediction purposes. The main reason for choosing these two metrics is that these are industry standard metrics to evaluate performance. The plot of test data and the predicted results falls closely to the y=x line for polynomial regression which isn’t the case for other machine learning algorithms.
While other algorithms also could have been tried, computation time for the algorithms and submission deadline has been prime reasons while only these 5 techniques could be tested.

## BUSINESS IMPLICATIONS

The output of the Polynomial Regression shows a Root Mean Squared Error of 1837 seconds which is almost half an hour. For a company like Deliveroo this might lead to considerable losses to all the parties involved – the restaurant, the driver the customer and Deliveroo. Hence this model needs to be improved.
 
## CONCLUSION AND FUTURE STEPS

The analysis above deals with predicting the estimate a restaurant takes to prepare food. A number of steps were undertaken like data importing, cleaning and exploratory data analysis before identifying machine learning models. Based on a month of data, 5 models were tried for prediction, out of which Polynomial Regression technique proved to have the least Root Mean Squared Error and hence it was chosen to proceed with the same.
As future steps, there are a number of things that could be done to improve the prediction of the model and they are enumerated below:
• As this analysis is based on 30 days of data, all but one rows from the month of June while the other from the month of July, more data would definitely do well towards the waning of the error. Generally, a year of data would do justice to the model.
• The data should be bucketed for different countries and the same model could be run.
• The data should be bucketed for different cities and the same model could be run.
• Adding more features like historical whether data would definitely increase the
model efficiency.
• A proper demand forecasting based on the data would increase model efficacy as
well.With more data and new features as mentioned above, this model can be improved. Other machine learning algorithms like Neural Network could also be tried to reduce the prediction time errors.
 
