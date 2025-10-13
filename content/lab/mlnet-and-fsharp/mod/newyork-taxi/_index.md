---
title: "4. Predict Taxi Fares"
description: "Predict taxi fares in New York City"
type: "mod"
layout: "single"
image: mod-tlctaxi.jpg
sortKey: 40
---
In this lab, you're going to write an app in F# that predicts taxi fares in New York City. You'll use the New York TLC Dataset, which tracks every taxi trip made in the New York City area. The data includes pickup and dropoff dates and times, passenger counts, trip duration, fares, tolls and taxes. Everything you need to build an accurate taxi fare estimator. 

Just like with the California Housing dataset, you will have to perform feature engineering by detecting and dealing with outliers, normalizing columns, one-hot encoding categorical data columns, and adding new calculated features where applicable. You'll also have to split the dataset, select a regression learning algorithm and use it to train a model on the training data partition, then test the fully trained model on the evaluation partition and calculate the regression metrics. 

Finally, you'll imagine a taxi trip through New York City and use your model to generate a fare prediction for that trip. 
