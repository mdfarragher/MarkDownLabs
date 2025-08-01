---
title: "Evaluate The Results"
type: "lesson"
layout: "default"
sortkey: 120
---

Now let's evaluate the quality of the model by comparing the predictions made on the 20% test data to the actual fare amounts, and calculate the regression evaluation metrics.

So imagine you take a taxi trip in New York city an you use your model to predict the fare beforehand. What kind of prediction error would you consider acceptable?

Determine the minimum mean absolute error or root mean square error values you deem acceptable. This will be the target your model needs to beat.
{ .homework }

#### Calculate Evaluation Metrics

Enter the following prompt:

"Use the trained model to create predictions for the test set, and then calculate evaluation metrics for these predictions and print them."
{ .prompt }

That should create the following code:

```csharp
// Use the trained model to create predictions for the test set
Console.WriteLine("Evaluating model on test data...");
var predictions = model.Transform(testingData);

// Display the model evaluation metrics
var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "FareAmount");
Console.WriteLine();
Console.WriteLine($"**** Model Metrics ****");
Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:F3}");
Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError:F3}");
Console.WriteLine($"R-Squared: {metrics.RSquared:F3}");
```

This code calls `Transform` to set up predictions for every single taxi trip in the test partition. The `Evaluate` method then compares these predictions to the actual fare amounts and automatically calculates these metrics:

- **RSquared**: this is the coefficient of determination, a common evaluation metric for regression models. It tells you how well your model explains the variance in the data, or how good the predictions are compared to simply predicting the mean.
- **RootMeanSquaredError**: this is the root mean squared error or RMSE value. It’s the go-to metric in the field of machine learning to evaluate models and rate their accuracy. RMSE represents the length of a vector in n-dimensional space, made up of the error in each individual prediction.
- **MeanSquaredError**: this is the mean squared error, or MSE value. Note that RMSE and MSE are related: RMSE is the square root of MSE.
- **MeanAbsoluteError**: this is the mean absolute prediction error.

Note that both RMSE and MAE are expressed in dollars. They can both be interpreted as a kind of 'average error' value, but the RMSE will respond much more strongly to large prediction errors. Therefore, if RMSE > MAE, it means the model struggles with some predictions and generates relatively large errors. 

If you used the same transformations as I did, you should get the following output:

![Regression Model Evaluation](../img/evaluate.jpg)
{ .img-fluid .mb-4 }

Let's analyze my results:

The R-squared value is **0.992**. This means that the model explains approximately 99% of the variance in the fare amount. This is an exceptionally high level of explanatory power, suggesting that the model captures nearly all of the underlying patterns in the data. But this may be a sign of **overfitting**, where the model has simply memorized the entire dataset.

The mean absolute error (MAE) is **$0.425**. Given that NYC taxi fares typically range from around $2.50 to $50 or more for most trips, this level of error is extremely low. This means that on average, the model's predictions deviate from the actual fares by less than fifty cents.

The root mean squared error (RMSE) is **$0.541**. The RMSE penalizes larger errors more heavily than the MAE, and this value suggests that most predictions are very close to the true fare values, with almost no large deviations. 

So how did your model do?

Compare your model with the target you set earlier. Did it make predictions that beat the target? Are you happy with the predictive quality of your model? Can you explain what each regression metric means for the quality of your predictions? 
{ .homework }

#### Conclusion

Being able to generate fare predictions with an average error of only 42 cents is really good. It means that almost every taxy trip fare can be fully explained from its duration, distance covered, rate code and other relevant factors. The machine learning model discovered this pattern during training, and is applying the pattern to make near-perfect predictions for every fare.

Unfortunately we've given ourselves a very easy goal here. The full TLC dataset covers more than 8 million trips, but we are working with a fraction of that data. Our dataset holds 10,000 trips from shortly after midnight, on December 1st 2018. This is a very easy dataset to work with, and we may be looking at a situation where the SDCA algorithm is memorizing each trip. We won't know anything for sure until we run the app again on all 8 million trips.

We'll do that shortly, but first, let's add a prediction engine to the machine learning app to make a few ad-hoc fare predictions.
