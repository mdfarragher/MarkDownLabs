---
title: "Evaluate The Results"
type: "lesson"
layout: "default"
sortkey: 20
---

Now let's evaluate the quality of the model by having it generate predictions (called **scoring**) for the remaining 20% of data in the test partition. Then we'll compare those predictions to the actual median house values, and calculate the regression evaluation metrics.

So imagine you are a realtor in California selling houses. What kind of prediction accuracy would you consider acceptable?

{{< encrypt >}}

Determine the minimum mean absolute error or root mean square error values you deem acceptable. This will be the target your model needs to beat.
{ .homework }

#### Calculate Evaluation Metrics

Enter the following prompt:

"Use the trained model to create predictions for the test set, and then calculate evaluation metrics for these predictions and print them."
{ .prompt }

That should create the following code:

```csharp
// Prepare test data
var testX = np.zeros((testingData.Count, numFeatures), dtype: np.float32);
var testY = np.zeros((testingData.Count, 1), dtype: np.float32);

for (int i = 0; i < testingData.Count; i++)
{
    var record = testingData[i];
    testX[i, 0] = record.Longitude;
    testX[i, 1] = record.Latitude;
    testX[i, 2] = record.HousingMedianAge;
    testX[i, 3] = record.TotalRooms;
    testX[i, 4] = record.TotalBedrooms;
    testX[i, 5] = record.Population;
    testX[i, 6] = record.Households;
    testX[i, 7] = record.MedianIncome;
    
    testY[i, 0] = record.MedianHouseValue;
}

// Make predictions using TensorFlow operations
var predictions = sess.run(pred, new FeedItem(X, testX));

// Calculate evaluation metrics
// Convert to NumSharp arrays for vectorized operations (Python-like)
var y_true = np.array(testY.ToArray<float>());
var y_pred = np.array(predictions.ToArray<float>());

// Calculate metrics using vectorized operations (sklearn-like)
var mae = np.mean(np.abs(y_true - y_pred));
var mse = np.mean(np.power(y_true - y_pred, 2));
var rmse = np.sqrt(mse);

// Calculate R-squared (sklearn-like)
var y_mean = np.mean(y_true);
var ss_res = np.sum(np.power(y_true - y_pred, 2));
var ss_tot = np.sum(np.power(y_true - y_mean, 2));
var r2 = 1.0 - (ss_res / ss_tot);

Console.WriteLine($"Mean Absolute Error: ${mae:F0}");
Console.WriteLine($"Root Mean Squared Error: ${rmse:F0}");
Console.WriteLine($"R-Squared: {r2:F4}");
```

This code evaluates the trained TensorFlow.NET model on the test set by calculating predictions and computing regression metrics including MAE, RMSE, and R-squared. These metrics help assess how well the model generalizes to unseen data.


This code calculates the following metrics:

- **RSquared**: this is the coefficient of determination, a common evaluation metric for regression models. It tells you how well your model explains the variance in the data, or how good the predictions are compared to simply predicting the mean.
- **RootMeanSquaredError**: this is the root mean squared error or RMSE value. It's the go-to metric in the field of machine learning to evaluate models and rate their accuracy. RMSE represents the length of a vector in n-dimensional space, made up of the error in each individual prediction.
- **MeanSquaredError**: this is the mean squared error, or MSE value. Note that RMSE and MSE are related: RMSE is the square root of MSE.
- **MeanAbsoluteError**: this is the mean absolute prediction error.

Note that both RMSE and MAE are expressed in dollars. They can both be interpreted as a kind of 'average error' value, but the RMSE will respond much more strongly to large prediction errors. Therefore, if RMSE > MAE, it means the model struggles with some predictions and generates relatively large errors. 

You should get the following output:

![Regression Model Evaluation](../img/evaluate.jpg)
{ .img-fluid .mb-4 }

Let's analyze my results:

The R-squared value is **0.6013**. This means the model is able to explain ~60% of the variance in housing prices. This is not bad, it's capturing a majority of the pattern in the data, but still leaves ~40% unexplained. This indicates a moderate-to-strong fit, and for real-world housing data, we could consider this a reasonable result.

The mean absolute error (MAE) is **$42,760**. So on average, the model's predictions are off by about $42k. That's not bad at all, given that the most expensive house in our dataset is $500k.

The mean squared error (MSE) is **~3.65 billion**. Large errors are penalized much more heavily in this metric because of the squaring, so this large number indicates the presence of some high-error predictions.

The root mean squared error (RMSE) is **~$60,439**. This metric is similar to MAE, but gives more weight to large errors. The fact that the RMSE > MAE suggests that the model occasionally makes big errors in its predictions.

So how did your model do?

Compare your model with the target you set earlier. Did it make predictions that beat the target? Are you happy with the predictive quality of your model? Feel free to experiment with other learning algorithms to try and get a better result.
{ .homework }

#### Summary

Evaluation is an essential step in machine learning, because this is where we check if the predictions our model is making are any good. If our model cannot beat the target we set, we'll have to go back to the drawing board and tweak the data transformations or pick a different learning algorithm.

A target of $50k is perfectly reasonable for housing, it means you'll accept a 10% prediction error for houses worth $500k. And the SDCA learning algorithm combined with all the data transformations we discussed was able to beat that target.

Let's continue working on our machine learning app and add a couple more features.

{{< /encrypt >}}