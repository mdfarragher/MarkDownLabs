---
title: "Train A Regression Model"
type: "lesson"
layout: "default"
sortkey: 10
---

We're going to continue with the code we wrote in the previous lab. That F# application set up an ML.NET pipeline to load the California Housing dataset and clean up the data using several feature engineering techniques. All we need to do is append a few command to the end of the pipeline to train and evaluate a regression model on the data.

{{< encrypt >}}

#### Split The Dataset

But first, we need to split the dataset into two partitions: one for training and one for testing. The training partition is typically a randomly shuffled subset of around 80% of all data, with the remaining 20% reserved for testing.

We do this, because sometimes a machine learning model will memorize all the labels in a dataset, instead of learning the subtle patterns hidden in the data itself. When this happens, the model will produce excellent predictions for all the data it has been trained on, but struggle with data it has never seen before.

By keeping 20% of our data hidden from the model, we can check if this unwanted process of memorization (called **overfitting**) is actually happening.

So let's split our data into an 80% partition for training and a 20% partition for testing.

Open the code from the previous lesson in Visual Studio Code. Keep the data transformation pipeline intact, but remove any other code you don't need anymore.

Then open the Copilot panel and type the following prompt:

"Split the data into two partitions: 80% for training and 20% for testing."
{ .prompt }

You should get the following code:

```fsharp
// Split the filtered data into training (80%) and testing (20%) partitions
let trainTestData = mlContext.Data.TrainTestSplit(filteredData2, testFraction = 0.2)
let trainData = trainTestData.TrainSet
let testData = trainTestData.TestSet
```

The `TrainTestSplit` method splits a dataset into two parts, with the `testFraction` argument specifying how much data ends up in the second part.

#### Add A Machine Learning Algorithm

Now let's add a machine learning algorithm to the pipeline.

"Add a linear regression algorithm to the pipeline."
{ .prompt }

That should produce the following code:

```fsharp
// feature columns to train on
let featureColumns =
    [| "housing_median_age"; "median_income"; "rooms_per_person"; "location_cross" |]

// set up training pipeline
let regressionPipeline =
    pipeline
        .Append(mlContext.Transforms.Concatenate("Features", featureColumns))
        .Append(mlContext.Regression.Trainers.Sdca(labelColumnName = "median_house_value", featureColumnName = "Features"))
```

This code sets up a new `regressionPipeline` and adds two new components to it:

- `Concatenate` which combines all features from the `featureColumns` array into a single column called Features. This is a required step because ML.NET can only train on a single input column.
- An `Sdca` regression trainer which will train the model to make accurate predictions.

SDCA is an optimized stochastic variance reduction algorithm that converges very quickly on an optimal solution. If you're interested, you can read more about the algorithm on Wikipedia:

https://en.wikipedia.org/wiki/Stochastic_variance_reduction#SDCA


#### Train A Machine Learning Model

Now let's train a machine learning model using our data transformation pipeline and the SDCA learning algorithm:

"Train a machine learning model on the training set using the pipeline."
{ .prompt }

That will produce the following code:

```fsharp
// Train the model
let regressionModel = regressionPipeline.Fit(trainData)
```

The `Fit` method is all you need, it will return a fully trained machine learning model, which has been trained on the specified data using the data transformation pipeline.

#### Summary

In this lesson, you completed the machine learning pipeline you built in the previous lesson, by adding a machine learning algorithm. Then you split the data into a training and testing set, and trained a model on the training set.

In the next lesson, we'll calculate the prediction evaluation metrics to find out how good the model is at predicting house prices.

{{< /encrypt >}}