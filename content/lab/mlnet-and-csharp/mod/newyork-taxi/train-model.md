---
title: "Train A Regression Model"
type: "lesson"
layout: "default"
sortkey: 110
---

We're going to continue with the code we wrote in the previous lab. That C# application set up an ML.NET pipeline to load the New York TLC dataset and clean up the data using several feature engineering techniques.

So all we need to do is append a few command to the end of the pipeline to train and evaluate a regression model on the data.

#### Split The Dataset

But first, we need to split the dataset into two partitions: one for training and one for testing. The training partition is typically a randomly shuffled subset of around 80% of all data, with the remaining 20% reserved for testing.

Open the Copilot panel and type the following prompt:

"Split the transformed data into two partitions: 80% for training and 20% for testing."
{ .prompt }

You should get the following code:

```csharp
// Split the data into training (80%) and testing (20%) datasets
var dataSplit = mlContext.Data.TrainTestSplit(transformedDataWithFeatures, testFraction: 0.2);
var trainingData = dataSplit.TrainSet;
var testingData = dataSplit.TestSet;
```

The `TrainTestSplit` method splits a dataset into two parts, with the `testFraction` argument specifying how much data ends up in the second part.

#### Train The Model Using SDCA

Now let's add a machine learning algorithm to the pipeline.

"Create a regression pipeline that uses the SDCA algorithm to train a model on the 80% training data partition."
{ .prompt }

You should now see the SDCA algorithm at the end of your pipeline:

```csharp
// Train model with SDCA algorithm
.Append(mlContext.Regression.Trainers.Sdca(
    labelColumnName: "FareAmount",
    featureColumnName: "Features"));
```

And the `Fit` and `Transform` code should now look like this:

```csharp
// Train the model on the data
var model = mlPipeline.Fit(trainingData);
```

This code trains the model on the 80% training data partition.

In the next lesson, we'll calculate the prediction evaluation metrics to find out how good the model is at predicting taxi fares.
