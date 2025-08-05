---
title: "Train A Binary Classification Model"
type: "lesson"
layout: "default"
sortkey: 110
---

We're going to continue with the code we wrote in the previous lab. Our app sets up a pipeline to load the Cleveland CAD dataset and clean up the data using several feature engineering techniques.

So all that remains is to append a step to the end of the pipeline to train a binary classification model on the data.

#### Split The Dataset

But first, we need to split the dataset into two partitions: one for training and one for testing. The training partition is a randomly shuffled subset of 80% of all data, with the remaining 20% reserved for testing.

Open the Copilot panel and type the following prompt:

"Split the transformed data into two partitions: 80% for training and 20% for testing."
{ .prompt }

You should get the following code:

```csharp
// Split the data into training (80%) and testing (20%) datasets
var dataSplit = mlContext.Data.TrainTestSplit(transformedData, testFraction: 0.2);
var trainingData = dataSplit.TrainSet;
var testingData = dataSplit.TestSet;
```

The `TrainTestSplit` method splits a dataset into two parts, with the `testFraction` argument specifying how much data ends up in the second part.

#### Train The Model

Now let's add a machine learning algorithm to the pipeline.

"Create a binary classification pipeline that uses a learning algorithm to train a model on the training data partition. Use an algorithm that is well suited for the problem domain (healthcare, identifying patients with cardiovascular disease)"
{ .prompt }

You should now see a learning algorithm appended to your pipeline:

```csharp
// Add a binary classification trainer to the pipeline
Console.WriteLine("Adding binary classification trainer to pipeline...");
var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
    labelColumnName: "Diag", 
    featureColumnName: "Features");

var trainingPipeline = mlPipeline.Append(trainer);
```

In my case, the AI agent decided to use the L-BFGS logistic regression algorithm which apparently is well-suited for healthcare datasets. Unfortunately, the algorithm has a disadvantage: the scores it produces cannot be interpreted as probability values, which makes it difficult to interpret the predictions it makes.

Fortunately, there's a fix for that. A process called [Platt Calibration](https://en.wikipedia.org/wiki/Platt_scaling) can fit a logistic regression algorithm to the L-BFGS scores and restore the probabilities. Platt Calibration is available in the `Calibrators.Platt` pipeline step built into the ML.NET library:

```csharp
// Add calibrator for probability output
var calibratedPipeline = trainingPipeline
    .Append(mlContext.BinaryClassification.Calibrators.Platt(labelColumnName: "Diag"));

// Fit the pipeline to the data
var mlModel = calibratedPipeline.Fit(trainingData);
```

We now have a `calibratedPipeline` that produces reliable probability scores.

In the next lesson, we'll calculate the prediction evaluation metrics to find out how good the model is at predicting heart disease.
