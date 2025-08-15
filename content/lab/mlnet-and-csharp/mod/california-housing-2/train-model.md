---
title: "Train A Regression Model"
type: "lesson"
layout: "default"
sortkey: 10
---

We're going to continue with the code we wrote in the previous lab. That C# application set up an ML.NET pipeline to load the California Housing dataset and clean up the data using several feature engineering techniques.

So all we need to do is append a few command to the end of the pipeline to train and evaluate a regression model on the data.

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

```csharp
// Split the filtered data into training (80%) and testing (20%) partitions
var trainTestSplit = mlContext.Data.TrainTestSplit(filteredData, testFraction: 0.2);
var trainingData = trainTestSplit.TrainSet;
var testingData = trainTestSplit.TestSet;
```

The `TrainTestSplit` method splits a dataset into two parts, with the `testFraction` argument specifying how much data ends up in the second part.

#### Add A Machine Learning Algorithm

Now let's add a machine learning algorithm to the pipeline.

"Add a linear regression algorithm to the pipeline."
{ .prompt }

That should produce the following code:

```csharp
// Combine all features into a single vector column
.Append(mlContext.Transforms.Concatenate("Features",
    nameof(HousingData.HousingMedianAge),
    nameof(HousingData.MedianIncome),
    nameof(TransformedHousingData.RoomsPerPerson),
    nameof(TransformedHousingData.LocationCrossProduct))
)

// Add a linear regression trainer to the pipeline
.Append(mlContext.Regression.Trainers.Sdca(labelColumnName: nameof(HousingData.MedianHouseValue), featureColumnName: "Features"));
```

This code adds two new components to the pipeline:

- `Concatenate` which combines all features into a single column called Features. This is a required step because ML.NET can only train on a single input column.
- An `Sdca` regression trainer which will train the model to make accurate predictions.

Be careful when you run this prompt! My AI agent generated a Concatenate step that included all features, including **MedianHouseValue**, **Latitude**, **Longitude**, **LatitudeEncoded**, **LongitudeEncoded**, **TotalRooms**, **TotalBedrooms**, **Population** and **Househoulds**.

This is obviously wrong, as **LocationCrossProduct** replaces all other latitude and longitude columns, and **RoomsPerPerson** replaces all other room- and person-related columns.

Even worse, did you notice the **MedianHouseValue** column in that list? This is the label that we're trying to predict. If we train a model on the label itself, the model can simply ignore all other features and output the label directly. This is like asking the model to make a prediction, and then giving it the actual answer it is supposed to predict. 

So I had to manually edit the list of columns to fix this.

Always be vigilant. AI agents can easily make mistakes like this, because they do not understand the meaning of each dataset column. Your job as a data scientist is to make sure that the generated code does not contain any bugs.
{ .tip }

By the way, SDCA is an optimized stochastic variance reduction algorithm that converges very quickly on an optimal solution. If you're interested, you can read more about the algorithm on Wikipedia:

https://en.wikipedia.org/wiki/Stochastic_variance_reduction#SDCA


#### Train A Machine Learning Model

Now let's train a machine learning model using our data transformation pipeline and the SDCA learning algorithm:

"Train a machine learning model on the training set using the pipeline."
{ .prompt }

That will produce the following code:

```csharp
// Train the model using the training partition
Console.WriteLine("Training the model...");
var model = pipeline.Fit(trainingData);
Console.WriteLine("Model training completed.");
```

The `Fit` method is all you need, it will return a fully trained machine learning model, which has been trained on the specified data using the data transformation pipeline.

#### Summary

In this lesson, you completed the machine learning pipeline you built in the previous lesson, by adding a machine learning algorithm. Then you split the data into a training and testing set, and trained a model on the training set.

In the next lesson, we'll calculate the prediction evaluation metrics to find out how good the model is at predicting house prices.
