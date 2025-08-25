---
title: "Design And Build The Transformation Pipeline"
type: "lesson"
layout: "default"
sortkey: 50
---

Now let's start designing the Accord.NET data transformation pipeline. This is the sequence of feature engineering steps that will transform the dataset into something suitable for a machine learning algorithm to train on.

{{< encrypt >}}

#### Decide Feature Engineering Steps

Actually there's only one step we can do:

- Normalize the pixel values

Because nothing else is applicable here. There are no outliers, so we don't need to filter the data. There are no numerical features we can bin or categorical features we can one-hot encode. And there is no class imbalance, so there's no need for over- or undersampling. 

In fact, the only transformation we usually do for image datasets is scaling (and optionally converting to grayscale). We don't use any of the usual feature engineering steps. 

#### Implement The Transformation Pipeline

Let's ask Copilot to add a normalization step to the machine learning pipeline. Enter the following prompt in the Copilot panel:

"Add a normalization step to the machine learning pipeline to normalize the pixel values"
{ .prompt }

Let's take a look at the code. The pipeline will look like this:

```csharp
// Step 1: Convert Label to Key (categorical)
var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
    // Step 2: Normalize pixel values from 0-255 to 0-1 range
    .Append(mlContext.Transforms.NormalizeMinMax("PixelValues", "PixelValues"))
    // Step 3: Add a multiclass classifier (using SDCA)
    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "PixelValues"))
    // Step 4: Convert prediction back to original values
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
```

This code uses `MapValueToKey` to convert the **Label** column to a key value that a learning algorithm can work with. Then the `NormalizeMinMax` method normalizes the **PixelValues** column and we select the `SdcaMaximumEntropy` algorithm to train the machine learning model. Finally, we need a closing `MapKeyToValue` step to convert any predicted keys back to their corresponding label values.

With multiclass classification, we always need pipelines that start with `MapValueToKey` and end with `MapKeyToValue`. This removes the need for one-hot encoding the label, because a 'key' in Accord.NET is comparable to a one-hot encoded column.

Your AI agent may have used a different learning algorithm in your pipeline, but there's a good chance it picked SDCA too. LLMs know about many public machine learning datasets, and will usually pick the best learning algorithm for the job at hand. 

#### Split The Dataset

If your AI agent is smart, it will have also generated code to split the dataset, train a model, generate predictions for the test partition and evaluate the quality of the predictions. My Claude 4.0 agent did all that when I only asked for the normalization step. 

If you don't have the code to split your dataset yet, feel free to prompt your AI agent now: 

"Split the transformed data into two partitions: 80% for training and 20% for testing."
{ .prompt }

You should get the following code:

```csharp
// Split data into training and test sets
var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
var trainData = trainTestSplit.TrainSet;
var testData = trainTestSplit.TestSet;
```

The `TrainTestSplit` method splits a dataset into two parts, with the `testFraction` argument specifying how much data ends up in the second part.

#### Run The Pipeline And Generate Predictions

And finally, you'll see the following code to perform the transformations and get access to the transformed data:

```csharp
// Train the model
var model = pipeline.Fit(trainData);

// Make predictions on test set
var predictions = model.Transform(testData);
```

This code calls `Fit` to generate a machine learning model that implements the pipeline. The `Transform` method then uses this model to generate predictions for each image in the `testData` partition. 

Now we're ready to calculate the classification metrics. 

{{< /encrypt >}}
