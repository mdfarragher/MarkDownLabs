---
title: "Design And Build The Transformation Pipeline"
type: "lesson"
layout: "default"
sortkey: 50
---

Now let's start designing the ML.NET data transformation pipeline. This is the sequence of feature engineering steps that will transform the dataset into something suitable for a machine learning algorithm to train on.

#### Decide Feature Engineering Steps

After completing the previous lessons, you should have a pretty good idea which feature engineering steps are needed to get this dataset ready for machine learning training.

You're already performing these transformations:

- Add a new column with the trip duration
- Remove trips with a duration > 60 minutes
- Remove trips with a negative fare or > 100 dollars

Here are some additional steps you could consider:

- Normalize features
- Remove trips with a distance > 15 miles
- Remove trips with 0 passengers
- Remove trips with tips > 15 dollars
- One-hot encode rate code ID
- One-hot encode payment type

And the correlation matrix showed that the columns **VendorID**, **PassengerCount**, **PULocationID** and **DOLocationID** are very weakly correlated with the label, so you could consider leaving them out of the training data.

Which steps will you choose?

Write down all feature engineering steps you want to perform on the New York TLC dataset, in order.
{ .homework }

#### Implement The Transformation Pipeline

Now let's ask Copilot to implement our chosen data transformation steps with an ML.NET machine learning pipeline. Enter the following prompt in the Copilot panel:

"Implement the following data transformations by building a machine learning pipeline:<br>- [your first transformation step]<br>- [your second transformation step]<br>- ..."
{ .prompt }

You should now have a nice data transformation pipeline that prepares your dataset for machine learning training. Let's take a look at the code.

#### Filter outliers

If you decided to remove outliers, your code should look like this:

```csharp
// Filter outliers
var filteredData = mlContext.Data.FilterRowsByColumn(transformedData, "TripDuration", upperBound: 60);
filteredData = mlContext.Data.FilterRowsByColumn(filteredData, "FareAmount", lowerBound: 0.01, upperBound: 100);
filteredData = mlContext.Data.FilterRowsByColumn(filteredData, "TripDistance", upperBound: 15);
filteredData = mlContext.Data.FilterRowsByColumn(filteredData, "TipAmount", upperBound: 15);
filteredData = mlContext.Data.FilterByCustomPredicate<TaxiTripWithDuration>(filteredData, 
    row => row.PassengerCount >= 1);
```

This code uses `FilterRowsByColumn` to filter all columns of type `float`, and `FilterByCustomPredicate` to filter **PassengerCount** to exclude trips with zero passengers.

### Normalize Features

If you decided to normalize any features in the dataset, it will look like this:

```csharp
// Build ML pipeline with data transformations
var mlPipeline = mlContext.Transforms.Concatenate(
        "NumericFeatures",
        "TripDistance",
        "TripDuration")

    // Normalize numeric features
    .Append(mlContext.Transforms.NormalizeMinMax(
        outputColumnName: "NormalizedFeatures",
        inputColumnName: "NumericFeatures"))
```

This code uses `Concatenate` to combine all numeric features (just **TripDistance** and **TripDuration** in my case) into a new combined feature called **NumericFeatures**. The `NormalizeMinMax` method then normalizes these features into a new **NormalizedFeatures** column.

#### One-Hot Encode Categories

If you decided to one-hot encode **RatecodeID** and **PaymentType**, you'll see the following code:

```csharp
// One-hot encode RatecodeID
.Append(mlContext.Transforms.Categorical.OneHotEncoding(
    outputColumnName: "RatecodeIDEncoded",
    inputColumnName: "RatecodeID"))

// One-hot encode PaymentType
.Append(mlContext.Transforms.Categorical.OneHotEncoding(
    outputColumnName: "PaymentTypeEncoded",
    inputColumnName: "PaymentType"))
    
// Combine all features into a single feature vector
.Append(mlContext.Transforms.Concatenate(
    "Features", 
    "NormalizedFeatures", 
    "RatecodeIDEncoded", 
    "PaymentTypeEncoded"));
```
The `OneHotEncoding` methods perform one-hot encoding on **RatecodeID** and **PaymentType**, and **Concatenate** combines the encoded features and the **NormalizedFeatures** column set up earlier into one new column called **Features**.

These code examples are reference implementations of common data transformations in ML.NET. Compare the output of your AI agent with this code, and correct your agent if needed.
{ .tip }

And finally, you'll see some code to actually perform the transformations and get access to the transformed data:

```csharp
// Apply the feature engineering transformations
var model = mlPipeline.Fit(filteredData);
var transformedDataWithFeatures = model.Transform(filteredData);
```

This code calls `Fit` to generate a machine learning model that implements the pipeline. The `Transform` method then uses this model to transform the original dataview into a new transformed dataview with all data transformations applied. 

Now we're ready to add a regression learning algorithm to the machine learning pipeline, so that we can train the model on the data and calculate the regression metrics. 