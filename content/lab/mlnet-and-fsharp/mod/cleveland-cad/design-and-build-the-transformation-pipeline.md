---
title: "Design And Build The Transformation Pipeline"
type: "lesson"
layout: "default"
sortkey: 80
---

# Design And Build The Transformation Pipeline

Now let's start designing the ML.NET data transformation pipeline. This is the sequence of feature engineering steps that will transform the dataset into something suitable for a machine learning algorithm to train on.

{{< encrypt >}}

#### Decide Feature Engineering Steps

After completing the previous lessons, you should have a pretty good idea which feature engineering steps are needed to get this dataset ready for machine learning training.

You're already performing these transformations:

- Replace missing values for **NumMajorVessels** and **Thalassemia**
- Remove patients with cholesterol levels > 400
- Remove patients with blood pressure > 180
- Remove patients with max heart rate < 80

Here are some additional steps you could consider:

- Normalize the numerical features
- Undersample male patients to remove the sex bias
- One-hot encode all categorical columns

Which steps will you choose?

Write down all feature engineering steps you want to perform on the Cleveland CAD dataset, in order.
{ .homework }

#### Implement The Transformation Pipeline

Now let's ask Copilot to implement our chosen data transformation steps with an ML.NET machine learning pipeline. Enter the following prompt in the Copilot panel:

"Implement the following data transformations by extending the machine learning pipeline in F#:<br>- [your first transformation step]<br>- [your second transformation step]<br>- ..."
{ .prompt }

You should now have a nice data transformation pipeline that prepares your dataset for machine learning training. Let's take a look at the code.

#### Filter outliers

If you decided to remove outliers, your code should look like this (you probably had this code already):

```fsharp
// Filter outliers
let filteredData = mlContext.Data.FilterRowsByColumn(rawDataView, "Chol", upperBound = 400.0)
let filteredData = mlContext.Data.FilterRowsByColumn(filteredData, "TrestBps", upperBound = 180.0)
let filteredData = mlContext.Data.FilterRowsByColumn(filteredData, "Thalac", lowerBound = 80.0)
```

This code uses `FilterRowsByColumn` to filter all numeric columns.

### Normalize Features

If you decided to normalize any features in the dataset, it will look like this:

```fsharp
// Create a new ML pipeline for feature engineering
let mlPipeline = mlContext.Transforms.Concatenate(
    "NumericFeatures", "Age", "TrestBps", "Chol", "Thalac", "OldPeak")
    
    // Normalize numeric features
    .Append(mlContext.Transforms.NormalizeMinMax("NormalizedNumericFeatures", "NumericFeatures"))
```

This code uses `Concatenate` to combine all numeric features into a new combined feature called **NumericFeatures**. The `NormalizeMinMax` method then normalizes these features into a new **NormalizedNumericFeatures** column.

#### One-Hot Encode Categories

If you decided to one-hot encode the categorical columns, you'll see the following code:

```fsharp
// One-hot encode categorical features
.Append(mlContext.Transforms.Categorical.OneHotEncoding("SexEncoded", "Sex"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("CpEncoded", "Cp"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("FbsEncoded", "Fbs"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("RestEcgEncoded", "RestEcg"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("ExangEncoded", "Exang"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("SlopeEncoded", "Slope"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("CaEncoded", "Ca"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("ThalEncoded", "Thal"))
    
// Combine all features into a single feature vector
.Append(mlContext.Transforms.Concatenate(
    "Features", 
    "NormalizedNumericFeatures",
    "SexEncoded", "CpEncoded", "FbsEncoded", "RestEcgEncoded", 
    "ExangEncoded", "SlopeEncoded", "CaEncoded", "ThalEncoded"))
```
The `OneHotEncoding` methods perform one-hot encoding on all categorical features, and **Concatenate** combines the encoded features and the **NormalizedNumericFeatures** column set up earlier into one new column called **Features**.

#### Undersample Male Patients

There is no built-in pipeline stage in ML.NET to undersample a feature, but instead it can be done with F# list operations on the dataview, like this:

```fsharp
// Shuffle patients for sampling
let shuffledPatients = mlContext.Data.ShuffleRows(filteredData)

// Convert patients to enumerable for sampling
let unbalancedList = 
    mlContext.Data.CreateEnumerable<HeartDataInput>(shuffledPatients, reuseRowObject = false)
    |> List.ofSeq

// Group patients by sex
let groupedData = 
    unbalancedList 
    |> List.groupBy (fun p -> p.Sex)
let minority = groupedData |> List.minBy (fun (_, patients) -> patients.Length)
let majority = groupedData |> List.maxBy (fun (_, patients) -> patients.Length)

// Undersample males and combine with females
let balancedData = 
    (snd majority)
    |> List.take (snd minority |> List.length)
    |> List.append (snd minority)

// Create new IDataView
let balancedView = mlContext.Data.LoadFromEnumerable(balancedData)
```

This code shuffles the dataset randomly with `ShuffleRows`, then creates a list of patients and groups them by sex. Then the code takes a sample of the majoriy (male patients) by calling `List.take`, and uses `List.append` to combine the undersampled patients with the full list of female patients. Finally, a call to `LoadFromEnumerable` converts the list back to a dataview. 

This will produce a new dataview with an equal number of male and female patients. 

If you want, you can calculate the histogram of the **Sex** column right after undersampling the male patients. It should look like this:

![Histogram Of Sex After Undersampling](../img/histogram-sex.png)
{.img-fluid .mb-4}

#### Run The Pipeline

And finally, you'll see some code to actually perform the transformations and get access to the transformed data:

```fsharp
// Fit the pipeline to the data
let mlModel = mlPipeline.Fit(transformedData)

// Transform the data
let transformedMLData = mlModel.Transform(transformedData)
```

This code calls `Fit` to generate a machine learning model that implements the pipeline. The `Transform` method then uses this model to transform the original dataview into a new transformed dataview with all data transformations applied. 

Now we're ready to add a binary classification learning algorithm to the machine learning pipeline, so that we can train the model on the data and calculate the classification metrics. 

{{< /encrypt >}}