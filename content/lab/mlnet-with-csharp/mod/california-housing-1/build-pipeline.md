---
title: "Design And Build The Transformation Pipeline"
type: "lesson"
layout: "default"
sortkey: 50
---

Now let's start designing the ML.NET data transformation pipeline. This is the sequence of feature engineering steps that will transform the dataset into something suitable for a machine learning algorithm to train on.

#### Decide Feature Engineering Steps

After completing the previous lessons, you should have a pretty good idea which feature engineering steps are needed to get this dataset ready for machine learning training.

Homework: write down all feature engineering steps you want to perform on the California Housing dataset, in order.
{ .homework }

Here are some steps you could consider:

-   Normalize every feature in the dataset
-   Remove outliers with very high **TotalRooms** or **Population** values
-   Remove the 'clipped' houses with a median house value > $499,999
-   Condense the **TotalRooms**, **TotalBedrooms**, **Population** and **Households** columns into one or two computed columns.
-   Bin and one-hot encode the **Latitude** and **Longitude**.

Which steps will you choose?

#### Implement The Transformation Pipeline

Now let's ask our AI agent to implement our chosen data transformation steps with an ML.NET machine learning pipeline.

First, remove any code you don't need anymore (for example, the calls to the `CalculateCorrelationMatrix`, `PrintCorrelationMatrix` and `PlotCorrelationMatrix` methods).

Then, enter the following prompt in the Copilot panel:

"Implement the following data transformations by building a machine learning pipeline:<br>- [your first transformation step]<br>- [your second transformation step]<br>- ..."
{ .prompt }

You should now have a nice data transformation pipeline that prepares your dataset for machine learning training. Let's take a look at the code.

#### Removing outliers

If you decided to remove outliers, for example by removing any rows that have **Population** > 5000 and **MedianHouseValue** > $499,999, your code should look like this:

```csharp
// Filter out outliers with Population > 5000
var filteredData = mlContext.Data.FilterRowsByColumn(dataView, nameof(HousingData.Population), upperBound: 5000);

// Filter out expensive houses with MedianHouseValue > 499999
filteredData = mlContext.Data.FilterRowsByColumn(filteredData, nameof(HousingData.MedianHouseValue), upperBound: 499999);
```

The `FilterRowsByColumn` method is a handy tool to quickly filter a dataview by a specific column. You can specify upper- and lower bounds for filtering.

#### Adding computed columns

If you decided to create a new computed column, for example **RoomsPerPerson**, you'll notice a new class definition in your code:

```csharp
// Class to hold transformed data including the computed column
public class TransformedHousingData : HousingData
{
    public float RoomsPerPerson { get; set; }
}
```

This is a new helper class that holds a single row in the transformed dataset, with an extra property for the **RoomsPerPerson** column.

Your pipeline would then be built as follows:

```csharp
// Compute RoomsPerPerson
var pipeline = mlContext.Transforms.CustomMapping<HousingData, TransformedHousingData>(
    (input, output) => 
    {
        output.Longitude = input.Longitude;
        output.Latitude = input.Latitude;
        output.HousingMedianAge = input.HousingMedianAge;
        output.TotalRooms = input.TotalRooms;
        output.TotalBedrooms = input.TotalBedrooms;
        output.Population = input.Population;
        output.Households = input.Households;
        output.MedianIncome = input.MedianIncome;
        output.MedianHouseValue = input.MedianHouseValue;
        output.RoomsPerPerson = input.Population > 0 ? input.TotalRooms / input.Population : 0;
    },
    "RoomsPerPersonMapping")
```

The `CustomMapping` transformation uses two class types and a lambda expression to transform the original data and add the new **RoomsPerPerson** column.

#### Bin- and one-hot encode latitude and longitude

If you decided to bin- and one-hot encode **Latitude** and **Longitude**, you'll notice two extra properties in the TransformedHousingData class:

```csharp
// Class to hold transformed data including the computed column
public class TransformedHousingData : HousingData
{
    ...
    
    // Added properties for transformed columns with nullable arrays
    [VectorType(10)]
    public float[]? LatitudeEncoded { get; set; }
    
    [VectorType(10)]
    public float[]? LongitudeEncoded { get; set; }
}
```

These properties will hold the transformed latitude and longitude, after they have been converted to one-hot encoded vectors.

Note that the properties have the attribute `VectorType` set, which indicates that these columns are 10-element `float[]` vectors.

In the main code, you'll find the following transformations:

```csharp
// Bin and one-hot encode Latitude and Longitude
.Append(mlContext.Transforms.NormalizeBinning(
    outputColumnName: "LatitudeBinned",
    inputColumnName: nameof(HousingData.Latitude),
    maximumBinCount: 10))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(
    outputColumnName: nameof(TransformedHousingData.LatitudeEncoded),
    inputColumnName: "LatitudeBinned"))    
.Append(mlContext.Transforms.NormalizeBinning(
    outputColumnName: "LongitudeBinned",
    inputColumnName: nameof(HousingData.Longitude),
    maximumBinCount: 10))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(
    outputColumnName: nameof(TransformedHousingData.LongitudeEncoded),
    inputColumnName: "LongitudeBinned"))
```

The `NormalizeBinning` transformation bins the latitude and longitude columns into 10 bins of equal size, and `OneHotEncoding` performs one-hot encoding on these bin numbers to create a 10-element vector of zeroes and ones.

### Normalization

If you decided to normalize any columns in the dataset, it will look like this:

```csharp
// Normalize all columns except Latitude, Longitude and MedianHouseValue
.Append(mlContext.Transforms.NormalizeMinMax(nameof(HousingData.HousingMedianAge)))
.Append(mlContext.Transforms.NormalizeMinMax(nameof(HousingData.TotalRooms)))
.Append(mlContext.Transforms.NormalizeMinMax(nameof(HousingData.TotalBedrooms)))
.Append(mlContext.Transforms.NormalizeMinMax(nameof(HousingData.Population)))
.Append(mlContext.Transforms.NormalizeMinMax(nameof(HousingData.Households)))
.Append(mlContext.Transforms.NormalizeMinMax(nameof(HousingData.MedianIncome)))
.Append(mlContext.Transforms.NormalizeMinMax(nameof(TransformedHousingData.RoomsPerPerson)));
```

This stack of transformations will normalize every column except **MedianHouseValue**, **Latitude** and **Longitude**.

These code examples are reference implementations of common data transformations in ML.NET. Compare the output of your AI agent with this code, and correct your agent if needed.
{ .tip }

To actually perform the transformations and get access to the transformed data, you'll need code like this:

```csharp
// Apply the pipeline to the filtered data
Console.WriteLine("Applying transformations...");
var transformModel = pipeline.Fit(filteredData);
var transformedData = transformModel.Transform(filteredData);

// Convert to enumerable to verify transformations
var transformedHousingData = mlContext.Data.CreateEnumerable<TransformedHousingData>(
    transformedData, reuseRowObject: false).ToList();
```

This code calls `Fit` to generate a machine learning model that implements the data transformation pipeline. The `Transform` method then uses this model to transform the original dataview into a new transformed dataview. Finally, the `CreateEnumerable` method converts the transformed dataview into a list of `TransformedHousingData` instances.

#### Test The Code

My Claude 3.7 agent added a bit of extra code after the pipeline to output a sample row from the transformed data. My run looked like this:

![Pipeline Run Output](../img/pipeline-run.png)
{ .img-fluid .mb-4 }

You can see that I decided to remove outliers by getting rid of all rows with a population larger than 5000. There were 265 housing blocks matching that condition in the dataset.

The new computed column **RoomsPerPerson** has a numeric range from 0.0019 to 1.0, this is because I normalized all columns, including this one.

And in the sample row, you can clearly see that the latitude and longitude values have been one-hot encoded into 10-element numerical vectors.

Everything seems to be working.

#### Summary

In this lesson, you put on your data scientist hat and made a plan that specified which transformations to create and in what order to apply them. The agent then generated the code for you.

Coming up with the correct data transformations for any given dataset requires deep domain knowledge and a fair bit of intuition, and this is not something an agent can do for you. Don't fall into the trap of asking the agent to come up with transformations on its own. This is your job!

So always make a plan first, based on your analysis of the dataset. Then prompt the agent and ask it to follow your plan.

To wrap this lab up, let's see if we can create a cross product of the latitude and longitude vectors.