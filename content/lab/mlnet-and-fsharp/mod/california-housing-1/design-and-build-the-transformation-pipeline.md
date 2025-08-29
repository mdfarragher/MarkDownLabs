---
title: "Design And Build The Transformation Pipeline"
type: "lesson"
layout: "default"
sortkey: 70
---

Now let's start designing the ML.NET data transformation pipeline. This is the sequence of feature engineering steps that will transform the dataset into something suitable for a machine learning algorithm to train on.

{{< encrypt >}}

#### Decide Feature Engineering Steps

After completing the previous lessons, you should have a pretty good idea which feature engineering steps are needed to get this dataset ready for machine learning training.

Write down all feature engineering steps you want to perform on the California Housing dataset, in order.
{ .homework }

Here are some steps you could consider:

-   Normalize every feature in the dataset
-   Remove outliers with very high **TotalRooms** or **Population** values
-   Remove the 'clipped' houses with a median house value > $499,999
-   Condense the **TotalRooms**, **TotalBedrooms**, **Population** and **Households** columns into one or two computed columns.
-   Bin and one-hot encode the **Latitude** and **Longitude**.

Which steps will you choose?

#### Implement The Transformation Pipeline

Now let's ask Copilot to implement our chosen data transformation steps with an ML.NET machine learning pipeline.

First, remove any code you don't need anymore (for example, the calls to the `CalculateCorrelationMatrix`, `PrintCorrelationMatrix` and `PlotCorrelationMatrix` methods).

Then, enter the following prompt in the Copilot panel:

"Implement the following data transformations by building a machine learning pipeline in F#:<br>- [your first transformation step]<br>- [your second transformation step]<br>- ..."
{ .prompt }

You should now have a nice data transformation pipeline that prepares your dataset for machine learning training. Let's take a look at the code.

#### Removing outliers

If you decided to remove outliers, for example by removing any rows that have **Population** > 5000 and **MedianHouseValue** > $499,999, your code should look like this:

```fsharp
// Filter out outliers with Population > 5000
let filteredData = mlContext.Data.FilterRowsByColumn(dataView, "population", upperBound = 5000.0)

// Filter out expensive houses with MedianHouseValue > 499999
let filteredData2 = mlContext.Data.FilterRowsByColumn(filteredData, "median_house_value", upperBound = 499999.0)
```

The `FilterRowsByColumn` method is a great tool to quickly filter a dataview by a specific column. You can specify upper- and lower bounds for filtering.

#### Adding computed columns

If you decided to create a new computed column like **RoomsPerPerson**, you'll notice a new class definition in your code:

```fsharp
// Type to hold transformed data
[<CLIMutable>]
type TransformedHousingData = {
    mutable longitude: float32
    mutable latitude: float32
    mutable housing_median_age: float32
    mutable median_income: float32
    mutable median_house_value: float32

    mutable rooms_per_person: float32
}
```

This is a new helper class that holds a single row in the transformed dataset, with an extra property for the **rooms_per_person** column. Also note the `mutable` keywords that make each field mutable, this is a requirement for what comes next.

Your pipeline would then be built as follows:

```fsharp
// Set up pipeline to transform data  
let pipeline = 
    EstimatorChain()

        // Add the rooms_per_person custom field
        .Append(mlContext.Transforms.CustomMapping<HousingData, TransformedHousingData>(
            (fun input output ->
                output.longitude <- input.longitude
                output.latitude <- input.latitude  
                output.housing_median_age <- input.housing_median_age
                output.median_income <- input.median_income
                output.median_house_value <- input.median_house_value
                output.rooms_per_person <- if input.population > 0.0f then input.total_rooms / input.population else 0.0f
            ),
            "RoomsPerPersonMapping"))
```

The `CustomMapping` transformation uses a function to convert data from `HousingData` to `TransformedHousingData` and add the new **rooms_per_person** column. Note that each field in `TransformedHousingData` has to be set to mutable for the assignments to work.

#### Bin- and one-hot encode latitude and longitude

If you decided to bin- and one-hot encode **latitude** and **longitude**, you'll notice two extra properties in the TransformedHousingData class:

```fsharp
// Type to hold transformed data
[<CLIMutable>]
type TransformedHousingData = {
    mutable longitude: float32
    mutable latitude: float32
    mutable housing_median_age: float32
    mutable median_income: float32
    mutable median_house_value: float32

    [<VectorType(10)>] mutable latitude_encoded: float32[]
    [<VectorType(10)>] mutable longitude_encoded: float32[]
}
```

These properties will hold the transformed latitude and longitude, after they have been converted to one-hot encoded vectors. Note that the properties have the attribute `VectorType` set, which indicates that these columns are 10-element `float[]` vectors.

In the main code, you'll find the following transformations:

```fsharp
// Bin and one-hot encode latitude and longitude
pipeline
    .Append(mlContext.Transforms.NormalizeBinning(
        outputColumnName = "latitude_binned",
        inputColumnName = "latitude",
        maximumBinCount = 10))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding(
        outputColumnName = "latitude_encoded",
        inputColumnName = "latitude_binned"))
    .Append(mlContext.Transforms.NormalizeBinning(
        outputColumnName = "longitude_binned",
        inputColumnName = "longitude",
        maximumBinCount = 10))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding(
        outputColumnName = "longitude_encoded",
        inputColumnName = "longitude_binned"))
```

The `NormalizeBinning` transformation bins the latitude and longitude columns into 10 bins of equal size, and `OneHotEncoding` performs one-hot encoding on these bin numbers to create a 10-element vector of zeroes and ones.

### Normalization

If you decided to normalize any columns in the dataset, it will look like this:

```fsharp
// Normalize all columns except Latitude, Longitude and MedianHouseValue
pipeline
    .Append(mlContext.Transforms.NormalizeMinMax("housing_median_age"))
    .Append(mlContext.Transforms.NormalizeMinMax("median_income"))
    .Append(mlContext.Transforms.NormalizeMinMax("rooms_per_person"))
```

This stack of transformations will normalize every column except **MedianHouseValue**, **Latitude** and **Longitude**.

These code examples are reference implementations of common data transformations in ML.NET. Compare the output of your AI agent with this code, and correct your agent if needed.
{ .tip }

To actually perform the transformations and get access to the transformed data, you'll need code like this:

```fsharp
// Apply the pipeline to the filtered data
let model = pipeline.Fit(filteredData2)
let transformedData = model.Transform(filteredData2)

// Convert to enumerable to verify transformations
let transformedHousingData = 
    mlContext.Data.CreateEnumerable<TransformedHousingData>(transformedData, reuseRowObject = false)
    |> Seq.toList
```

This code calls `Fit` to generate a machine learning model that implements the data transformation pipeline. The `Transform` method then uses this model to transform the original dataview into a new transformed dataview. Finally, the `CreateEnumerable` method converts the transformed dataview into a list of `TransformedHousingData` instances.

#### Test The Code

My Claude 3.7 agent added a bit of extra code after the pipeline to output a sample row from the transformed data. My run looked like this:

![Pipeline Run Output](../img/pipeline-run.png)
{ .img-fluid .mb-4 }

You can see that I decided to remove outliers by getting rid of all rows with a population larger than 5000, and all houses with a value greater than $499,999. There were 265 overpopulated and 831 overvalued housing blocks matching those conditions in the dataset. The new computed column **rooms_per_person** has a numeric range from 0.0019 to 1.0, this is because I normalized the columns. And in the sample row, you can clearly see that the latitude and longitude values have been one-hot encoded into 10-element numerical vectors.

Everything seems to be working.

#### Summary

In this lesson, you put on your data scientist hat and decided which data transformation steps to apply to the dataset. The AI agent then generated the corresponding MLNET pipeline code for you.

Coming up with the correct data transformations for any given dataset requires deep domain knowledge and a fair bit of intuition, and this is not something an agent can reliably do for you. Don't fall into the trap of asking the agent to come up with the transformations, because this is your job!

So always make a plan first, based on your analysis of the dataset. Then prompt the agent and ask it to follow your plan.

To wrap this lab up, let's see if we can create a cross product of the latitude and longitude vectors.

{{< /encrypt >}}