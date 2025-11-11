---
title: "Design And Build The Transformation Pipeline"
type: "lesson"
layout: "default"
sortkey: 70
---

Now let's start designing the TensorFlow.NET data transformation pipeline. This is the sequence of feature engineering steps that will transform the dataset into something suitable for a machine learning algorithm to train on.

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

Now let's ask Copilot to implement our chosen data transformation steps with a TensorFlow.NET machine learning pipeline.

First, remove any code you don't need anymore (for example, the calls to the `CalculateCorrelationMatrix`, `PrintCorrelationMatrix` and `PlotCorrelationMatrix` methods).

Then, enter the following prompt in the Copilot panel:

"Implement the following data transformations by building a machine learning pipeline:<br>- [your first transformation step]<br>- [your second transformation step]<br>- ..."
{ .prompt }

You should now have a nice data transformation pipeline that prepares your dataset for machine learning training. Let's take a look at the code.

#### Removing outliers

If you decided to remove outliers, for example by removing any rows that have **Population** > 5000 and **MedianHouseValue** > $499,999, your code should look like this:

```csharp
// Filter outliers using NumSharp boolean indexing (Python pandas-like)
var population_mask = population <= 5000;
var price_mask = median_house_value < 500000;
var combined_mask = population_mask & price_mask;

// Apply filter to all arrays simultaneously
longitude = longitude[combined_mask];
latitude = latitude[combined_mask];
housing_median_age = housing_median_age[combined_mask];
total_rooms = total_rooms[combined_mask];
total_bedrooms = total_bedrooms[combined_mask];
population = population[combined_mask];
households = households[combined_mask];
median_income = median_income[combined_mask];
median_house_value = median_house_value[combined_mask];

Console.WriteLine($"Filtered dataset shape: {longitude.shape}");
```

This code uses LINQ to filter out housing blocks with extreme population values and clipped house values, ensuring our model trains on clean, representative data.

#### Adding computed columns

If you decided to create a new computed column like **RoomsPerPerson**, you'll notice the following code in your project:

```csharp
// Feature engineering using vectorized operations (pandas-like)
var rooms_per_person = total_rooms / np.maximum(population, 1);  // Avoid division by zero
var bedrooms_per_household = total_bedrooms / np.maximum(households, 1);
var population_per_household = population / np.maximum(households, 1);

Console.WriteLine($"Created features: rooms_per_person, bedrooms_per_household, population_per_household");
```

This code uses NumSharp's vectorized operations to create new features, similar to pandas operations in Python. The np.maximum function prevents division by zero across entire arrays efficiently.

#### Bin- and one-hot encode latitude and longitude

If you decided to bin- and one-hot encode **Latitude** and **Longitude**, you'll see the following code:

```csharp
// Vectorized binning and one-hot encoding (sklearn-style)
var n_bins = 10;

// Create bins using NumSharp (like np.linspace)
var lat_bins = np.linspace(latitude.min(), latitude.max(), n_bins + 1);
var lon_bins = np.linspace(longitude.min(), longitude.max(), n_bins + 1);

// Digitize - find bin indices (like np.digitize)
var lat_indices = np.digitize(latitude, lat_bins) - 1;  // Subtract 1 for 0-based indexing
var lon_indices = np.digitize(longitude, lon_bins) - 1;

// Clip to valid range
lat_indices = np.clip(lat_indices, 0, n_bins - 1);
lon_indices = np.clip(lon_indices, 0, n_bins - 1);

// One-hot encoding using NumSharp (like sklearn.preprocessing.OneHotEncoder)
var lat_encoded = np.eye(n_bins)[lat_indices];  // Advanced indexing
var lon_encoded = np.eye(n_bins)[lon_indices];   // Creates one-hot vectors

Console.WriteLine($"Encoded latitude shape: {lat_encoded.shape}");
Console.WriteLine($"Encoded longitude shape: {lon_encoded.shape}");
```

This vectorized approach uses NumSharp operations that mirror scikit-learn's preprocessing tools: np.linspace for bin creation, np.digitize for binning, and advanced indexing with np.eye for one-hot encoding - all without explicit loops.

 
### Normalization

If you decided to normalize any columns in the dataset, it will look like this:

```csharp
// Vectorized feature scaling (sklearn StandardScaler/MinMaxScaler style)
var feature_matrix = np.column_stack(new[] { 
    longitude, latitude, housing_median_age, total_rooms, 
    total_bedrooms, population, households, median_income 
});

// Min-max scaling (equivalent to sklearn.preprocessing.MinMaxScaler)
var feature_min = feature_matrix.min(axis: 0, keepdims: true);
var feature_max = feature_matrix.max(axis: 0, keepdims: true);
var scaled_features = (feature_matrix - feature_min) / (feature_max - feature_min);

// Handle outliers with clipping
scaled_features = np.clip(scaled_features, 0.0f, 1.0f);

Console.WriteLine($"Scaled features shape: {scaled_features.shape}");
```

This code applies min-max normalization to scale all features to a [0,1] range. We cap extreme outliers to prevent them from skewing the normalization, which helps with model convergence during training.


These code examples are reference implementations of common data transformations in TensorFlow.NET. Compare the output of your AI agent with this code, and correct your agent if needed.
{ .tip }

To perform the transformations and get access to the transformed data, you'll need code like this:

```csharp
// Final data transformation for TensorFlow.NET
var finalData = normalizedData.Select(x => new
{
    Features = new float[] 
    {
        x.NormalizedLongitude,
        x.NormalizedLatitude,
        x.NormalizedAge,
        x.NormalizedIncome,
        x.NormalizedRooms,
        x.NormalizedBedrooms,
        x.NormalizedPopulation,
        x.NormalizedHouseholds
    }.Concat(x.LatitudeEncoded).Concat(x.LongitudeEncoded).ToArray(),
    Target = x.MedianHouseValue
}).ToList();

Console.WriteLine($"Final feature vector size: {finalData.First().Features.Length}");
```

This code combines all normalized features and one-hot encoded location vectors into a single feature array suitable for TensorFlow.NET training. Each record now has a standardized feature vector and target value.


#### Test The Code

My Claude 3.7 agent added a bit of extra code after the pipeline to output a sample row from the transformed data. My run looked like this:

![Pipeline Run Output](../img/pipeline-run.png)
{ .img-fluid .mb-4 }

You can see that I decided to remove outliers by getting rid of all rows with a population larger than 5000. There were 265 housing blocks matching that condition in the dataset. The new computed column **RoomsPerPerson** has a numeric range from 0.0019 to 1.0, this is because I normalized all columns, including this one. And in the sample row, you can clearly see that the latitude and longitude values have been one-hot encoded into 10-element numerical vectors.

Everything seems to be working.

#### Summary

In this lesson, you put on your data scientist hat and decided which data transformation steps to apply to the dataset. The AI agent then generated the corresponding TensorFlow.NET pipeline code for you.

Coming up with the correct data transformations for any given dataset requires deep domain knowledge and a fair bit of intuition, and this is not something an agent can reliably do for you. Don't fall into the trap of asking the agent to come up with the transformations, because this is your job!

So always make a plan first, based on your analysis of the dataset. Then prompt the agent and ask it to follow your plan.

To wrap this lab up, let's see if we can create a cross product of the latitude and longitude vectors.

{{< /encrypt >}}