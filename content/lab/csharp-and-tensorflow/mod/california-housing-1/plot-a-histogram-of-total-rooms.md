---
title: "Plot A Histogram Of Total Rooms"
type: "lesson"
layout: "default"
sortkey: 40
---

Before you build a machine learning model, it's important to understand your data visually. Just looking at the numbers, like you did in the previous lesson, may not be enough. A good chart can clearly reveal patterns in the dataset.

In this section, you'll going to build a visualization to detect any outliers in the **total_rooms** feature.

Let's get started.

{{< encrypt >}}

#### Install ScottPlot

ScottPlot is a very nice plotting and visualization library for C# and NET that can go toe-to-toe with Python libraries like matplotlib and seaborn. We will use it in these labs whenever we want to visualize a dataset.

Since we've already installed ScottPlot and other dependencies, let's set up our development environment.

Then open Visual Studio Code in the current folder, like this:

```bash
code .
```

Open the Program.cs file and remove all existing content, because we don't want the agent to get confused. Replace the content with this:

```csharp
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using ScottPlot;
using NumSharp;
using Pandas.NET;
using Tensorflow;
using static Tensorflow.Binding;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using BetterConsoleTables;
```

#### Create a Histogram of total_rooms

Now let's ask Copilot to write the code for us.

At the bottom of the Copilot panel in Visual Studio Code, make sure the AI mode is set to 'Agent'. Then select your favorite model. I like to use GPT 4.1 or Claude 3.7 for coding work.

![Enable Agent Mode](../img/agent-mode.jpg)
{ .img-fluid .mb-4 .border }

Now enter the following prompt:

"Write C# code using ScottPlot and Pandas.NET to load the California-Housing.csv file and generate a histogram of the total_rooms column."
{ .prompt }

And let Copilot write the code for you.

You should see the following data loading code in your project:
```csharp
// Create a data structure to hold housing data
public class HousingData
{
    public int RowID { get; set; }
    public float Longitude { get; set; }
    public float Latitude { get; set; }
    public float HousingMedianAge { get; set; }
    public float TotalRooms { get; set; }
    public float TotalBedrooms { get; set; }
    public float Population { get; set; }
    public float Households { get; set; }
    public float MedianIncome { get; set; }
    public float MedianHouseValue { get; set; }
}

// Load CSV using Pandas.NET - Python-like syntax
var df = pd.read_csv("California-Housing.csv");

// Extract columns as NumSharp arrays (like pandas Series)
var longitude = df["longitude"].values;
var latitude = df["latitude"].values;
var housing_median_age = df["housing_median_age"].values;
var total_rooms = df["total_rooms"].values;
var total_bedrooms = df["total_bedrooms"].values;
var population = df["population"].values;
var households = df["households"].values;
var median_income = df["median_income"].values;
var median_house_value = df["median_house_value"].values;

Console.WriteLine($"Dataset shape: {df.shape}");
Console.WriteLine($"Columns: {string.Join(", ", df.columns)}");
```

This code loads the CSV using pandas-like syntax and extracts each column as a NumSharp array (equivalent to pandas Series). This approach is much more Python-like and enables vectorized operations throughout the pipeline.

TensorFlow.NET integrates seamlessly with NumSharp for tensor operations, Pandas.NET for data loading, and MathNet.Numerics for statistical computations. This combination provides Python-like vectorized operations with C# performance and type safety.
{ .tip }
 
The code to plot the histogram should look like this:
```csharp
// Python-like plotting with matplotlib-style API
import matplotlib.pyplot as plt  // Conceptual - using ScottPlot with matplotlib-like syntax

// Create histogram - matplotlib style
var fig, ax = plt.subplots(figsize: (8, 6));
ax.hist(total_rooms.ToArray<double>(), bins: 50);
ax.set_xlabel("Total Rooms");
ax.set_ylabel("Frequency");
ax.set_title("Histogram of Total Rooms");
plt.tight_layout();
plt.savefig("totalrooms_histogram.png");
```

Note that the code uses LINQ to extract the `TotalRooms` values from our data structure, then creates a ScottPlot histogram with 50 bins. ScottPlot provides a clean, matplotlib-like API for creating publication-quality visualizations in C#.

Now let's look at the histogram. It should look like this:

![Histogram Of TotalRooms](../img/totalrooms-histogram.png)
{ .img-fluid .mb-4 }

Think about the following:

-    Do you notice the long tail (outliers)?
-    How should you deal with this?
-    Could other columns in the dataset have the same issue?

Write down which data transformation steps you are going to apply to deal with the outliers in the total_rooms column.
{ .homework }

#### Create a Histogram of Every Feature

Now let's modify the code to generate histograms for all the columns in the dataset. Enter the following prompt:

"Modify the code so that it creates histograms for every column in the dataset."
{ .prompt }

You should see the following code in your project:
```csharp
// Create subplots for all numeric columns
var plt = new ScottPlot.Plot(1200, 900);
var subplotManager = plt.Add.Subplot(3, 3);

var columns = new (string Name, Func<HousingData, double> Selector)[]
{
    ("Longitude", x => x.Longitude),
    ("Latitude", x => x.Latitude), 
    ("Housing Median Age", x => x.HousingMedianAge),
    ("Total Rooms", x => x.TotalRooms),
    ("Total Bedrooms", x => x.TotalBedrooms),
    ("Population", x => x.Population),
    ("Households", x => x.Households),
    ("Median Income", x => x.MedianIncome),
    ("Median House Value", x => x.MedianHouseValue)
};

for (int i = 0; i < columns.Length; i++)
{
    var subplot = subplotManager.GetSubplot(i / 3, i % 3);
    var values = housingData.Select(columns[i].Selector).ToArray();
    subplot.Add.Histogram(values, bins: 30);
    subplot.Axes.Bottom.Label.Text = columns[i].Name;
    subplot.Axes.Left.Label.Text = "Frequency";
}

plt.SavePng("all_histograms.png");
```

This code uses ScottPlot's subplot functionality to create a 3x3 grid of histograms. We define an array of column selectors using C# tuples and LINQ expressions, then iterate through each column to create individual histograms within the subplot grid.

Your histograms should look like this:

![Histogram Of All Columns](../img/all-histograms.png)
{ .img-fluid .mb-4 }

You can see that the **total_rooms**, **total_bedrooms**, **population** and **household** columns have outliers. These are apartment blocks with a very large number of occupants and rooms, and we'll have to deal with them.

#### Summary

Visualization is one of the most important sanity checks in machine learning.
It helps you spot issues, guide preprocessing, and understand how features behave—even before you build a model.

By using ScottPlot and agents together, you're learning how to both automate and supervise the exploration process.

{{< /encrypt >}}