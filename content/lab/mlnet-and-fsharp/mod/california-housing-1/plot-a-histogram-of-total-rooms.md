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

ScottPlot is a very nice plotting and visualization library for F# and NET that can go toe-to-toe with Python libraries like matplotlib and seaborn. We will use it in these labs whenever we want to visualize a dataset.

First, let's install the ScottPlot NuGet package. In your terminal (inside the CaliforniaHousing folder), install ScottPlot like this:

```bash
dotnet add package ScottPlot
```

Then open Visual Studio Code in the current folder, like this:

```bash
code .
```

Open the Program.fs file and remove all existing content, because we don't want the agent to get confused. Replace the content with this:

```fsharp
open ScottPlot
open Microsoft.ML
open System
```

#### Create a Histogram of total_rooms

Now let's ask Copilot to write the code for us.

At the bottom of the Copilot panel in Visual Studio Code, make sure the AI mode is set to 'Agent'. Then select your favorite model. I like to use GPT 4.1 or Claude 3.7 for coding work.

![Enable Agent Mode](../img/agent-mode.jpg)
{ .img-fluid .mb-4 .border }

Now enter the following prompt:

"Write F# code using ScottPlot to generate a histogram of the total_rooms column from the CSV file."
{ .prompt }

And let Copilot write the code for you.

A thing you'll want to check is how the generated code loads the CSV file. The correct approach is to use the method `LoadFromTextFile`, which is part of the Microsoft.ML library.

You should see the following data loading code in your project:

```fsharp
// Set up a data loader
let mlContext = new MLContext()
let loader =
    mlContext.Data.CreateTextLoader(
        columns = [|
            TextLoader.Column("row_id", DataKind.Single, 0)
            TextLoader.Column("longitude", DataKind.Single, 1)
            TextLoader.Column("latitude", DataKind.Single, 2)
            TextLoader.Column("housing_median_age", DataKind.Single, 3)
            TextLoader.Column("total_rooms", DataKind.Single, 4)
            TextLoader.Column("total_bedrooms", DataKind.Single, 5)
            TextLoader.Column("population", DataKind.Single, 6)
            TextLoader.Column("households", DataKind.Single, 7)
            TextLoader.Column("median_income", DataKind.Single, 8)
            TextLoader.Column("median_house_value", DataKind.Single, 9)
        |],
        hasHeader = true,
        separatorChar = ','
    )

let dataView = loader.Load("California-Housing.csv")

// Extract total_rooms column
let totalRooms =
    mlContext.Data.CreateEnumerable<HousingData>(dataView, reuseRowObject=false)
    |> Seq.map (fun row -> float row.total_rooms)
    |> Seq.toArray
```

This code calls `CreateTextLoader` to set up a text data loader with the correct column names and indices, and then calls the `Load` function to load the CSV file into a data view. 

Finally, the code calls `CreateEnumerable` to convert the data view into an enumeration of `HousingData` instances. The `map` function extracts the **total_rooms** column and converts it to a `float`, and `toArray` converts the enumeration to a `float[]` array.

This implementation is by the book, and exactly what we want to see in auto-generated F# machine learning code that uses Microsoft.ML.
{ .tip }

This is what the `HousingData` type looks like:

```fsharp
// Define the data schema
[<CLIMutable>]
type HousingData = {
    row_id: float32
    longitude: float32
    latitude: float32
    housing_median_age: float32
    total_rooms: float32
    total_bedrooms: float32
    population: float32
    households: float32
    median_income: float32
    median_house_value: float32
}
```

Each column in the dataset is implemented as a field, with a matching name and the correct data type.

If instead you get generated code that uses `File.ReadAllLines` to manually load the CSV file, you may want to adjust your prompt and explicitly ask for code that uses `LoadFromTextFile` to load the data.

We want to keep our code elegant and lean. The Microsoft.ML library has built-in support for loading CSV files, so we don't want to import additional packages that clutter up our codebase.
{ .tip }

Here is the plotting code I ended up with:

```fsharp
// Create a new plot
let plot = new Plot()

// Create a histogram
let hist = ScottPlot.Statistics.Histogram.WithBinCount(10, totalRooms)

// Add the bars to the plot
let barPlot = plot.Add.Bars(hist.Bins, hist.Counts)

// Size each bar slightly less than the width of a bin
for bar in barPlot.Bars do
    bar.Size <- hist.FirstBinSize * 0.8

// Customize appearance
plot.Title("Histogram of Total Rooms")
plot.XLabel("Total Rooms")
plot.YLabel("Frequency")

// Save the plot
plot.SavePng("totalrooms-histogram.png", 600, 400) |> ignore
```

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

You should get something like this:

![Histogram Of All Columns](../img/all-histograms.png)
{ .img-fluid .mb-4 }

You can see that the **total_rooms**, **total_bedrooms**, **population** and **household** columns have outliers. These are apartment blocks with a very large number of occupants and rooms, and we'll have to deal with them.

#### Summary

Visualization is one of the most important sanity checks in machine learning.
It helps you spot issues, guide preprocessing, and understand how features behave—even before you build a model.

By using ScottPlot and agents together, you're learning how to both automate and supervise the exploration process.

{{< /encrypt >}}