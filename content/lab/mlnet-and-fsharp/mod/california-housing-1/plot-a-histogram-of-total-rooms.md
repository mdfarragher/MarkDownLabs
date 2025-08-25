---
title: "Plot A Histogram Of Total Rooms"
type: "lesson"
layout: "default"
sortkey: 40
---

# Plot A Histogram Of Total Rooms

Before you build a machine learning model, it's important to understand your data visually. Just looking at the numbers, like you did in the previous lesson, may not be enough. A good chart can clearly reveal patterns in the dataset.

In this section, you'll going to build a visualization to detect any outliers in the **total_rooms** feature.

Let's get started.

{{< encrypt >}}

#### Install ScottPlot

ScottPlot is a very nice plotting and visualization library for C# and NET that can go toe-to-toe with Python libraries like matplotlib and seaborn. We will use it in these labs whenever we want to visualize a dataset.

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
// Get the path to the CSV file
let projectDirectory = Directory.GetCurrentDirectory()
let dataPath = Path.Combine(projectDirectory, "California-Housing.csv")

// Create ML context
let mlContext = MLContext()

// Load data from CSV
printfn "Loading data from CSV file..."
let dataView = mlContext.Data.LoadFromTextFile<HousingData>(
    path = dataPath,
    hasHeader = true,
    separatorChar = ',')

// Extract the total_rooms column into an array
printfn "Extracting total_rooms data..."
let totalRoomsColumn = 
    mlContext.Data.CreateEnumerable<HousingData>(dataView, reuseRowObject = false)
    |> Seq.map (fun row -> row.TotalRooms)
    |> Array.ofSeq
```

This code uses `LoadFromTextFile` to load the CSV file into a data view, which can be used for later machine learning training and evaluation. The code uses a helper class `HousingData` which represents a single row in the dataset.

The code then uses `CreateEnumerable` to convert the loaded data into an enumeration of `HousingData` instances, and F# pipeline operators to convert that to a `float32[]` containing only the TotalRooms values.

This implementation is by the book, and exactly what we want to see in auto-generated machine learning code that uses Microsoft.ML.
{ .tip }

This is what the HousingData record type looks like:

```fsharp
// Data model for California housing dataset
open Microsoft.ML.Data

[<CLIMutable>]
type HousingData = {
    [<LoadColumn(0)>] CsvRowId: float32
    [<LoadColumn(1)>] Longitude: float32
    [<LoadColumn(2)>] Latitude: float32
    [<LoadColumn(3)>] HousingMedianAge: float32
    [<LoadColumn(4)>] TotalRooms: float32
    [<LoadColumn(5)>] TotalBedrooms: float32
    [<LoadColumn(6)>] Population: float32
    [<LoadColumn(7)>] Households: float32
    [<LoadColumn(8)>] MedianIncome: float32
    [<LoadColumn(9)>] MedianHouseValue: float32
}
```

Each column in the dataset is implemented as a property, with the correct data type, and annotated with a `LoadColumn` attribute that specifies the corresponding CSV column index, starting from zero.

If instead you get generated code that uses `File.ReadAllLines` or `Microsoft.VisualBasic.FileIO.TextFieldParser` to manually load the CSV file, you may want to adjust your prompt and explicitly ask for code that uses `LoadFromTextFile` to load the data.

We want to keep our code elegant and lean. The Microsoft.ML library has built-in support for loading CSV files, so we don't want to import additional packages that clutter up our codebase.
{ .tip }

You may get an issue where the agent struggles with the ScottPlot 5 syntax and tries to generate code for earlier versions. That code will not compile and you'll get errors for the source lines that set up and plot the histogram.

This can happen, because AI agents are trained on data up until a specific cutoff point, and libraries may have changed their APIs after this date. In my testing, I noticed that at the time of this writing (June 2025), Claude 3.7 was unaware of the new syntax and would get stuck in a loop trying to fix my code. I had to abort the agent and fix the code manually.

For reference, [this is how to create a histogram In ScottPlot 5](https://www.scottplot.net/cookbook/5.0/Histograms/).

Here is the plotting code I ended up with:

```fsharp
// Convert float array to double array (required by ScottPlot 5)
let doubleData = totalRoomsColumn |> Array.map (fun x -> double x)

// Create a new plot
let plot = Plot()

// Create a histogram
let hist = ScottPlot.Statistics.Histogram.WithBinCount(10, doubleData)

// Add the bars to the plot
let barPlot = plot.Add.Bars(hist.Bins, hist.Counts)

// Size each bar slightly less than the width of a bin
for bar in barPlot.Bars do
    bar.Size <- hist.FirstBinSize * 0.8

// Customize appearance
plot.Title("Total Rooms Histogram") |> ignore
plot.XLabel("Total Rooms") |> ignore
plot.YLabel("Frequency") |> ignore

// Save the plot
plot.SavePng("histogram.png", 600, 400)
```

Note the first line, it converts the `totalRoomsColumn` (a `float[]` with all the **total_rooms** values) to `double[]`, because ScottPlot histograms work with double values.

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