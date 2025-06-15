---
title: "Plot A Histogram Of Total Rooms"
type: "lesson"
layout: "default"
sortkey: 40
---

Before you build a machine learning model, it’s important to understand your data visually. Just looking at the numbers, like you did in the previous lesson, may not be enough. A good chart can clearly reveal patterns in the dataset.

In this section, you’ll going to build a visualization to detect any outliers in the **total_rooms** feature.

Let's get started.

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

Open the Program.cs file and remove all existing content, because we don't want the agent to get confused. Replace the content with this:

```csharp
using ScottPlot;
```

#### Create a Histogram of total_rooms

Now let's ask our AI agent to write all the code for us.

At the bottom of the Copilot panel in Visual Studio Code, make sure the AI mode is set to Agent. Then select your favorite model (I like Anthropic Claude 3.7 for coding work).

![Enable Agent Mode](../img/agent-mode.jpg)
{ .img-fluid .mb-4 .border }

Now enter the following prompt:

"Write C# code using ScottPlot to generate a histogram of the total_rooms column from the CSV file."
{ .prompt }

And let the agent write all the code for you.

You may get an issue where the agent struggles with the ScottPlot 5 syntax and tries to generate code for earlier versions. You'll get compile errors on the lines that set up the histogram and try to plot it.

This can happen, because agents are trained on data up until a specific cutoff point, and libraries may have changed their APIs after this date. In my testing, I noticed that at the time of this writing (June 2025), Anthropic Claude 3.7 was unaware of the new syntax and would get stuck in a loop trying to fix my code. I had to abort the agent and fix the code manually.

For reference, [this is how to create a histogram In ScottPlot 5](https://www.scottplot.net/cookbook/5.0/Histograms/).

Here is the plotting code I ended up with:

```csharp
// Convert float array to double array (required by ScottPlot 5)
double[] doubleData = totalRoomsColumn.Select(x => (double)x).ToArray();

// Create a new plot
var plot = new Plot();

// Create a histogram
var hist = ScottPlot.Statistics.Histogram.WithBinCount(10, doubleData);

// Add the bars to the plot
var barPlot = plot.Add.Bars(hist.Bins, hist.Counts);

// Size each bar slightly less than the width of a bin
foreach (var bar in barPlot.Bars)
    bar.Size = hist.FirstBinSize * .8;

// Customize appearance
plot.Title(title);
plot.XLabel("Total Rooms");
plot.YLabel("Frequency");

// Save the plot
plot.SavePng(outputPath, 600, 400);
```

Note the first line, it converts the `totalRoomsColumn` (a `float[]` with all the **total_rooms** values) to `double[]`, because ScottPlot histograms can only work with double values.

Another thing you'll want to check is how the code loads the CSV file. The correct approach is to use the method `LoadFromTextFile`, which is part of the Microsoft.ML library.

You should see the following data loading code in your project:

```csharp
// Get the path to the CSV file
string projectDirectory = Directory.GetCurrentDirectory();
string dataPath = Path.Combine(projectDirectory, "California-Housing.csv");

// Create ML context
var mlContext = new MLContext();

// Load data from CSV
Console.WriteLine("Loading data from CSV file...");
var dataView = mlContext.Data.LoadFromTextFile<HousingData>(
    path: dataPath,
    hasHeader: true,
    separatorChar: ',');

// Extract the total_rooms column into an array
Console.WriteLine("Extracting total_rooms data...");
var totalRoomsColumn = mlContext.Data.CreateEnumerable<HousingData>(dataView, reuseRowObject: false)
    .Select(row => row.TotalRooms)
    .ToArray();
```

This code uses `LoadFromTextFile` to load the CSV file into a data view, which can be used for later machine learning training and evaluation. The code uses a helper class `HousingData` which represents a single row in the dataset.

The code then uses `CreateEnumerable` to convert the loaded data into an enumeration of `HousingData` instances, and a LINQ expression to convert that to a `float[]` containing only the TotalRooms values.

This implementation is by the book, and exactly what we want to see in auto-generated machine learning code that uses Microsoft.ML.
{ .tip }

This is what the HousingData class looks like:

```csharp
// Data model for California housing dataset
public class HousingData
{
    [LoadColumn(0)] public float CsvRowId { get; set; }
    [LoadColumn(1)] public float Longitude { get; set; }
    [LoadColumn(2)] public float Latitude { get; set; }
    [LoadColumn(3)] public float HousingMedianAge { get; set; }
    [LoadColumn(4)] public float TotalRooms { get; set; }
    [LoadColumn(5)] public float TotalBedrooms { get; set; }
    [LoadColumn(6)] public float Population { get; set; }
    [LoadColumn(7)] public float Households { get; set; }
    [LoadColumn(8)] public float MedianIncome { get; set; }
    [LoadColumn(9)] public float MedianHouseValue { get; set; }
}
```

You can see each column in the dataset implemented as a property, with the correct data type, and annotated with a `LoadColumn` attribute that specifies from which CSV column the data should be loaded.

If instead you get generated code that uses `File.ReadAllLines` or `Microsoft.VisualBasic.FileIO.TextFieldParser` to manually load the CSV file, you may want to adjust your prompt and explicitly ask for code that uses `LoadFromTextFile` to load the data.

We want to keep our code elegant and lean. The Microsoft.ML library has built-in support for loading CSV files, so we don't want to import additional packages that clutter up our codebase.
{ .tip }

Now let's look at the histogram. It should look like this:

![Histogram Of TotalRooms](../img/totalrooms-histogram.png)
{ .img-fluid .mb-4 }

Think about the following:

-    Do you notice the long tail (outliers)?
-    How should you deal with this?
-    Could other columns in the dataset have the same issue?

Write down which data transformation steps you are going to apply to deal with the outliers in the total_rooms column.
{ .homework }

#### Bonus: Create a Histogram of Every Feature

Now let's modify the code to generate histograms for all the columns in the dataset. Enter the following prompt:

"Modify the code so that it creates one large plot with histograms for every column in the dataset."
{ .prompt }

When I ran this prompt, Claude 3.7 refactored my code to save each histogram in a separate PNG file, and then it added a HTML page to my project that shows all graphs in a nice grid.

But ScottPlot 5 actually has the ability to combine multiple charts into a single plot. However, ,the agent was not able to get the code working correctly, and in the end decided to use this workaround instead.

When reviewing the output of an agent, always make sure you agree with the chosen workarounds. We now have an app that requires a web browser to see some of the charts. Do you agree with that? If not, ask the agent to give you something better.
{ .tip }

Now let's review the output. You should get something like this:

![Histogram Of All Columns](../img/all-histograms.png)
{ .img-fluid .mb-4 }

You can see that the **total_rooms**, **total_bedrooms**, **population** and **household** columns have outliers. These are apartment blocks with a very large number of occupants and rooms, and we'll have to deal with them.

#### Summary

Visualization is one of the most important sanity checks in machine learning.
It helps you spot issues, guide preprocessing, and understand how features behave—even before you build a model.

By using ScottPlot and agents together, you’re learning how to both automate and supervise the exploration process.
