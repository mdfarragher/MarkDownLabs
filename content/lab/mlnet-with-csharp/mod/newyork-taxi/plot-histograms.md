---
title: "Plot A Histogram Of Every Feature"
type: "lesson"
layout: "default"
sortkey: 40
---

Before you build a machine learning model, it’s important to understand your data visually. Just looking at the numbers, like you did in the previous lesson, may not be enough. A good chart can clearly reveal patterns in the dataset.

In this section, you’ll going to generate code to plot a histogram for every (interesting) feature in the dataset.

Let's get started.

#### Install ScottPlot

First, let's install the ScottPlot NuGet package. In your terminal (inside the TaxiFarePrediction folder), install ScottPlot like this:

```bash
dotnet add package ScottPlot
```

Then open Visual Studio Code in the current folder, like this:

```bash
code .
```

Open the Program.cs file and remove all existing content, replace the content with this:

```csharp
using ScottPlot;
```

#### Create a Histogram of passenger_count

Now let's ask Copilot to write the code for us.

At the bottom of the Copilot panel in Visual Studio Code, make sure the AI mode is set to 'Agent'. Then select your favorite model (I used GPT 4o while preparing this lab).

Enter the following prompt:

"Write C# code using ScottPlot to generate a histogram of the passenger_count column from the Taxi-Trips.csv file. Make sure you load the file using a Microsoft.ML pipeline."
{ .prompt }

And let Copilot write the code for you.

You should see the following data loading code in your project:

```csharp
// Create MLContext
var mlContext = new MLContext();

// Load data
var data = mlContext.Data.LoadFromTextFile<TaxiTrip>(
    path: "Taxi-Trips.csv",
    hasHeader: true,
    separatorChar: ',');

// Extract passenger_count column
var passengerCounts = mlContext.Data.CreateEnumerable<TaxiTrip>(data, reuseRowObject: false)
    .Select(trip => trip.PassengerCount)
    .ToArray();
```

This code uses `LoadFromTextFile` to load the CSV file into a data view, which can be used for later machine learning training and evaluation. The code uses a helper class `TaxiTrip` which represents a single row in the dataset.

The code then uses `CreateEnumerable` to convert the loaded data into an enumeration of `TaxiTrip` instances, and a LINQ expression to convert that to a `float[]` containing only the PassengerCount values.

This implementation is by the book, and exactly what we want to see in auto-generated machine learning code that uses Microsoft.ML.
{ .tip }

This is what the TaxiTrip class looks like:

```csharp
class TaxiTrip
{
    [LoadColumn(4)]
    public float PassengerCount { get; set; }
}
```

The passenger_count column in the dataset is implemented as a property, with the correct data type, and annotated with a `LoadColumn` attribute that specifies the corresponding CSV column index, in this case index number 4.

And this is the plotting code you will probably end up with:

```csharp
// Create histogram data
var histogram = ScottPlot.Statistics.Histogram.WithBinCount(20, 
    passengerCounts.Select(x => (double)x).ToArray());

// Generate histogram using ScottPlot
var plt = new ScottPlot.Plot();
plt.Add.Bars(histogram.Bins, histogram.Counts);
plt.Title("Passenger Count Histogram");
plt.XLabel("Passenger Count");
plt.YLabel("Frequency");

// Save the plot
plt.SavePng("PassengerCountHistogram.png", 600, 400);
```

Let's look at the histogram:

![Histogram Of PassengerCount](../img/passengercount-histogram.png)
{ .img-fluid .mb-4 }

This looks plausible. Most taxi trips have only a single passenger, and the second most popular trip is with two passengers. And there are no trips with more than six passengers.

But look at the bar at zero, apparently some trips have zero passengers? You'll need to decide how you're going to deal with these trips. You could clip the trips by setting the count to 1, or you could trim the dataset by removing these trips from the training data.

Write down the data transformation steps you are going to use to process the PassengerCount column.
{ .homework }

#### Create a Histogram of Every Feature

Now let's modify the code to generate histograms for all the columns in the dataset. First, we're going to make sure that we load every column in the dataset. Enter the following prompt:

"Modify the code so that it loads every column in the dataset and populates each column as a property in the TaxiTrip class."
{ .prompt }

The agent will extend the `TaxiTrip` class with new properties for each dataset column:

```csharp
class TaxiTrip
{
    [LoadColumn(0)] public int RowID { get; set; }
    [LoadColumn(1)] public int VendorID { get; set; }
    [LoadColumn(2)] public DateTime PickupDateTime { get; set; }
    [LoadColumn(3)] public DateTime DropoffDateTime { get; set; }
    [LoadColumn(4)] public int PassengerCount { get; set; }
    [LoadColumn(5)] public float TripDistance { get; set; }
    [LoadColumn(6)] public int RatecodeID { get; set; }
    [LoadColumn(7)] public string? StoreAndFwdFlag { get; set; }
    [LoadColumn(8)] public int PULocationID { get; set; }
    [LoadColumn(9)] public int DOLocationID { get; set; }
    [LoadColumn(10)] public int PaymentType { get; set; }
    [LoadColumn(11)] public float FareAmount { get; set; }
    [LoadColumn(12)] public float Extra { get; set; }
    [LoadColumn(13)] public float MtaTax { get; set; }
    [LoadColumn(14)] public float TipAmount { get; set; }
    [LoadColumn(15)] public float TollsAmount { get; set; }
    [LoadColumn(16)] public float ImprovementSurcharge { get; set; }
    [LoadColumn(17)] public float TotalAmount { get; set; }
}
```

Now we're going to use the same generic and reusable code scaffold we used for the Pearson correlation matrix in the previous lab module. We will use reflection to get the list of property names, and then use a generic helper method to create a histogram for that property.

Add the following code right after the data loading code, but before the plotting code:

```csharp
// get column names, skip row id and non-numeric columns
var columnNames = (from p in typeof(TaxiTrip).GetProperties()
                   where p.Name != "RowID"
                         && (p.PropertyType == typeof(float)
                         || p.PropertyType == typeof(int))
                   select p.Name).ToArray();

// set up a 5x4 grid of plots
var grid = new ScottPlot.Multiplot();
grid.Layout = new ScottPlot.MultiplotLayouts.Grid(columns: 5, rows: 4);

// generate histograms
foreach (var columnName in columnNames)
{
    var plot = PlotHistogram<TaxiTrip>(taxiTrips, columnName);
    grid.AddPlot(plot);
}

// save the grid
grid.SavePng("histograms.png", 1900, 1280);
```

This won't compile, because we don't have the `taxiTrips` variable yet, and the `PlotHistogram` method doesn't exist yet. So let's fix those issues one by one. Enter the following prompt:

"Change the code that calls CreateEnumerable so that it produces a list of TaxiTrip instances with each property populated. Put the list in a variable called taxiTrips."
{ .prompt }

That should fix the data loading code and give us a `taxiTrips` variable which is a populated `List<TaxiTrip>`.

Now let's add the `PlotHistogram` method by hand. We already have working histogram code, we just need to make it reusable by adding a little bit of reflection. You'll want something like this:

```csharp
static Plot PlotHistogram<T>(List<T> data, string columnName)
{
    // get all column values as doubles
    var prop = typeof(T).GetProperty(columnName);
    var values = data.Select(t => Convert.ToDouble(prop?.GetValue(t))).ToArray();

    // Create histogram data
    var histogram = ScottPlot.Statistics.Histogram.WithBinCount(50, values);

    // Generate histogram using ScottPlot
    var plt = new ScottPlot.Plot();
    plt.Add.Bars(histogram.Bins, histogram.Counts);
    plt.Title($"{columnName} Histogram");
    plt.XLabel(columnName);
    plt.YLabel("Frequency");

    return plt;
}
```

This method uses reflection to create a `double[]` array of the dataset column specified by `columnName`, and then creates a histogram from this data and returns it as a new plot. The code in the main program class assembles these plots into a nice 5x4 grid. 

When you run the app, you should get something like this:

![Histogram Of All Columns](../img/all-histograms.png)
{ .img-fluid .mb-4 }

We have already covered the PassengerCount histogram with the weird zero-passenger trips, but check out the other plots. There are lots of outliers in this dataset:

- **TripDistance** has outliers for trips > 10 miles
- **FareAmount** has outliers for fares > 50 dollars
- There are trips where **FareAmount** and **TotalAmount** are negative
- **TipAmount** has outliers for tips > 10 dollars

At the very least, we'll have to remove (or clip) all trips with zero passengers or a negative fare amount. 

Write down the data transformation steps you are going to use to process the dataset, based on what you've just learned from these histograms.
{ .homework }

#### Create a Utility Class

We will reuse this histogram code in later lessons, so this is a good moment to preserve your work.

In Visual Studio Code, select the code that sets up the 5x4 grid of plots, calls the `PlotHistogram` method and saves the grid. Then press CTRL+I to launch the in-line AI prompt window, and type the following prompt:

"Move this code to a utility method called PlotAllHistograms, with arguments for the list of taxi trips, the list of column names, and the number of rows and columns in the plot grid"
{ .prompt }

And now that we have these two methods, we can move them both to a utility class. Select all source code lines for `PlotAllHistograms` and `PlotHistogram`, press CTRL+I again and enter the following inline prompt:

"Move all of this code to a separate utility class called HistogramUtils."
{ .prompt }

This will produce a new class file called **HistogramUtils.cs**, with all of the code for creating and plotting the grid of histograms for any given feature. We can now use this method in other projects.

And if you want to clean up your code and make it as side-effect-free as possible, you can edit `PlotAllHistograms` and have it return the `Multiplot` grid instance. You can then save the grid in the main program class instead. Your main calling code will then look like this:

```csharp
// calculate the histogram grid
var grid = HistogramUtils.PlotAllHistograms<TaxiTrip>(taxiTrips, columnNames, columns: 5, rows: 4);

// save the grid
grid.SavePng("histograms.png", 1900, 1280);
```

Perfect!

If you get stuck or want to save some time, feel free to download my completed HistogramUtils class from Codeberg and use it in your own project:

https://codeberg.org/mdft/ml-mlnet-csharp/src/branch/main/TaxiFarePrediction

