---
title: "Plot A Histogram Of Every Feature"
type: "lesson"
layout: "default"
sortkey: 40
---

# Plot A Histogram Of Every Feature

Before you build a machine learning model, it's important to understand your data visually. Just looking at the numbers, like you did in the previous lesson, may not be enough. A good chart can clearly reveal patterns in the dataset.

In this section, you'll going to generate code to plot a histogram for every (interesting) feature in the dataset.

Let's get started.

{{< encrypt >}}

#### Install ScottPlot

First, let's install the ScottPlot NuGet package. In your terminal (inside the TaxiFarePrediction folder), install ScottPlot like this:

```bash
dotnet add package ScottPlot
```

Then open Visual Studio Code in the current folder, like this:

```bash
code .
```

Open the Program.fs file and remove all existing content, replace the content with this:

```fsharp
open ScottPlot
open Microsoft.ML
open System
```

#### Create a Histogram of passenger_count

Now let's ask Copilot to write the code for us.

At the bottom of the Copilot panel in Visual Studio Code, make sure the AI mode is set to 'Agent'. Then select your favorite model (I used GPT 4o while preparing this lab).

Enter the following prompt:

"Write F# code using ScottPlot to generate a histogram of the passenger_count column from the Taxi-Trips.csv file. Make sure you load the file using a Microsoft.ML pipeline."
{ .prompt }

And let Copilot write the code for you.

You should see the following data loading code in your project:

```fsharp
// Create MLContext
let mlContext = MLContext()

// Load data
let data = 
    mlContext.Data.LoadFromTextFile<TaxiTrip>(
        path = "Taxi-Trips.csv",
        hasHeader = true,
        separatorChar = ',')

// Extract passenger_count column
let passengerCounts = 
    mlContext.Data.CreateEnumerable<TaxiTrip>(data, reuseRowObject = false)
    |> Seq.map (fun trip -> trip.PassengerCount)
    |> Array.ofSeq
```

This code uses `LoadFromTextFile` to load the CSV file into a data view, which can be used for later machine learning training and evaluation. The code uses a helper type `TaxiTrip` which represents a single row in the dataset.

The code then uses `CreateEnumerable` to convert the loaded data into a sequence of `TaxiTrip` instances, and F# pipeline operators to convert that to a `float[]` containing only the PassengerCount values.

This implementation is by the book, and exactly what we want to see in auto-generated machine learning code that uses Microsoft.ML with F#.
{ .tip }

This is what the TaxiTrip class looks like:

```fsharp
open Microsoft.ML.Data

[<CLIMutable>]
type TaxiTrip = {
    [<LoadColumn(4)>]
    PassengerCount: float32
}
```

The passenger_count column in the dataset is implemented as a record field, with the correct data type, and annotated with a `LoadColumn` attribute that specifies the corresponding CSV column index, in this case index number 4.

And this is the plotting code you will probably end up with:

```fsharp
// Create histogram data
let histogram = ScottPlot.Statistics.Histogram.WithBinCount(20, 
    passengerCounts |> Array.map float)

// Generate histogram using ScottPlot
let plt = Plot()
plt.Add.Bars(histogram.Bins, histogram.Counts) |> ignore
plt.Title("Passenger Count Histogram")
plt.XLabel("Passenger Count")
plt.YLabel("Frequency")

// Save the plot
plt.SavePng("PassengerCountHistogram.png", 600, 400)
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

"Modify the code so that it loads every column in the dataset and populates each column as a field in the TaxiTrip record type."
{ .prompt }

The agent will extend the `TaxiTrip` class with new properties for each dataset column:

```fsharp
open System

[<CLIMutable>]
type TaxiTrip = {
    [<LoadColumn(0)>] RowID: int
    [<LoadColumn(1)>] VendorID: int
    [<LoadColumn(2)>] PickupDateTime: DateTime
    [<LoadColumn(3)>] DropoffDateTime: DateTime
    [<LoadColumn(4)>] PassengerCount: int
    [<LoadColumn(5)>] TripDistance: float32
    [<LoadColumn(6)>] RatecodeID: int
    [<LoadColumn(7)>] StoreAndFwdFlag: string
    [<LoadColumn(8)>] PULocationID: int
    [<LoadColumn(9)>] DOLocationID: int
    [<LoadColumn(10)>] PaymentType: int
    [<LoadColumn(11)>] FareAmount: float32
    [<LoadColumn(12)>] Extra: float32
    [<LoadColumn(13)>] MtaTax: float32
    [<LoadColumn(14)>] TipAmount: float32
    [<LoadColumn(15)>] TollsAmount: float32
    [<LoadColumn(16)>] ImprovementSurcharge: float32
    [<LoadColumn(17)>] TotalAmount: float32
}
```

Now we're going to use the same generic and reusable code scaffold we used for the Pearson correlation matrix in the previous lab module. We will use reflection to get the list of property names, and then use a generic helper method to create a histogram for that property.

Add the following code right after the data loading code, but before the plotting code:

```fsharp
open System.Reflection

// get column names, skip row id and non-numeric columns
let columnNames = 
    typeof<TaxiTrip>.GetProperties()
    |> Array.filter (fun p -> p.Name <> "RowID" && 
                             (p.PropertyType = typeof<float32> || p.PropertyType = typeof<int>))
    |> Array.map (fun p -> p.Name)

// set up a 4x4 grid of plots
let grid = Multiplot()
grid.RemovePlot(grid.GetPlot(0)) // remove default plot
grid.Layout <- MultiplotLayouts.Grid(columns = 4, rows = 4)

// generate histograms
for columnName in columnNames do
    let plot = PlotHistogram<TaxiTrip>(taxiTrips, columnName)
    grid.AddPlot(plot) |> ignore

// save the grid
grid.SavePng("histograms.png", 1900, 1280)
```

This won't compile, because we don't have the `taxiTrips` variable yet, and the `PlotHistogram` method doesn't exist yet. So let's fix those issues one by one. Enter the following prompt:

"Change the code that calls CreateEnumerable so that it produces a list of TaxiTrip instances with each field populated. Put the list in a variable called taxiTrips."
{ .prompt }

That should fix the data loading code and give us a `taxiTrips` variable which is a populated `List<TaxiTrip>`.

Now let's add the `PlotHistogram` method by hand. We already have working histogram code, we just need to make it reusable by adding a little bit of reflection. You'll want something like this:

```fsharp
let PlotHistogram<'T> (data: 'T list) (columnName: string) =
    // get all column values as doubles
    let prop = typeof<'T>.GetProperty(columnName)
    let values = data |> List.map (fun t -> Convert.ToDouble(prop.GetValue(t))) |> Array.ofList

    // Create histogram data
    let histogram = ScottPlot.Statistics.Histogram.WithBinCount(50, values)

    // Generate histogram using ScottPlot
    let plt = Plot()
    plt.Add.Bars(histogram.Bins, histogram.Counts) |> ignore
    plt.XLabel(columnName)
    plt.YLabel("Frequency")

    plt
```

This function uses reflection to create a `double[]` array of the dataset column specified by `columnName`, and then creates a histogram from this data and returns it as a new plot. The code in the main program assembles these plots into a nice 4x4 grid. 

When you run the app, you should get something like this:

![Histogram Of All Columns](../img/all-histograms.png)
{ .img-fluid .mb-4 }

We have already covered the PassengerCount histogram with the weird zero-passenger trips, but check out the other plots. There are lots of outliers in this dataset:

- **TripDistance** has outliers for trips > 15 miles
- **FareAmount** has outliers for fares > 100 dollars
- There are trips where **FareAmount** and **TotalAmount** are negative
- **TipAmount** has outliers for tips > 15 dollars

At the very least, we'll have to remove (or clip) all trips with zero passengers or a negative fare amount. 

Write down the data transformation steps you are going to use to process the dataset, based on what you've just learned from these histograms.
{ .homework }

#### Create a Utility Class

We will reuse this histogram code in later lessons, so this is a good moment to preserve your work.

In Visual Studio Code, select the code that sets up the 4x4 grid of plots, calls the `PlotHistogram` method and saves the grid. Then press CTRL+I to launch the in-line AI prompt window, and type the following prompt:

"Move this code to a utility function called PlotAllHistograms, with arguments for the list of taxi trips, the list of column names, and the number of rows and columns in the plot grid"
{ .prompt }

And now that we have these two methods, we can move them both to a utility class. Select all source code lines for `PlotAllHistograms` and `PlotHistogram`, press CTRL+I again and enter the following inline prompt:

"Move all of this code to a separate utility module called HistogramUtils."
{ .prompt }

This will produce a new module file called `HistogramUtils`, with all of the code for creating and plotting the grid of histograms for any given feature. We can now use this module in other projects.

And if you want to clean up your code and make it as side-effect-free as possible, you can edit `PlotAllHistograms` and have it return the `Multiplot` grid instance. You can then save the grid in the main program class instead. Your main calling code will then look like this:

```fsharp
// calculate the histogram grid
let grid = HistogramUtils.PlotAllHistograms<TaxiTrip>(taxiTrips, columnNames, columns = 5, rows = 4)

// save the grid
grid.SavePng("histograms.png", 1900, 1280)
```

Perfect!

If you get stuck or want to save some time, feel free to download my completed HistogramUtils class from Codeberg and use it in your own project:

https://codeberg.org/mdft/ml-mlnet-csharp/src/branch/main/TaxiFarePrediction/HistogramUtils.cs

{{< /encrypt >}}