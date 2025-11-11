---
title: "Plot The Scatterplot Matrix"
type: "lesson"
layout: "default"
sortkey: 70
---

In data science, there is a visualisation type called the scatterplot matrix which shows a the relationship of every dataset column to every other column. The visualisation can be used to quickly determine if there is a linear or semi-linear relationship between a feature and the label. 

In the previous lesson, we discovered that **FareAmount** is the best candidate for the label and that **TripDuration**, **TripDistance** and **RateCodeID** are strongly correlated with the label.

In this lesson, we're going to generate a scatterplot matrix that shows the relationhip between each of these four dataset columns. 

{{< encrypt >}}

#### Create a Scatterplot Matrix

We'll start by setting up our top-level code just like with the Pearson correlation matrix. Add the following code to the main program:

```fsharp
// column names to plot
let scatterPlotColumns = [| "TripDuration"; "TripDistance"; "RatecodeID"; "FareAmount" |]

// plot scatterplot matrix
let smplot = PlotScatterplotMatrix<TaxiTripWithDuration> taxiTripsWithDuration scatterPlotColumns

// Save the plot to a file
smplot.SavePng("scatterplot-matrix.png", 1900, 1200) |> ignore
```

The `PlotScatterplotMatrix` method is going to assemble a `Multiplot` using individual scatter plots, just like we did with the histogram grid. So we can simply copy and paste code from the `HistogramUtils` class and tweak it a little, like this:

```fsharp
let PlotScatterplotMatrix<'T> (data: 'T list) (columnNames: string[]) =
    let grid = Multiplot()
    grid.RemovePlot(grid.GetPlot(0)) // remove default plot
    let size = columnNames.Length
    grid.Layout <- MultiplotLayouts.Grid(columns = size, rows = size)

    for rowName in columnNames do
        for colName in columnNames do
            let plot = PlotScatter<'T> data colName rowName
            grid.AddPlot(plot) |> ignore

    grid
```

This will loop through the column names and create a grid of scatterplots with each plot showing a unique pair of columns. 

Now we just need to implement the `PlotScatter` method. But we have this method already, because we used it in the California Housing lab to plot a graph of **MedianIncome** versus **MedianHouseValue**. 

If you don't have that code anymore, prompt your AI agent to recreate the code or use my example for reference:

```fsharp
let PlotScatter<'T> (data: 'T list) (colName: string) (rowName: string) =
    // get column values
    let colProp = typeof<'T>.GetProperty(colName)
    let colValues = data |> List.map (fun h -> Convert.ToDouble(colProp.GetValue(h))) |> Array.ofList

    // get row values
    let rowProp = typeof<'T>.GetProperty(rowName)
    let rowValues = data |> List.map (fun h -> Convert.ToDouble(rowProp.GetValue(h))) |> Array.ofList

    // generate scatterplot
    let plt = new Plot()
    plt.Add.ScatterPoints(colValues, rowValues) |> ignore
    plt.XLabel(colName)
    plt.YLabel(rowName)

    plt
```

And that should get you this scatterplot matrix:

![Scatterplot Matrix](../img/scatterplot-matrix.png)
{ .img-fluid .mb-4 }

If you look at the two plots from the left in the bottom row, you can see that there is indeed a linear relationship between **FareAmount**, **TripDistance** and **TripDuration**. But the outlier, a trip with a fare of $350, distorts the graph and makes the relationship difficult to see. 

So let's get rid of the outlier. And we can use the same code to eliminate all trips with negative fares as well.

#### Filter The FareAmount

Filtering data in ML.NET is very easy, all you need is to call the `FilterRowsByColumn` method to filter a dataview by a given column. In fact, you've done that already in your code when you filtered the trip duration to anything up to 60 minutes. 

Locate the line in your code that filters by **TripDuration**:

```fsharp
// Filter out taxi trips longer than 60 minutes
let filteredData = mlContext.Data.FilterRowsByColumn(transformedData, "TripDuration", upperBound = 60.0)
```

And simply add this:

```fsharp
// Keep fares between $0 and $100
let filteredData2 = mlContext.Data.FilterRowsByColumn(filteredData, "FareAmount", lowerBound = 0.0, upperBound = 100.0)
```

And then make sure the next line in your app loads `taxiTripsWithDuration` using the `filteredData2` dataview, like this:

```fsharp
// Extract all TaxiTrip instances with properties populated
let taxiTripsWithDuration = 
    mlContext.Data.CreateEnumerable<TaxiTripWithDuration>(filteredData2, reuseRowObject = false)
    |> Seq.toList
```

And with that, your scatterplot matrix should now look like this:

![Scatterplot Matrix](../img/scatterplot-matrix-2.png)
{ .img-fluid .mb-4 }

There are three linear relationships clearly visible in the matrix:

- **FareAmount** by **TripDuration**
- **FareAmount** by **TripDistance**
- **TripDuration** by **TripDistance**

Also interesting is the statistical artefact at a fare of $50, visible in the graphs as a horizontal or vertical line. It means taxi drivers often charge $50 for a trip, even though by distance or duration the fare should actually be different (usually lower). 

#### Create a Utility Class

We will reuse this scatterplot code in later lessons, so this is a good moment to preserve your work.

In Visual Studio Code, select the code that declares the `PlotScatterplotMatrix` and `PlotScatter` methods. Then press CTRL+I to launch the in-line AI prompt window, and type the following prompt:

"Move all of this code to a separate utility module called ScatterUtils."
{ .prompt }

This will produce a new module file called `ScatterUtils`, with all of the code for creating and plotting the scatterplot matrix. We can now use this module in other projects.

If you get stuck or want to save some time, feel free to download my completed ScatterUtils class from Codeberg and use it in your own project: https://codeberg.org/mdft/ml-mlnet-fsharp/src/branch/main/TaxiFarePrediction/ScatterUtils.fs


#### Summary

We have used the Pearson correlation matrix to identify the features most strongly correlated with the label, and we generated a scatterplot matrix to verify that the relationships between these features and the label are indeed linear (with some noise added).

We're now ready to implement the data transformations and build the machine learning pipeline. 

{{< /encrypt >}}