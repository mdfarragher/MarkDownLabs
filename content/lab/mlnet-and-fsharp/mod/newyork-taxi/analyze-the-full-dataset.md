---
title: "Analyze The Full Dataset"
type: "lesson"
layout: "default"
sortkey: 130
---

# Analyze The Full Dataset

So, to recap: we designed the data transformation pipeline specifically for a dataset of 10,000 trips, covering the early morning of December 1st, 2018. We made many assumptions based off the histograms, correlation matrix and scatterplot grid and baked those assumptions into the design of the machine learning pipeline. 

Then we loaded the parquet dataset that covers the full month of December. The regression metrics of the full dataset are considerably worse than the ones for the limited set. This might be because the model was overfitting on the small dataset, or it could be because we now need to alter the data transformations to better match the new patters in the data. 

We won't know for sure until we recalculate the histograms, the correlation matrix and the scatterplots. 

{{< encrypt >}}

#### Generate The Histogram Matrix

Generating a new histogram matrix is very easy, because the code is still somewhere in your app. The code should look like this:

```fsharp
// Get a list of taxi trips with trip duration
let taxiTrips = 
    mlContext.Data.CreateEnumerable<TaxiTripWithDuration>(filteredData, reuseRowObject = false)
    |> List.ofSeq

// get column names, skip row id and non-numeric columns
let columnNames = 
    typeof<TaxiTripWithDuration>.GetProperties()
    |> Array.filter (fun p -> p.Name <> "RowID" && 
                             (p.PropertyType = typeof<float32> || p.PropertyType = typeof<int>))
    |> Array.map (fun p -> p.Name)

// calculate the histogram grid
printfn "Generating histograms..."
let grid = HistogramUtils.PlotAllHistograms<TaxiTripWithDuration>(taxiTrips, columnNames, columns = 4, rows = 4)

// save the grid
grid.SavePng("histograms.png", 1900, 1280)
```

Make sure you place this code right after the data filtering code, because the TLC dataset is full of weird taxi trips that will completely distort the histograms. You want to generate histograms from the filtered data. 

Here are the filters I used:

```fsharp
// Filter outliers
let mutable filteredData = mlContext.Data.FilterRowsByColumn(transformedData, "TripDuration", lowerBound = 0.0, upperBound = 60.0)
filteredData <- mlContext.Data.FilterRowsByColumn(filteredData, "FareAmount", lowerBound = 0.0, upperBound = 100.0)
filteredData <- mlContext.Data.FilterRowsByColumn(filteredData, "TripDistance", lowerBound = 0.0, upperBound = 15.0)
filteredData <- mlContext.Data.FilterRowsByColumn(filteredData, "TipAmount", lowerBound = 0.0, upperBound = 15.0)
```

These are almost the same filters I have used previously, but I had to remove the passenger count filter and set all lower bounds to zero. I had to make this change, because Scottplot cannot calculate a histogram from a column with a nonzero lower bound. If you try, you'll get an error message.

Add or enable the code that generates the histogram grid and adjust the data filters. Then run your app, choose to load the parquet file, and examine the histograms. Does anything look different? 
{ .homework }

Here's what I got:

![Histogram Grid For Full Dataset](../img/histograms-parquet.png)
{.img-fluid .mb-4}

This actually looks quite nice. The **TripDuration** and **TripDistance** histograms look okay with a long tail that's actually not too long. Looking at **FareAmount**, we could consider a new upper bound of $60 to shrink the long tail. And did  you notice the little blip at $52? This is the same statistical artefact that we also noticed in the scatterplot grid. There are a statistically distinct number of taxi trips with a fare amount of exactly $52, for unknown reasons. 

The **ImprovementSurcharge** histogram shows crazy outliers with a surcharge of up to $3000. This completely distorts the **TotalAmount** histogram. Fortunately, we decided to use **FareAmount** as the label, so we don't have to worry about this. 

Based on these new histograms, decide if you want to adjust your data filters.
{ .homework }

#### Generate The Correlation Matrix

Now let's do the correlation matrix. Here is the code you need to enable in your app:

```fsharp
// Calculate the correlation matrix
let matrix = CorrelationUtils.CalculateCorrelationMatrix<TaxiTripWithDuration>(taxiTrips, columnNames)

// plot correlation matrix
let plot = CorrelationUtils.PlotCorrelationMatrix(matrix, columnNames)

// Save the plot to a file
plot.SavePng("correlation-heatmap.png", 900, 800)
```

Make sure you place this code right after the histogram code, because you want to calculate the Pearson correlation matrix from the same data as shown in the histograms, for consistency. 

Add or enable the code that generates the Pearson correlation matrix. Then run your app, choose to load the parquet file, and examine the matrix. Does anything look different? 
{ .homework }

Here's what I got:

![Correlation Matrix For Full Dataset](../img/correlation-parquet.png)
{.img-fluid .mb-4}

Let's zoom in on the **FareAmount** column and compare it to the old heatmap. The new heatmap for all 8.1 million trips is at the top, and the old heatmap for the first 10,000 trips is at the bottom:

![Correlation Matrix For Full Dataset](../img/correlation-parquet-detail-1.png)
{.img-fluid .mb-4}
![Correlation Matrix For Full Dataset](../img/correlation-parquet-detail-2.png)
{.img-fluid .mb-4}

 If we ignore every column to the right of **FareAmount**, we can see that the correlation factors are more or less the same. There is a slightly stronger correlation with **TripDuration**, but the **RatecodeID** is now a lot weaker. 

Based on the new correlation matrix, decide if you want to adjust your feature columns.
{ .homework }

#### Generate The Scatterplot Matrix

Finally, let's generate the scatterplot matrix to see if we can still spot linear relationships between our prime feature candidates and the fare amount. The correlation matrix did not reveal any new feature candidates so we keep the existing set of **TripDuration**, **TripDistance** and **RatecodeID** (even though the RatecodeID correlation is now very weak). 

Here is the code you need to enable in your app:

```fsharp
// plot scatterplot matrix
let scatterPlotColumns = [| "TripDuration"; "TripDistance"; "RatecodeID"; "FareAmount" |]
let smplot = ScatterUtils.PlotScatterplotMatrix<TaxiTripWithDuration>(taxiTrips, scatterPlotColumns)

// Save the plot to a file
smplot.SavePng("scatterplot-matrix.png", 1900, 1200)
```

Make sure you place this code right after the correlation matrix code, again for consistency. 

Add or enable the code that generates the scatterplot matrix. Then run your app, choose to load the parquet file, and examine the matrix. Does anything look different? 
{ .homework }

Here's what I got:

![Scatterplot Grid For Full Dataset](../img/scatterplot-parquet.png)
{.img-fluid .mb-4}

You can see the problem right away. There still is a vaguely linear relationship between the fare amount and the trip distance and duration, but now there's a lot more noise in the graphs. This dataset is full of outliers and we can no longer easily predict the fare for any given trip. 

Based on the new scatterplot grid, decide if you want to adjust your outlier filters.
{ .homework }

Here are the new filters I'm going to use:

- **TripDuration** between 0 and 60 minutes (must be greater than 0)
- **TripDistance** between 0 and 15 miles (must be greater than 0)
- **FareAmount** between $0 and $60 (must be greater than 0)
- **TipAmount** between $0 and $15
- **PassengerCount** at least 1
- **RatecodeID** between 1 and 6

These filters will crash the histogram code, so I'll comment out those lines in my app.

In the next lesson, you'll have the opportunity to optimize your data transformation pipeline and regression model to get the best possible predictions for this dataset. How accurate can you make the fare predictions?  

{{< /encrypt >}}