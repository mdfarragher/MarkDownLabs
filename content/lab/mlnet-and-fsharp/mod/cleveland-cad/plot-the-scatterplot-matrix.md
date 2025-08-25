---
title: "Plot The Scatterplot Matrix"
type: "lesson"
layout: "default"
sortkey: 70
---

# Plot The Scatterplot Matrix

The **Diagnosis** column is a boolean label, either 0 for healthy patients or 1 for sick patients. So, for the scatterplots, we should only plot features with high cardinality (= having lots of discrete values). That will produce nice plots where we can hopefully spot some linear relationships. 

The high-cardinality columns in the Cleveland CAD dataset are **Age**, **RestingBloodPressure**, **Cholesterol**, **MaxHeartRate** and **STDepression**.

{{< encrypt >}}

#### Create a Scatterplot Matrix

It's easy to generate the scatterplot matrix for the Cleveland CAD dataset, because you already imported the **ScatterUtils** class. 

All you need to add is the following code:

```fsharp
// column names for scatterplot
let scatterPlotColumns = [| "Age"; "RestingBloodPressure"; "Cholesterol"; "MaxHeartRate"; "STDepression"; "Diagnosis" |]

// plot scatterplot matrix
Console.WriteLine("Generating scatterplot matrix...")
let smplot = ScatterUtils.PlotScatterplotMatrix<HeartData>(heartDataList, scatterPlotColumns)

// Save the plot to a file
smplot.SavePng("scatterplot-matrix.png", 1900, 1200)
```

When you run the code, you'll get the scatterplot matrix saved as a PNG image in the same file as usual. 

Homework: add code to generate the scatterplot matrix. Then run your app and examine the matrix. What do you notice? Write down your conclusions.  
{ .homework }

Here is what I got:

![Correlation Heatmap](../img/scatterplot-matrix.png)
{ .img-fluid .mb-4 }

If you look at the plots in the bottom row (diagnosis by feature), you'll notice that outliers are important in healthcare. A high **RestingBloodPressure**, low **MaxHeartRate** and high **STDepression** leads to a positive diagnosis. 

The **Cholesterol** value does not clearly drive the diagnosis, and our outlier with a cholesterol level of 564 actually turned out to be healthy! We'll probably have to remove this patient, or the model might start thinking that high cholesterol is a good thing. 

There are a few vaguely linear relationships in the other plots. In the top row, you can see that **RestingBloodPressure** and **Cholesterol** go up and **MaxHeartRate** goes down as we age.

Let's get rid of outliers and regenerate the scatterplot matrix.

#### Filter The Dataset

Let's start by filtering the data and removing the high cholesterol value. Locate the line of code that calls Fit on the pipeline to fill in the missing values:

```fsharp
// Apply the transformation pipeline
let transformedData = pipeline.Fit(rawDataView).Transform(rawDataView)
```

And replace it with this:

```fsharp
// Filter out cholesterol above 400
let filteredData = mlContext.Data.FilterRowsByColumn(rawDataView, "Chol", upperBound = 400.0)

// Apply the transformation pipeline
let transformedData = pipeline.Fit(filteredData).Transform(filteredData)
```

If you want, you can also filter out a resting blood pressure above 180 and a max heart rate below 80:

```fsharp
// Filter out resting blood pressure above 180
let filteredData = mlContext.Data.FilterRowsByColumn(filteredData, "TrestBps", upperBound = 180.0)

// Filter out max heart rate below 80
let filteredData = mlContext.Data.FilterRowsByColumn(filteredData, "Thalac", lowerBound = 80.0)
```

And with that, your scatterplot matrix should now look like this:

![Scatterplot Matrix](../img/scatterplot-matrix-2.png)
{ .img-fluid .mb-4 }

Nothing really jumps out right now, but we can still vaguely see three linear relationships:

- **Age** by **RestingBloodPressure**
- **Age** by **Cholesterol**
- **Age** by **MaxHeartRate**

And the two features that most clearly drive the diagnosis are now **STDepression** and **MaxHeartRate**.

#### Summary

We have used the Pearson correlation matrix to identify the features most strongly correlated with the label, and we generated a scatterplot matrix to identify any relationships between these features and the label.

We're now ready to implement the data transformations and build the machine learning pipeline. 

{{< /encrypt >}}