---
title: "Plot Median House Value By Median Income"
type: "lesson"
layout: "default"
sortkey: 50
---

In machine learning training, we're looking for features with a balanced distributions of values. Basically, you want the feature histogram to look like a symmetric hump, with flanks on the left and right tapering to zero. 

But do you remember the histogram of the **median_house_value** column from the previous lab lesson? It looks like this:

{{< encrypt >}}

![Histogram of Median House Value](../img/medianhousevalue-histogram.png)
{ .img-fluid .mb-4 }

So why does this histogram have that weird peak on the right?

The peak represents a statistically significant number of housing blocks in the dataset with a house value around half a million dollars. This is an example of **bias**, and it will make it harder for a machine learning model to generate good predictions.

Let's explore this a little further by creating a scatterplot of median house value by median income.

#### Create a Scatterplot

Enter the following prompt:

"Write F# code using ScottPlot to generate a scatterplot of the median house value column versus the median income column."
{ .prompt }

You will probably get code that looks like this:

```fsharp
// Extract median_house_value column
let medianHouseValue =
    houses
    |> Seq.map (fun row -> float row.median_house_value)
    |> Seq.toArray

// Extract median_house_value column
let medianIncome =
    houses
    |> Seq.map (fun row -> float row.median_income)
    |> Seq.toArray
```

These two statements are very similar to what you saw in the previous lesson. They use `Seq.map` to extract the **median_house_value** and **median_income** columns from the `houses` array, and `Seq.toArray` to convert the sequence of floats to an array. 

With that in place, the plotting code is very simple and looks like this:

```fsharp
// Create a scatterplot
let scPlot = new Plot()
scPlot.Add.ScatterPoints(medianIncome, medianHouseValue) |> ignore

// Customize appearance
scPlot.Title("Median House Value vs Median Income")
scPlot.XLabel("Median Income")
scPlot.YLabel("Median House Value")

// Save the plot
scPlot.SavePng("income-vs-housevalue.png", 600, 400) |> ignore
```

Note the use of `|> ignore`, which discards function return values that we don't actually use. Without these statements, the compiler would generate warnings. 

You should get a plot that looks like this:

![Scatterplot of Median House Value by Median Income](../img/income-vs-housevalue.png)
{ .img-fluid .mb-4 }

As the median income level increases, the median house value also increases. There's still a big spread in the house values, but a vague 'cigar' shape is visible which suggests a roughly linear relationship between these two variables.

But look at the horizontal line at 500,000. What's that all about?

This is what **clipping** looks like. The creator of this dataset has clipped all housing blocks with a median house value above $500,000 back down to $500,000. We see this appear in the graph as a horizontal line that disrupts the linear 'cigar' shape.

We will have to deal with this in the upcoming lessons, because we don't want a machine learning model to learn that house prices can somehow never be higher than half a million dollars.

#### Summary

Did I mention that visualization is one of the most important sanity checks in machine learning? It not only helped us identify outliers in the total_rooms column, but it also clearly shows clipping in the median_house_value column.

In upcoming lessons, I'll show you how to deal with outliers and clipping.

{{< /encrypt >}}