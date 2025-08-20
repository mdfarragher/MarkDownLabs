---
title: "Plot Median House Value By Median Income"
type: "lesson"
layout: "default"
sortkey: 41
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

Before we start, let's remove all the old code we don't need anymore. The data loading code can stay but we'll delete all the plotting code because we don't want the agent to get confused.

Always remove dead code, incorrect code, or workarounds you don't agree with from the code base before giving your agent the next assignment. This keeps your code clean and prevents the agent from going off the rails.
{ .tip }

Delete any code you don't need anymore, and don't worry about compile errors. The agent can fix those in the next pass.

Another trick you can use is to add the following comment in your Program.cs file at the point where you want the agent to add new code:

```csharp
// the rest of the code goes here
```

This will guide the agent to the correct location in your code base where you want to add new code.

Now enter the following prompt:

"Write C# code using ScottPlot to generate a scatterplot of the median house value column versus the median income column."
{ .prompt }

Inspect your new code. Make sure the agent uses `Plot.Add.ScatterPoints` instead of `Plot.Add.Scatter`, because the latter draws lines between the data points, and we don't want that here.

For reference, [here is the ScottPlot documentation on scatterplots](https://www.scottplot.net/cookbook/5.0/Scatter/).

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
