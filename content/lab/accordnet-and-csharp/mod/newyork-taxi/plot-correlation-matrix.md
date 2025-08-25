---
title: "Plot The Pearson Correlation Matrix"
type: "lesson"
layout: "default"
sortkey: 42
---

It's very easy to calculate and plot the correlation matrix for the New York TLC dataset, because we can use the **CorrelationUtils** helper class from the previous lab. The class is completely reusable and will work on any dataset.

Let's see if our AI agent is smart enough to import code from another project.

{{< encrypt >}}

#### Import The CorrelationUtils Helper Class

For the next prompt, you'll need the raw url of the CorrelationUtils class you created in the previous lab. We will ask the agent to import the class into our current project.

I pushed the class to a repository on Codeberg, so my url is: https://codeberg.org/mdft/ml-mlnet-csharp/raw/branch/main/CaliforniaHousing/CorrelationUtils.cs. Here is the prompt I used:

"Copy the entire CorrelationUtils class from this repository: codeberg.org/mdft/ml-mlnet-csharp/raw/branch/main/CaliforniaHousing/CorrelationUtils.cs. Add the class to this project."
{ .prompt }

You can see that I'm referring the agent to my CorrelationUtils class stored on Codeberg. You will have to substitute the url with your own (or you can use my class, that's fine too).

The class needs the BetterConsoleTables and MathNet.Numerics packages, so let's install them too:

"Add the Nuget packages BetterConsoleTables and MathNet.Numerics"
{ .prompt }

That should fix any Intellisense errors in your code. Now we can simply add calls to `CalculateCorrelationMatrix` and `PlotCorrelationMatrix`. 

#### Calculate and Plot the Correlation Matrix

In your main program class, locate the following code snippet:

```csharp
// get column names, skip row id and non-numeric columns
var columnNames = (from p in typeof(TaxiTripWithDuration).GetProperties()
                    where p.Name != "RowID"
                            && (p.PropertyType == typeof(float)
                            || p.PropertyType == typeof(int))
                    select p.Name).ToArray();
```

And then put this code right underneath it:

```csharp
// Calculate the correlation matrix
var matrix = CorrelationUtils.CalculateCorrelationMatrix<TaxiTripWithDuration>(taxiTrips, columnNames);

// plot correlation matrix
var plot = CorrelationUtils.PlotCorrelationMatrix(matrix, columnNames);

// Save the plot to a file
plot.SavePng("correlation-heatmap.png", 900, 800);
```

And that's it! That should be enough to calculate and plot the Pearson correlation matrix, complete with the new **TripDuration** column.

Here is the matrix in all its glory:

![Correlation Heatmap](../img/correlation-heatmap.png)
{ .img-fluid .mb-4 }

There's a correlation block visible in the bottom right corner of the matrix, where all the monetary dataset columns are positively or negatively correlated with each other. There are some fun insights in this block, for example that **TipAmount** is moderately correlated with **PaymentType**. 

But the most interesting part of the matrix is the last row which shows how each feature correlates with **TotalAmount**. Strongly correlated features in this row are **TripDuration**, **TripDistance** and **RateCodeID**. In other words, longer trips that cover more distance are more expensive, and the rate code (which indicates if you're travelling to JFK airport for example) also has a strong impact on the total fare. 

It's also clear from the matrix that we can safely ignore **VendorID**, **PassengerCount**, **PULocationID**, **DOLocationID**, **MtaTax** and **ImprovementSurchage** because these columns have almost no impact whatsoever on the total amount. 

#### Choose The Label To Predict

If you look closely at the correlation matrix, you'll see that **TotalAmount** is strongly correlated to **FareAmount**, **TipAmount**, **TollsAmount** and **Extra**. And this makes sense, because the total amount is simply the fare plus tip plus tolls plus extra charges. 

But expecting a machine learning model to accurately predict tips is going to be difficult, because tips will be more or less random per trip. The model will also struggle with tolls, because they depend on the route that the taxi took and our dataset only contains trip duration and distance values per trip. 

Having no reliable way to calculate (or predict) tolls and tips makes it very difficult for a machine learning model to accurately predict **TotalAmount**. So I propose that we predict **FareAmount** instead, and remove all other monetary data columns from the training set.

If you compare the rows for **FareAmount** and **TotalAmount** in the correlation matrix, you'll see that they have near identical correlation factors. This suggests that we can safely switch labels without losing any predictive accuracy. 

#### Next Steps

We're almost ready to build the machine learning pipeline. But before we do that, let's analyze the relationship between **FareAmount**, the label we're going to predict, and **TripDuration**, **TripDistance** and **RateCodeID** to see if we can spot any linear relationships between them. 

{{< /encrypt >}}
