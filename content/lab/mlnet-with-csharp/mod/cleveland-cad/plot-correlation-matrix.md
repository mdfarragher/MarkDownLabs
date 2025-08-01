---
title: "Plot The Pearson Correlation Matrix"
type: "lesson"
layout: "default"
sortkey: 44
---

It's very easy to calculate and plot the Pearson correlation matrix for the Cleveland CAD dataset, because we already imported the **CorrelationUtils** class. 

All you have to do is add following code:

```csharp
// Calculate the correlation matrix
var matrix = CorrelationUtils.CalculateCorrelationMatrix<HeartData>(heartDataList, columnNames);

// plot correlation matrix
var plot = CorrelationUtils.PlotCorrelationMatrix(matrix, columnNames);

// Save the plot to a file
plot.SavePng("correlation-heatmap.png", 900, 800);
```

And that’s it! When you run the code, you'll get the Pearson correlation heatmap saved as a PNG image in the same file as usual. 

Homework: add code to generate the correlation matrix. Then run your app and examine the matrix. What do you notice? Write down your conclusions.  
{ .homework }

Here is what I got:

![Correlation Heatmap](../img/correlation-heatmap.png)
{ .img-fluid .mb-4 }

Looking at the bottom row, the columns that strongly correlate with the label are either orange or blue, and clearly visible. They are, in descending order of correlation strength:

- **Thalassemia**
- **NumMajorVessels**
- **ExerciseInducedAngina**
- **STDepression**
- **MaxHeartRate**
- **ChestPainType**

All of these columns have correlation factors above 0.4 (or below -0.4) and strongly affect the diagnosis. The remaining group of columns with moderate to strong correlation factors (above 0.2) are:

- **Slope**
- **Sex**
- **Age**

It's interesting to see how the outcome of the blood test has the smallest impact on the diagnosis. Both **FastingBloodSugar** and **Cholesterol** have correlation factors below 0.1. Also interesting is that **Sex** is more strongly correlated to cardiovascular disease than **Age**. Both factors are positive, meaning the risk of getting the disease is higher for men and goes up as we age. 

#### Choosing Which Features To Train On

We now need to decide on a cutoff value for the correlation matrix. Any features with correlation factors below the cutoff value will not be considered for machine learning training. 

But which cutoff value should we use?

There's actually a mathematical formula for that. We can calculate the cutoff factors for a dataset of 303 rows and 13 features, for four different selection strategies:

| Strategy   | Correlation cutoff value |
|------------|--------------------------|
| None       | 0.0   |
| Default    | 0.113 |
| Strict     | 0.148 |
| Bonferroni | 0.173 |

The strategies are:

- **None**. We keep every feature and train our model on the full set of columns. This is a good strategy if we want to minimize the risk of false negatives.

- **Default**, the everyday threshold. It lets us be wrong about one feature out of twenty on average (5% risk) and is fine when a false positive or negative is not a big deal. 

- **Strict** turns the dial up. Only features that look clearly connected to the label survive (~1% risk), so we miss a few weak signals but we can feel safer about the ones we keep. 

- **Bonferroni** is the safety-first mode. Because we’re testing 13 features at once, we slice that 5% risk into 13 tiny pieces and demand a much stronger signal before declaring a win; this almost eliminates false positives, but at the cost of throwing away any feature whose link to the label is subtle.

In healthcare, a false negative is a catastrophic error because we would be sending a sick patient home without treatment. Therefore, we should lean toward the **Default** or **None** strategies. More strict strategies like Bonferroni are designed to protect against false discoveries, but in a medical-screening context they do so by discarding moderate yet genuine disease signals, thereby raising the odds that the model overlooks a sick patient.

In the next lesson, we're going to generate the scatterplot grid to see if any features have a linear relationship with the diagnosis. 
