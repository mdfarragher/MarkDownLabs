---
title: "Plot The Pearson Correlation Matrix"
type: "lesson"
layout: "default"
sortkey: 42
---

When you train a machine learning model on a dataset, you always face the challenge of deciding which columns to include in the training.

You only want to include columns that are uncorrelated with each other. This means that the columns behave independently from one another, and do not increase or decrease together.

In the California Housing dataset, we have two sets of columns that at first glance appear to be highly correlated: 

- **total_rooms** and **total_bedrooms**
- **population** and **households**

Think about it: a housing block with many rooms will probably also have a lot of bedrooms. And a block with a large population will also host many households.

If we find strongly correlated pairs of columns, we can consider training a machine learning model on a single column. Including the other correlated columns is not useful, because they are not independent variables and we would be wasting valuable algorithmic memory space.

So let's write some code (= have Copilot write it for us) to calculate the correlation matrix.

#### Set Up a Code Structure

In Visual Studio Code, remove all unwanted code from the Program.cs file, and then create the following code structure right after the data loading code:

```csharp
// get column names
var columnNames = (from p in typeof(HousingData).GetProperties()
                   where p.Name != "RowID"
                   select p.Name).ToArray();

// Calculate the correlation matrix
var matrix = CalculateCorrelationMatrix<HousingData>(housingData, columnNames);

// print correlaton matrix
PrintCorrelationMatrix(matrix, columnNames);

// plot correlation matrix
PlotCorrelationMatrix(matrix, columnNames);
```

This creates a nice scaffold for the agent to work with. We set up an array of column names (using reflection), and then call `CalculateCorrelationMatrix` to calculate the Pearson correlation matrix. The method is generic so we can reuse it in other projects.

You may be wondering why we calculate the `columnNames` array at all? It's to ensure that `PrintCorrelationMatrix` and `PlotCorrelationMatrix` use the exact same column list and present the columns in the exact same order.

 A code scaffold is a great trick to run your agent incrementally, asking it to implement each method in turn. It also forces the agent to use the same data every time, ensuring that the final code works as expected.
 { .tip }

Now we can ask the agent to implement the `CalculateCorrelationMatrix`, `PrintCorrelationMatrix` and `PlotCorrelationMatrix` methods for us.

#### Calculate the Correlation Matrix

Enter the following prompt in Copilot:

"Implement the CalculateCorrelationMatrix method with C# that calculates the Pearson correlation matrix for all columns in the dataset. Use MathNet.Numerics to calculate the matrix."
{ .prompt }

Note how we're guiding the agent by explicitly mentioning **MathNet.Numerics**? We do that, because the Numerics library is the easiest way to calculate a correlation matrix. We can use other libraries (like Deedle or NumSharp), but with Numerics we only need a single call to `Correlation.PearsonMatrix` to calculate the matrix!

If you have a preference for a specific library, mention this in your prompt. This is much better than having the agent pick a library at random and possibly generate convoluted code to make everything work. 
{ .tip }

When I ran Copilot on this prompt, I got a pile of non-reusable code with hardcoded column names everywhere and a ton of local variables to build the jagged `double[][]` array for the correlation matrix.

So I decided I hated it, deleted all the code and wrote this instead:

```csharp
public static Matrix<double> CalculateCorrelationMatrix<T>(
    List<T> data,
    string[] featureColumns)
{
    // Build a jagged array where each inner array is a feature column
    var matrix = new List<double[]>();
    foreach (var col in featureColumns)
    {
        var prop = typeof(T).GetProperty(col);
        matrix.Add(
            data.Select(h => Convert.ToDouble(prop?.GetValue(h))).ToArray()
        );
    }

    // Calculate correlation matrix for the feature columns
    return Correlation.PearsonMatrix(matrix);
}
```

My handwritten code uses reflection to access each data column in turn. Then a `foreach` loop builds the jagged array of doubles I need to calculate the correlation matrix. This is much shorter and much more elegant than what the agent generated, it is completely reusable and can be used for any dataset.

Don't hesitate to get your hands dirty and completely rewrite sections of agent-generated code. Treat your AI agent as a junior developer who sometimes gets it wrong and needs to be corrected by a more senior peer. 
{ .tip }

#### Print the Correlation Matrix

Now ask the agent to implement the next method:

"Implement the PrintCorrelationMatrix method to print a nice matrix on the console using unicode lines. Use the BetterConsoleTables package."
{ .prompt }

You should get something like this:

![Correlation Matrix](../img/correlation-console.png)
{ .img-fluid .mb-4 }

My agent went a bit overboard and decided to add extra indicators in each matrix cell to show moderate and strong positive or negative correlation. That's a very nice touch.

You can clearly see that the **TotalRooms**, **TotalBedrooms**, **Population** and **Household** columns are strongly correlated. So we could consider condensing them into a single feature for machine learning training.

It's also interesting to look at the final column in the matrix. Notice how only **MedianIncome** is correlated to the median house value? This means that the median income level at the location of the apartment block most strongly affects the median house value in that block. And that makes perfect sense when you think about it. House prices are indeed strongly correlated to neighborhood income level.

#### Plot the Correlation Heatmap

Now let's see if Copilot can generate a heatmap for us with ScottPlot:

"Implement the PlotCorrelationMatrix method to plot a heatmap of the correlation matrix, using ScottPlot."
{ .prompt }

When I ran this prompt, I got a nice heatmap. But closer inspection of the plot revealed several bugs:

-    The numbers in the heatmap did not correspond to the colors of the cells
-    The colored backgrounds were offset by 0.5 in each cell

After some hacking, I discovered that both axes of the heatmap needed to be shifted by -0.5, and that the vertical axis of the heatmap needs to be in reverse order for the plot to make sense. Here's how you do that:

```csharp
// Set the axis limits to show the full heatmap
// plot.Axes.SetLimits(0, featureNames.Length, 0, featureNames.Length);
plot.Axes.SetLimits(-0.5, featureNames.Length - 0.5, featureNames.Length - 0.5, -0.5);
```

I left the original agent-generated line in, so you can see the changes I made. I reversed the vertical axis, and shifted both axes by -0.5 to line up the tick marks with the heatmap cells.

Despite repeated prompting, my Claude 3.7 agent was unable to fix this bug and kept getting stuck generating code for the wrong version of ScottPlot.

 Always double-check the output of your agent-generated code. My initial heatmap looked perfectly fine at first glance, and only after a closer inspection did I notice that the numbers didn't make sense. 
 { .tip }

Here is the final heatmap in all its glory:

![Correlation Heatmap](../img/correlation-heatmap.png)
{ .img-fluid .mb-4 }

The correlation blocks are clearly visible now. **TotalRooms**, **TotalBedrooms**, **Population** and **Households** are highly correlated (all numbers go up when housing blocks get bigger).

**Longitude** and **Latitude** are negatively correlated. This makes sense if you think about the shape of the state of California: as you travel south through the state, you also move east because the entire state is angled in the south-east direction.

For machine learning, we are most interested in what correlates with **MedianHouseValue**, our training label. The heatmap clearly shows that there's only a single column with a strong correlation: **MedianIncome**.

#### Create a Utility Class

We'll return to the correlation matrix in later lessons, so this is a good moment to preserve your work.

In Visual Studio Code, select the implementation of the `CalculateCorrelationMatrix`, `PrintCorrelationMatrix` and `PlotCorrelationMatrix` methods (plus any associated helper methods).

Then press CTRL+I to launch the in-line AI prompt window, and type the following prompt:

"Move all of this code to a separate utility class called CorrelationUtils."
{ .prompt }

This will produce a new class file called **CorrelationUtils.cs**, with all of the code for creating, printing and plotting the correlation matrix. You can now use these methods in other projects.

When you're happy with generated code and you want to keep it, move it aside into separate class files. That keeps your main code file (Program.cs) clean and ready for the next agent experiment.
{ .tip }

And if you want to clean up your code and make it as side-effect-free as possible, you can edit `PlotCorrelationMatrix` and have it return the `Plot` instance. You can then save the grid in the main program class instead. Your main calling code will then look like this:

```csharp
// plot correlation heatmap
var plot = PlotCorrelationMatrix(matrix, columnNames);

// Save the plot to a file
plot.SavePng("correlation_heatmap.png", 900, 800);
```

If you get stuck or want to save some time, feel free to download my completed CorrelationUtils class from Codeberg and use it in your own project:

https://codeberg.org/mdft/ml-mlnet-csharp/src/branch/main/CaliforniaHousing

#### Summary

The correlation matrix is a great tool to figure out which dataset columns are strongly correlated to the label column we're trying to predict. We must include these columns in machine learning training, because they will strongly contribute to good predictions.

We can also find feature columns that are strongly correlated to each other. These columns can be combined into one. For example, we could combine **TotalRooms**, **TotalBedrooms**, **Population** and **Households** into a new composite column named **RoomsPerPerson**, by dividing the total number of rooms by the population.

In the next lesson, we're going to do exactly that.