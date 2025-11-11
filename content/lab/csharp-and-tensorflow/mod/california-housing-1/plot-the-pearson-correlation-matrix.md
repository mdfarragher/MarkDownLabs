---
title: "Plot The Pearson Correlation Matrix"
type: "lesson"
layout: "default"
sortkey: 60
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

You should see the following code in your project:
```csharp
// Python pandas-style correlation calculation
public static NDArray CalculateCorrelationMatrix(params NDArray[] features)
{
    // Stack features into matrix (like pandas DataFrame)
    var data_matrix = np.column_stack(features);
    
    // Calculate correlation matrix using NumSharp + MathNet
    var n_features = features.Length;
    var corr_matrix = np.zeros((n_features, n_features));
    
    for (int i = 0; i < n_features; i++)
    {
        for (int j = 0; j < n_features; j++)
        {
            var corr = Correlation.Pearson(features[i].ToArray<double>(), features[j].ToArray<double>());
            corr_matrix[i, j] = corr;
        }
    }
    
    return corr_matrix;
}
```

This simplified approach accepts NumSharp arrays directly (like pandas Series) and computes correlations using vectorized operations, similar to pandas.DataFrame.corr().

When working with correlation matrices, always examine both positive and negative correlations. Strong correlations (above 0.7 or below -0.7) indicate features that might be redundant or could be combined into composite features.
{ .tip }

#### Print the Correlation Matrix

Now ask the agent to implement the next method:

"Implement the PrintCorrelationMatrix method to print a nice matrix on the console using unicode lines. Use the BetterConsoleTables package."
{ .prompt }

You should see the following code in your project:
```csharp
public static void PrintCorrelationMatrix(Matrix<double> matrix, string[] columnNames)
{
    var table = new Table();
    
    // Add header row
    var headers = new[] { "" }.Concat(columnNames).ToArray();
    table.AddColumn(headers);
    
    // Add data rows
    // Add data rows using LINQ
    columnNames.Select((name, i) => new[] { name }
        .Concat(Enumerable.Range(0, columnNames.Length)
            .Select(j => matrix[i, j].ToString("F3")))
        .ToArray())
    .ToList().ForEach(row => table.AddRow(row));
    
    Console.WriteLine(table.ToString());
}
```

This code uses BetterConsoleTables to create a formatted correlation matrix table. It builds a table with column names as headers and rows, formatting correlation values to 3 decimal places for readability.

You should get output that looks like this:

![Correlation Matrix](../img/correlation-console.png)
{ .img-fluid .mb-4 }

You can clearly see that the **TotalRooms**, **TotalBedrooms**, **Population** and **Household** columns are strongly correlated. So we could consider condensing them into a single feature for machine learning training.

It's also interesting to look at the final column in the matrix. Notice how only **MedianIncome** is correlated to the median house value? This means that the median income level at the location of the apartment block most strongly affects the median house value in that block. And that makes perfect sense when you think about it. House prices are indeed strongly correlated to neighborhood income level.

#### Plot the Correlation Heatmap

Now let's see if Copilot can generate a heatmap for us with ScottPlot:

"Implement the PlotCorrelationMatrix method to plot a heatmap of the correlation matrix, using ScottPlot."
{ .prompt }

You should see the following code in your project:
```csharp
public static void PlotCorrelationMatrix(Matrix<double> matrix, string[] columnNames)
{
    var plt = new ScottPlot.Plot(800, 800);
    
    // Convert matrix to 2D array for heatmap
    var heatmapData = new double[matrix.RowCount, matrix.ColumnCount];
    for (int i = 0; i < matrix.RowCount; i++)
    {
        for (int j = 0; j < matrix.ColumnCount; j++)
        {
            heatmapData[i, j] = matrix[i, j];
        }
    }
    
    // Add heatmap
    var heatmap = plt.Add.Heatmap(heatmapData);
    heatmap.Colormap = ScottPlot.Colormaps.RdYlBu;
    
    // Set labels
    plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(
        Enumerable.Range(0, columnNames.Length).Select(i => (double)i).ToArray(),
        columnNames);
    plt.Axes.Left.TickGenerator = new ScottPlot.TickGenerators.NumericManual(
        Enumerable.Range(0, columnNames.Length).Select(i => (double)i).ToArray(),
        columnNames);
    
    plt.Title("Correlation Matrix Heatmap");
    plt.SavePng("correlation_heatmap.png");
}
```

This code creates a correlation matrix heatmap using ScottPlot.Add.Heatmap. It converts the MathNet matrix to a 2D array, applies a red-yellow-blue colormap, and sets custom tick labels for both axes using the column names.

Always save your correlation visualizations when doing exploratory data analysis. Heatmaps make it easy to spot patterns and relationships that might not be obvious in numerical tables.
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

This will produce a new class file called `CorrelationUtils`, with all of the code for creating, printing and plotting the correlation matrix. You can now use this class in other projects.

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

https://codeberg.org/mdft/ml-mlnet-csharp/src/branch/main/CaliforniaHousing/CorrelationUtils.cs

#### Summary

The correlation matrix is a great tool to figure out which dataset columns are strongly correlated to the label column we're trying to predict. We must include these columns in machine learning training, because they will strongly contribute to good predictions.

We can also find feature columns that are strongly correlated to each other. These columns can be combined into one. For example, we could combine **TotalRooms**, **TotalBedrooms**, **Population** and **Households** into a new composite column named **RoomsPerPerson**, by dividing the total number of rooms by the population.

In the next lesson, we're going to do exactly that.