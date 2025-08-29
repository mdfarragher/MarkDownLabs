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

In Visual Studio Code, add the following code at the end of your **Program.fs** file:

```fsharp
// Calculate the correlation matrix
let matrix = CalculateCorrelationMatrix<HousingData>(houses, columnNames)

// print correlaton matrix
PrintCorrelationMatrix(matrix, columnNames)

// plot correlation matrix
PlotCorrelationMatrix(matrix, columnNames)
```

This creates a nice scaffold for the agent to work with. We already have the array of dataset column names in `columnNames` and the array of `HousingData` instances in `houses`. The three lines of code calculate the Pearson correlation matrix, print the matrix on the console, and plot the matrix as a heatmap. 

 A code scaffold is a great trick to run your agent incrementally, asking it to implement each method in turn. It also forces the agent to use the same data every time, ensuring that the final code works as expected.
 { .tip }

Now we can ask the agent to implement the `CalculateCorrelationMatrix`, `PrintCorrelationMatrix` and `PlotCorrelationMatrix` methods for us. Let's do them one by one.

#### Calculate the Correlation Matrix

Enter the following prompt in Copilot:

"Implement the CalculateCorrelationMatrix function with F# code that calculates the Pearson correlation matrix for all columns in the dataset. Use reflection to access each field in the HousingData type, and MathNet.Numerics to calculate the correlation matrix."
{ .prompt }

Note how we're guiding the agent by explicitly mentioning **MathNet.Numerics**? We do that, because the Numerics library is the easiest way to calculate a correlation matrix. We can use other libraries (like Deedle or NumSharp), but with Numerics we only need a single call to `Correlation.PearsonMatrix` to calculate the matrix!

If you have a preference for a specific library, mention this in your prompt. This is much better than having the agent pick a library at random and write convoluted code to make everything work. 
{ .tip }

Your prompt should produce code similar to this:

```fsharp
let CalculateCorrelationMatrix<'T> (data: 'T[]) (featureColumns: string[]) =

    // Build a jagged array where each inner array is a feature column
    let matrix = 
        featureColumns
        |> Array.map (fun column ->
            data
            |> Array.map (fun row -> Convert.ToDouble(typeof<'T>.GetProperty(column).GetValue(row)))
        )

    // Calculate correlation matrix for the feature columns
    Correlation.PearsonMatrix(matrix)
```

Note the nested `Array.map` calls that build an array of floats for every field in the `'T` type. This jagged array is then passed on to `Correlation.PearsonMatrix` to calculate the correlation matrix.

#### Print the Correlation Matrix

Now ask the agent to implement the next method:

"Implement the PrintCorrelationMatrix function with the BetterConsoleTables package to print a nice correlation matrix on the console. Annotate each correlation factor like this: <br> '++/--' for a strong positive/negative correlation (|r| > 0.7) <br> '+/-' for a moderate positive/negative correlation (0.4 < |r| < 0.7) <br> '~' for a very weak or no correlation (|r| < 0.2) <br> (no symbol) for a weak correlation (0.2 < |r| < 0.4)"
{ .prompt }

Your AI agent will generate code that might look like this:

```fsharp
let PrintCorrelationMatrix (mathNetMatrix: Matrix<double>) (columnNames: string[]) =
    let table = new Table(TableConfiguration.Unicode())
    table.AddColumn("") |> ignore
    columnNames |> Array.iter (fun col -> table.AddColumn col |> ignore)

    // Helper to annotate correlation
    let annotate r =
        if r > 0.7 then "++"
        elif r > 0.4 then "+"
        ...
 
    let matrix = mathNetMatrix.ToArray()
    let n = columnNames.Length
    for i in 0 .. n-1 do
        let row =
            Array.append 
                [| box columnNames.[i] |]
                [| for j in 0 .. n-1 ->
                    let r = matrix.[i,j]
                    box (sprintf "%.2f %s" r (annotate r))
                |]
        table.AddRow(row) |> ignore
    printfn "%s" (table.ToString())
```

There are several nice F# techniques in this code. Note the call to `Array.iter` that sets up the table column headers in a single line of code. And further down, `Array.append` joins two arrays together (using the `[| ... |]` syntax): one with only a column name, and one with the correlation factors for that column. This builds up the table, row by row, until the completed table is printed with a `printfn`. 

The correlation matrix looks like this:

![Correlation Matrix](../img/correlation-console.png)
{ .img-fluid .mb-4 }

You can clearly see that the **TotalRooms**, **TotalBedrooms**, **Population** and **Household** columns are strongly correlated. So we could consider condensing them into a single feature for machine learning training.

It's also interesting to look at the final column in the matrix. Notice how only **MedianIncome** is correlated to the median house value? This means that the median income level at the location of the apartment block most strongly affects the median house value in that block. And that makes perfect sense when you think about it. House prices are indeed strongly correlated to neighborhood income level.

#### Plot the Correlation Heatmap

Now let's see if Copilot can generate a heatmap for us with ScottPlot:

"Implement the PlotCorrelationMatrix function to plot a heatmap of the correlation matrix, using ScottPlot."
{ .prompt }

Your AI agent will write code that looks like this:

```fsharp
let PlotCorrelationMatrix (matrix : Matrix<double>) (featureNames : string[]) =

    // Convert MathNet matrix to 2D array for ScottPlot
    let correlationArray = matrix.ToArray()

    // Create a heatmap
    let plot = new Plot()
    let heatmap = plot.Add.Heatmap(correlationArray)
    heatmap.Colormap <- Colormaps.Turbo()
    let colorbar = plot.Add.ColorBar(heatmap)
    plot.Title("Feature Correlation Matrix")

    // Configure X-axis ticks with feature names
    let n = featureNames.Length
    plot.Axes.Bottom.TickGenerator <- TickGenerators.NumericManual(
        [| 0 .. n-1 |] |> Array.map (fun v -> double v),
        labels = featureNames)

    // Configure Y-axis ticks with feature names
    plot.Axes.Left.TickGenerator <- new TickGenerators.NumericManual(
        [| 0 .. n-1 |] |> Array.map (fun v -> double v),
        labels = featureNames)

    // Add text annotations for correlation values
    for i = 0 to matrix.RowCount-1 do
        for j = 0 to matrix.ColumnCount-1 do
            let value = matrix.[i, j]
            let text = value.ToString("F2")
            let annotation = plot.Add.Text(text, j, i)
    plot
```

Note the two `Tickgenerators` for adding the correct labels to the x- and y-axis. We also add a nice `ColorBar` to the right edge of the plot with the `Turbo` color scale. And the nested loop at the bottom adds text to each cell of the heatmap.

But when I ran this code, I got a heatmap with several mistakes:

-    The numbers in the heatmap did not correspond to the colors of the cells
-    The colored backgrounds were offset by 0.5 in each cell

After some hacking, I discovered that both axes of the heatmap needed to be shifted by -0.5, and that the vertical axis of the heatmap needs to be in reverse order for the plot to make sense. Here's how you do that:

```fsharp
    // Flip the y-axis and shift all ticks by 0.5
    plot.Axes.SetLimits(-0.5, float (featureNames.Length) - 0.5, float (featureNames.Length) - 0.5, -0.5)
```

So always double-check the output of your agent-generated code. My initial heatmap looked perfectly fine at first glance, and only after a closer inspection did I notice that the numbers didn't make sense. 

Here is the final heatmap in all its glory:

![Correlation Heatmap](../img/correlation-heatmap.png)
{ .img-fluid .mb-4 }

The correlation blocks are clearly visible now. **TotalRooms**, **TotalBedrooms**, **Population** and **Households** are highly correlated (all numbers go up when housing blocks get bigger).

**Longitude** and **Latitude** are negatively correlated. This makes sense if you think about the shape of the state of California: as you travel south through the state, you also move east because the entire state is angled in the south-east direction.

For machine learning, we are most interested in what correlates with **MedianHouseValue**, our training label. The heatmap clearly shows that there's only a single column with a strong correlation: **MedianIncome**.

#### Create a Utility Class

We'll return to the correlation matrix in later lessons, so this is a good moment to preserve your work.

In Visual Studio Code, create a new **CorrelationUtils.fs** file and copy the implementation of the `CalculateCorrelationMatrix`, `PrintCorrelationMatrix` and `PlotCorrelationMatrix` methods into it. Make sure the file starts like this:

```fsharp
module CorrelationUtils

open ScottPlot
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra
open BetterConsoleTables
open System
```

The `module` keyword on the first line defines a new F# module called **CorrelationUtils**. We can now use this module in any other file by simply putting `open CorrelationUtils` at the top of the file. 

So go ahead and add this line to the top of your **Program.fs** file:

```fsharp
open CorrelationUtils
```

There's one more thing we need to do. The F# compiler does not automatically determine the order of compilation to prevent dependency conflicts. You have to set this order manually, in this case by specifying that **CorrelationUtils.fs** needs to be compiled before **Program.fs**. 

You specify the compilation order in the **CaliforniaHousing.fsproj** project file, like this:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <ItemGroup>
    <Compile Include="CorrelationUtils.fs" />
    <Compile Include="Program.fs" />
  </ItemGroup>

</Project>
```

The **ItemGroup** element lists all F# project files in the exact order in which they will be compiled. So make sure to put any modules near the top of the list.  

If you get stuck or want to save some time, feel free to download my completed CorrelationUtils module from Codeberg and use it in your own project:

https://codeberg.org/mdft/ml-mlnet-fsharp/src/branch/main/CaliforniaHousing/CorrelationUtils.fs

#### Summary

The correlation matrix is a great tool to figure out which dataset columns are strongly correlated to the label column we're trying to predict. We must include these columns in machine learning training, because they will strongly contribute to good predictions.

We can also find feature columns that are strongly correlated to each other. These columns can be combined into one. For example, we could combine **TotalRooms**, **TotalBedrooms**, **Population** and **Households** into a new composite column named **RoomsPerPerson**, by dividing the total number of rooms by the population.

In the next lesson, we're going to do exactly that.