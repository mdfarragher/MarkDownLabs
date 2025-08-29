---
title: "Plot A Histogram Of Total Rooms"
type: "lesson"
layout: "default"
sortkey: 40
---

Before you build a machine learning model, it's important to understand your data visually. Just looking at the numbers, like you did in the previous lesson, may not be enough. A good chart can clearly reveal patterns in the dataset.

In this section, you'll going to build a visualization to detect any outliers in the **total_rooms** feature.

Let's get started.

{{< encrypt >}}

#### Install ScottPlot

ScottPlot is a very nice plotting and visualization library for F# and NET that can go toe-to-toe with Python libraries like matplotlib and seaborn. We will use it in these labs whenever we want to visualize a dataset.

First, let's install the ScottPlot NuGet package. In your terminal (inside the CaliforniaHousing folder), install ScottPlot like this:

```bash
dotnet add package ScottPlot
```

Then open Visual Studio Code in the current folder, like this:

```bash
code .
```

Open the Program.fs file and remove all existing content, because we don't want the agent to get confused. Replace the content with this:

```fsharp
open ScottPlot
open Microsoft.ML
open System
```

#### Create a Histogram of total_rooms

Now let's ask Copilot to write the code for us.

At the bottom of the Copilot panel in Visual Studio Code, make sure the AI mode is set to 'Agent'. Then select your favorite model. I like to use GPT 4.1 or Claude 3.7 for coding work.

![Enable Agent Mode](../img/agent-mode.jpg)
{ .img-fluid .mb-4 .border }

Now enter the following prompt:

"Write F# code using ScottPlot to generate a histogram of the total_rooms column from the CSV file. Use the LoadFromTextFile function in ML.NET to load the file."
{ .prompt }

And let Copilot write the code for you. You should see the following data loading code in your project:

```fsharp
// Set up a data loader
let mlContext = new MLContext()
let loader =
    mlContext.Data.CreateTextLoader(
        columns = [|
            TextLoader.Column("row_id", DataKind.Single, 0)
            TextLoader.Column("longitude", DataKind.Single, 1)
            TextLoader.Column("latitude", DataKind.Single, 2)
            TextLoader.Column("housing_median_age", DataKind.Single, 3)
            TextLoader.Column("total_rooms", DataKind.Single, 4)
            TextLoader.Column("total_bedrooms", DataKind.Single, 5)
            TextLoader.Column("population", DataKind.Single, 6)
            TextLoader.Column("households", DataKind.Single, 7)
            TextLoader.Column("median_income", DataKind.Single, 8)
            TextLoader.Column("median_house_value", DataKind.Single, 9)
        |],
        hasHeader = true,
        separatorChar = ','
    )

// Load the dataset
let dataView = loader.Load("California-Housing.csv")

// Extract total_rooms column
let totalRooms =
    mlContext.Data.CreateEnumerable<HousingData>(dataView, reuseRowObject=false)
    |> Seq.map (fun row -> float row.total_rooms)
    |> Seq.toArray
```

This code calls `CreateTextLoader` to set up a text data loader with the correct column names and indices, and then calls the `Load` function to load the CSV file into a data view. 

Finally, the code calls `CreateEnumerable` to convert the data view into an enumeration of `HousingData` instances. The `Seq.map` function extracts the **total_rooms** column and converts it to a `float`, and `Seq.toArray` converts the final sequence of floats to an array.

This implementation is by the book, and exactly what we want to see in auto-generated F# machine learning code that uses Microsoft.ML.
{ .tip }

If you had not yet seen the `|>` operator in F#, it's called the **forward pipe** and takes the value on the left hand side and passes it as the last argument into the function on the right hand side. We can use it to chain multiple statements together, in this case `CreateEnumerable`, `Seq.map` and `Seq.toArray`. 

This is what the `HousingData` type looks like:

```fsharp
// Define the data schema
[<CLIMutable>]
type HousingData = {
    row_id: float32
    longitude: float32
    latitude: float32
    housing_median_age: float32
    total_rooms: float32
    total_bedrooms: float32
    population: float32
    households: float32
    median_income: float32
    median_house_value: float32
}
```

Each column in the dataset is implemented as a field, with a matching name and the correct data type. The `CLIMutable` attribute specifies that this type is mutable, meaning the fields can be modified directly without creating an entirely new copy of the type. We need this attribute, because `CreateEnumerable` can only work with mutable types.

Here is the plotting code I ended up with:

```fsharp
// Create a new plot
let plot = new Plot()

// Create a histogram
let hist = ScottPlot.Statistics.Histogram.WithBinCount(10, totalRooms)

// Add the bars to the plot
let barPlot = plot.Add.Bars(hist.Bins, hist.Counts)

// Size each bar slightly less than the width of a bin
for bar in barPlot.Bars do
    bar.Size <- hist.FirstBinSize * 0.8

// Customize appearance
plot.Title("Histogram of Total Rooms")
plot.XLabel("Total Rooms")
plot.YLabel("Frequency")

// Save the plot
plot.SavePng("totalrooms-histogram.png", 600, 400) |> ignore
```

Now let's look at the histogram. It should look like this:

![Histogram Of TotalRooms](../img/totalrooms-histogram.png)
{ .img-fluid .mb-4 }

Think about the following:

-    Do you notice the long tail (outliers)?
-    How should you deal with this?
-    Could other columns in the dataset have the same issue?

Write down which data transformation steps you are going to apply to deal with the outliers in the total_rooms column.
{ .homework }

#### Create a Histogram of Every Feature

Now let's modify the code to generate histograms for all the columns in the dataset. Enter the following prompt:

"Write F# code that creates a Scottplot multiplot, where the plots are histograms of every column in the dataset. Arrange the plots in a 3x3 grid."
{ .prompt }

There are many ways to write this code, and the AI agent will probably come up with something decent. But we're going to have to generate many histograms over the course of this training, so I would like to see reusable code here that I can apply to any other dataset in the future. 

Remember the `HousingData` type we generated earlier to load the dataset? It has all the information we need. Each field represents one column of the dataset and has the same name. So instead of putting a hardcoded list of column names in the code, we should try to get the names directly from the `HousingData` type with a bit of reflection: 

```fsharp
// Get array of column names
let columnNames = 
    typeof<HousingData>.GetProperties()
    |> Array.map (fun prop -> prop.Name )
    |> Array.filter (fun name -> name <> "row_id")
```

This code calls `GetProperties` on the `HousingData` type and uses `Array.map` and `Array.filter` to extract the property names and skip the **row_id** column. This will produce a string array of column names, directly from the HousingData type, without any prior knowledge baked in the code.

The next step is to create a `float[]` array for any given column name, so that we can calculate a histogram from it. We can do that like this:

```fsharp
// Get array of houses
let houses = 
    mlContext.Data.CreateEnumerable<HousingData>(dataView, reuseRowObject=false) 
    |> Seq.toArray

// Example: get the array of values for total_rooms
let name = "total_rooms"
let values = 
    houses 
    |> Array.map (fun row -> 
        Convert.ToDouble(typeof<HousingData>.GetProperty(name).GetValue(row)))
```

This code first calls `CreateEnumerable` to set up an array of `HousingData` instances. Next, it uses a combination of `map`, `GetProperty` and `GetValue` to access the property we want (**total_rooms**) in every `HousingData` instance and get its value. A final call to `Convert.ToDouble` converts the property value to a `float`.

The end result is that the `values` variable will contain a `float[]` array of every **total_rooms** value in the dataset. We can feed this array directly into `Histogram.WithBinCount` (see below) to set up the histogram. 

And this is how you create a multiplot with Scottplot:

```fsharp
// Build multiplot (for column names in 'columnNames' )
let grid = new Multiplot()
grid.Layout <- new MultiplotLayouts.Grid(columns = 3, rows = 3)
for name in columnNames do

    // Create a new plot
    let plot = new Plot()

    // Create array of values
    let values = 
        houses 
        |> Array.map (fun row -> 
            Convert.ToDouble(typeof<HousingData>.GetProperty(name).GetValue(row)))

    // Add the histogram to the plot as a bar chart
    let hist = ScottPlot.Statistics.Histogram.WithBinCount(20, values)
    let barPlot = plot.Add.Bars(hist.Bins, hist.Counts)
    ....

    // Add the new bar chart to the grid
    grid.AddPlot(plot)
```

This code might seem a bit convoluted, and I definitely went off the vibecoding script by discarding the output of my AI agent and writing this code by hand, but I do like the sheer reusability of this code. I can throw in any helper class I want and simply provide a string array of column names to plot a grid of any combination of histograms. Very nice!

In my experience, code generated by AI agents is often very brittle and will only work for the particular scenario it's written for. When I need solid reusable code, I often decide to write everything by hand. And I think you should too.

Don't hesitate to put your AI agent aside and write reusable code by hand. You will probably do a much better job than your AI agent at creating something that you can use again and again. 
{ .tip } 

Now let's check out the histograms. Your grid should look like this:

![Histogram Of All Columns](../img/histogram-grid.png)
{ .img-fluid .mb-4 }

You can see that the **total_rooms**, **total_bedrooms**, **population** and **household** columns have outliers. These are apartment blocks with a very large number of occupants and rooms, and we'll have to deal with them.

#### Summary

Visualization is one of the most important sanity checks in machine learning.
It helps you spot issues, guide preprocessing, and understand how features behave—even before you build a model.

By using ScottPlot and agents together, you're learning how to both automate and supervise the exploration process.

{{< /encrypt >}}