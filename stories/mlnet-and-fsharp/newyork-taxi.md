# The New York TLC Dataset

The New York City Taxi and Limousine Commission (TLC), created in 1971, is the agency responsible for licensing and regulating New York City's yellow taxi cabs, for-hire vehicles, commuter vans, and paratransit vehicles. Over 200,000 TLC licensed drivers complete approximately 1,000,000 trips each day. To operate for hire, drivers must first undergo a background check, have a safe driving record, and complete 24 hours of driver training.

In partnership with the New York City Department of Information Technology and Telecommunications, TLC has published millions of trip records from both yellow and green taxis. The taxi trip records include fields capturing pick-up and drop-off dates/times, pick-up and drop-off locations, trip distances, itemized fares, rate types, payment types, and driver-reported passenger counts.

![New York TLC Dataset](../img/data.jpg)
{ .img-fluid .pb-4 }

In this assignment you're going to use the TLC data files to build a model that can predict the fare of any taxi trip in the New York city area.

# Get The Data

Let's start by downloading the New York TLC dataset. 

{{< encrypt >}}

Grab the file from here: [Yellow Taxi Trip Records From December 2018](https://csvbase.com/mdfarragher/Taxi-Trips).

Download the file and save it as **Taxi-Trips.csv**.

This is a truncated dataset with 9,998 records. The original dataset for December 2018 has over 8 million records and is close to 1 GB in size. In this lab, we'll use the smaller dataset to quickly set up our app and experiment with different machine learning pipelines. Later, we'll download the full dataset and test our app on all 8 million trips.

There are a lot of interesting columns in this dataset, for example:

- **tpep_pickup_datetime**: The pickup date and time
- **tpep_dropoff_datetime**: The dropoff date and time
- **passenger_count**: The number of passengers
- **trip_distance**: The trip distance
- **RatecodeID**: The rate code (standard, JFK, Newark, …)
- **payment_type**: The payment type (credit card, cash, …)
- **fare_amount**: The fare amount
- **total_amount**: The total amount (fare plus tip, tolls, tax, etc.)

Let's get started.

#### Set Up The Project

Now open your terminal and navigate to the folder where you want to create the project (e.g., **~/Documents**), and run:

```bash
dotnet new console -lang F# -o TaxiFarePrediction
cd TaxiFarePrediction
```

This creates a new F# console application with:

- **Program.fs** – your main program file
- **TaxiFarePrediction.fsproj** – your project file

Then move the Taxi-Trips.csv file into this folder.

Now run the following command to install the Microsoft.ML machine learning library:

```bash
dotnet add package Microsoft.ML
```

The F# syntax differs from C# in several key ways:
- F# uses immutable values by default with `let` bindings
- F# uses record types instead of classes for simple data structures
- F# uses pipeline operators (`|>`) for function composition
- F# infers types automatically in most cases
{ .tip }

Next, we're going to analyze the dataset and come up with a feature engineering plan.

{{< /encrypt >}}

# Analyze The Data

We’ll begin by analyzing the New York TLC dataset and come up with a plan for feature engineering. Our goal is to map out all required data transformation steps in advance to make later machine learning training as successful as possible.

{{< encrypt >}}

#### Manually Explore the Data

Let’s start by exploring the dataset manually.

Open **Taxi-Trips.csv** in Visual Studio Code, and start looking for patterns, issues, and feature characteristics.

What to look out for:

-    Are there any missing values, zeros, or inconsistent rows?
-    Are the values in each column within a reasonable range?
-    Can you spot any extremely large or very small values?
-    What’s the distribution of values in columns like passenger_count or trip_distance?
-    Are tpep_pickup_datetime and tpep_dropoff_datetime useful as-is, or will they need transformation?

Write down 3 insights from your analysis.
{.homework}

#### Ask Copilot To Analyze The Dataset

You can also ask Copilot to analyze the CSV data and determine feature engineering steps. You should never blindly trust AI advice, but it can be insightful to run an AI scan after you've done your own analysis of the data, and compare Copilot's feedback to your own conclusions. 

Make sure the CSV file is still open in Visual Studio Code. Then expand the Copilot panel on the right-hand side of the screen, and enter the following prompt:

"You are a machine learning expert. Analyze this CSV file for use in a regression model that predicts total_amount. What problems might the dataset have? What preprocessing steps would you suggest?"
{.prompt}

You can either paste in the column names and 5–10 sample rows, or upload the CSV file directly (if your agent supports file uploads).

![Analyze a dataset with an AI agent](../img/analyze.jpg)
{.img-fluid .mb-4}

#### What Might The Agent Suggest?

The agent may recommend steps like:

-    Normalizing data columns
-    Handling extreme outliers
-    Remove invalid rows, for example with passenger_count = 0 or trip_distance = 0
-    One-hot encoding of categorical features (like RatecodeID and payment_type)
-    Drop features that are tightly correlated with total_amount
-    Converting the pickup and dropoff times to a new trip duration feature
-    Adding new features like pickup_hour, pickup_day_of_week and pickup_weekend_flag

Write down 3 insights from the agent’s analysis.
{.homework}

Next, we'll generate a couple of histograms to see if we can find outliers in any of the data columns. 

{{< /encrypt >}}

# Plot A Histogram Of Every Feature

Before you build a machine learning model, it’s important to understand your data visually. Just looking at the numbers, like you did in the previous lesson, may not be enough. A good chart can clearly reveal patterns in the dataset.

In this section, you’ll going to generate code to plot a histogram for every (interesting) feature in the dataset.

Let's get started.

{{< encrypt >}}

#### Install ScottPlot

First, let's install the ScottPlot NuGet package. In your terminal (inside the TaxiFarePrediction folder), install ScottPlot like this:

```bash
dotnet add package ScottPlot
```

Then open Visual Studio Code in the current folder, like this:

```bash
code .
```

Open the Program.fs file and remove all existing content, replace the content with this:

```fsharp
open ScottPlot
open Microsoft.ML
open System
```

#### Create a Histogram of passenger_count

Now let's ask Copilot to write the code for us.

At the bottom of the Copilot panel in Visual Studio Code, make sure the AI mode is set to 'Agent'. Then select your favorite model (I used GPT 4o while preparing this lab).

Enter the following prompt:

"Write F# code using ScottPlot to generate a histogram of the passenger_count column from the Taxi-Trips.csv file. Make sure you load the file using a Microsoft.ML pipeline."
{ .prompt }

And let Copilot write the code for you.

You should see the following data loading code in your project:

```fsharp
// Create MLContext
let mlContext = MLContext()

// Load data
let data = 
    mlContext.Data.LoadFromTextFile<TaxiTrip>(
        path = "Taxi-Trips.csv",
        hasHeader = true,
        separatorChar = ',')

// Extract passenger_count column
let passengerCounts = 
    mlContext.Data.CreateEnumerable<TaxiTrip>(data, reuseRowObject = false)
    |> Seq.map (fun trip -> trip.PassengerCount)
    |> Array.ofSeq
```

This code uses `LoadFromTextFile` to load the CSV file into a data view, which can be used for later machine learning training and evaluation. The code uses a helper type `TaxiTrip` which represents a single row in the dataset.

The code then uses `CreateEnumerable` to convert the loaded data into a sequence of `TaxiTrip` instances, and F# pipeline operators to convert that to a `float[]` containing only the PassengerCount values.

This implementation is by the book, and exactly what we want to see in auto-generated machine learning code that uses Microsoft.ML with F#.
{ .tip }

This is what the TaxiTrip class looks like:

```fsharp
open Microsoft.ML.Data

[<CLIMutable>]
type TaxiTrip = {
    [<LoadColumn(4)>]
    PassengerCount: float32
}
```

The passenger_count column in the dataset is implemented as a record field, with the correct data type, and annotated with a `LoadColumn` attribute that specifies the corresponding CSV column index, in this case index number 4.

And this is the plotting code you will probably end up with:

```fsharp
// Create histogram data
let histogram = ScottPlot.Statistics.Histogram.WithBinCount(20, 
    passengerCounts |> Array.map float)

// Generate histogram using ScottPlot
let plt = Plot()
plt.Add.Bars(histogram.Bins, histogram.Counts) |> ignore
plt.Title("Passenger Count Histogram")
plt.XLabel("Passenger Count")
plt.YLabel("Frequency")

// Save the plot
plt.SavePng("PassengerCountHistogram.png", 600, 400)
```

Let's look at the histogram:

![Histogram Of PassengerCount](../img/passengercount-histogram.png)
{ .img-fluid .mb-4 }

This looks plausible. Most taxi trips have only a single passenger, and the second most popular trip is with two passengers. And there are no trips with more than six passengers.

But look at the bar at zero, apparently some trips have zero passengers? You'll need to decide how you're going to deal with these trips. You could clip the trips by setting the count to 1, or you could trim the dataset by removing these trips from the training data.

Write down the data transformation steps you are going to use to process the PassengerCount column.
{ .homework }

#### Create a Histogram of Every Feature

Now let's modify the code to generate histograms for all the columns in the dataset. First, we're going to make sure that we load every column in the dataset. Enter the following prompt:

"Modify the code so that it loads every column in the dataset and populates each column as a field in the TaxiTrip record type."
{ .prompt }

The agent will extend the `TaxiTrip` class with new properties for each dataset column:

```fsharp
open System

[<CLIMutable>]
type TaxiTrip = {
    [<LoadColumn(0)>] RowID: int
    [<LoadColumn(1)>] VendorID: int
    [<LoadColumn(2)>] PickupDateTime: DateTime
    [<LoadColumn(3)>] DropoffDateTime: DateTime
    [<LoadColumn(4)>] PassengerCount: int
    [<LoadColumn(5)>] TripDistance: float32
    [<LoadColumn(6)>] RatecodeID: int
    [<LoadColumn(7)>] StoreAndFwdFlag: string
    [<LoadColumn(8)>] PULocationID: int
    [<LoadColumn(9)>] DOLocationID: int
    [<LoadColumn(10)>] PaymentType: int
    [<LoadColumn(11)>] FareAmount: float32
    [<LoadColumn(12)>] Extra: float32
    [<LoadColumn(13)>] MtaTax: float32
    [<LoadColumn(14)>] TipAmount: float32
    [<LoadColumn(15)>] TollsAmount: float32
    [<LoadColumn(16)>] ImprovementSurcharge: float32
    [<LoadColumn(17)>] TotalAmount: float32
}
```

Now we're going to use the same generic and reusable code scaffold we used for the Pearson correlation matrix in the previous lab module. We will use reflection to get the list of property names, and then use a generic helper method to create a histogram for that property.

Add the following code right after the data loading code, but before the plotting code:

```fsharp
open System.Reflection

// get column names, skip row id and non-numeric columns
let columnNames = 
    typeof<TaxiTrip>.GetProperties()
    |> Array.filter (fun p -> p.Name <> "RowID" && 
                             (p.PropertyType = typeof<float32> || p.PropertyType = typeof<int>))
    |> Array.map (fun p -> p.Name)

// set up a 4x4 grid of plots
let grid = Multiplot()
grid.RemovePlot(grid.GetPlot(0)) // remove default plot
grid.Layout <- MultiplotLayouts.Grid(columns = 4, rows = 4)

// generate histograms
for columnName in columnNames do
    let plot = PlotHistogram<TaxiTrip>(taxiTrips, columnName)
    grid.AddPlot(plot) |> ignore

// save the grid
grid.SavePng("histograms.png", 1900, 1280)
```

This won't compile, because we don't have the `taxiTrips` variable yet, and the `PlotHistogram` method doesn't exist yet. So let's fix those issues one by one. Enter the following prompt:

"Change the code that calls CreateEnumerable so that it produces a list of TaxiTrip instances with each field populated. Put the list in a variable called taxiTrips."
{ .prompt }

That should fix the data loading code and give us a `taxiTrips` variable which is a populated `List<TaxiTrip>`.

Now let's add the `PlotHistogram` method by hand. We already have working histogram code, we just need to make it reusable by adding a little bit of reflection. You'll want something like this:

```fsharp
let PlotHistogram<'T> (data: 'T list) (columnName: string) =
    // get all column values as doubles
    let prop = typeof<'T>.GetProperty(columnName)
    let values = data |> List.map (fun t -> Convert.ToDouble(prop.GetValue(t))) |> Array.ofList

    // Create histogram data
    let histogram = ScottPlot.Statistics.Histogram.WithBinCount(50, values)

    // Generate histogram using ScottPlot
    let plt = Plot()
    plt.Add.Bars(histogram.Bins, histogram.Counts) |> ignore
    plt.XLabel(columnName)
    plt.YLabel("Frequency")

    plt
```

This function uses reflection to create a `double[]` array of the dataset column specified by `columnName`, and then creates a histogram from this data and returns it as a new plot. The code in the main program assembles these plots into a nice 4x4 grid. 

When you run the app, you should get something like this:

![Histogram Of All Columns](../img/all-histograms.png)
{ .img-fluid .mb-4 }

We have already covered the PassengerCount histogram with the weird zero-passenger trips, but check out the other plots. There are lots of outliers in this dataset:

- **TripDistance** has outliers for trips > 15 miles
- **FareAmount** has outliers for fares > 100 dollars
- There are trips where **FareAmount** and **TotalAmount** are negative
- **TipAmount** has outliers for tips > 15 dollars

At the very least, we'll have to remove (or clip) all trips with zero passengers or a negative fare amount. 

Write down the data transformation steps you are going to use to process the dataset, based on what you've just learned from these histograms.
{ .homework }

#### Create a Utility Class

We will reuse this histogram code in later lessons, so this is a good moment to preserve your work.

In Visual Studio Code, select the code that sets up the 4x4 grid of plots, calls the `PlotHistogram` method and saves the grid. Then press CTRL+I to launch the in-line AI prompt window, and type the following prompt:

"Move this code to a utility function called PlotAllHistograms, with arguments for the list of taxi trips, the list of column names, and the number of rows and columns in the plot grid"
{ .prompt }

And now that we have these two methods, we can move them both to a utility class. Select all source code lines for `PlotAllHistograms` and `PlotHistogram`, press CTRL+I again and enter the following inline prompt:

"Move all of this code to a separate utility module called HistogramUtils."
{ .prompt }

This will produce a new module file called `HistogramUtils`, with all of the code for creating and plotting the grid of histograms for any given feature. We can now use this module in other projects.

And if you want to clean up your code and make it as side-effect-free as possible, you can edit `PlotAllHistograms` and have it return the `Multiplot` grid instance. You can then save the grid in the main program class instead. Your main calling code will then look like this:

```fsharp
// calculate the histogram grid
let grid = HistogramUtils.PlotAllHistograms<TaxiTrip>(taxiTrips, columnNames, columns = 5, rows = 4)

// save the grid
grid.SavePng("histograms.png", 1900, 1280)
```

Perfect!

If you get stuck or want to save some time, feel free to download my completed HistogramUtils class from Codeberg and use it in your own project:

https://codeberg.org/mdft/ml-mlnet-csharp/src/branch/main/TaxiFarePrediction/HistogramUtils.cs

{{< /encrypt >}}

# Calculate The Trip Duration

In the New York TLC dataset, we have a set of columns that at first glance appear to be highly correlated: 

- **tpep_pickup_datetime**
- **tpep_dropoff_datetime**
- **trip_distance**

We would normally expect long-duration taxi trips to cover a large distance and short trips to only cover a small distance. So it's reasonable to assume that trip duration and trip distance are strongly positively correlated. 

{{< encrypt >}}

The confusion matrix will tell us how strong the correlation is between trip duration and trip distance, but for that, we first need to calculate the duration. Fortunately we have the trip pickup and dropoff date and time, so this should be easy.

#### Calculate the Trip Duration

Open the Copilot panel and enter the following prompt:

"Create a machine learning pipeline with a custom mapping that adds a new field called TripDuration to the TaxiTrip record type. The duration is the timespan between PickupDateTime and DropoffDateTime in minutes."
{ .prompt }

You may have to prod your AI agent a few times to generate the correct code. We want to see a new machine learning pipeline with the following custom mapping:

```fsharp
// Define a custom mapping to calculate TripDuration
let pipeline = 
    mlContext.Transforms.CustomMapping<TaxiTrip, TaxiTripWithDuration>(
        Action<TaxiTrip, TaxiTripWithDuration>(fun input output ->
            output.inherited <- input
            output.TripDuration <- float32 (input.DropoffDateTime - input.PickupDateTime).TotalMinutes
        ),
        contractName = null)

// Apply the transformation
let transformedData = pipeline.Fit(data).Transform(data)

// Extract all TaxiTrip instances with properties populated
let taxiTrips = 
    mlContext.Data.CreateEnumerable<TaxiTripWithDuration>(transformedData, reuseRowObject = false)
    |> List.ofSeq
```

Note the last line of the mapping which calculates the trip duration in minutes. Then a call to `Fit` runs the machine learning pipeline and produces `transformedData` (a dataview). And finally, a call to `CreateEnumerable` converts the dataview to a list of `TaxiTripWithDuration` objects.  

Note that we are not registering an assembly for this mapping, so we won't be able to save and load the weights of the fully trained machine learning model. Feel free to add this code yourself if you want maximum flexibility. 
{ .tip }

The `TaxiTripWithDuration` type looks like this:

```fsharp
[<CLIMutable>]
type TaxiTripWithDuration = {
    inherited: TaxiTrip
    TripDuration: float32
}
```

With the mapping set up, it's now very easy to calculate a histogram of the new **TripDuration** column. Just add the following code to your main program:

```fsharp
// Plot and save histogram of trip duration
let plot = HistogramUtils.PlotHistogram<TaxiTripWithDuration>(taxiTrips, "TripDuration")
plot.SavePng("tripduration-histogram.png", 600, 400)
```

When you run this code, you'll get the following histogram:

![Histogram Of Trip Duration](../img/tripduration-histogram.png)
{ .img-fluid .mb-4 }

And look at that! The histogram has a ton of outliers beyond trip durations longer than 60 minutes. We should definitely consider filtering them out to improve the quality of the fare predictions. 

In fact, let's do that right now.

#### Remove Taxi Trips Longer Than 60 Minutes

We're going to add a filter transformation that removes all taxi trips longer than 60 minutes. Open the Copilot panel and enter the following prompt:

"Use the FilterRowsByColumn method to remove all taxi trips longer than 60 minutes."
{ .prompt }

That should give you the following code (the second line is new):

```fsharp
// Apply the transformation
let transformedData = pipeline.Fit(data).Transform(data)

// Filter out taxi trips longer than 60 minutes
let filteredData = mlContext.Data.FilterRowsByColumn(transformedData, "TripDuration", upperBound = 60.0)

// Extract all TaxiTrip instances with properties populated
let taxiTrips = 
    mlContext.Data.CreateEnumerable<TaxiTripWithDuration>(filteredData, reuseRowObject = false)
    |> List.ofSeq
```

First, a call to `Fit` runs the pipeline and calculates the trip duration. Then the `FilterRowsByColumn` function filters the dataview by removing all outliers, and finally the `CreateEnumerable` function produces the list of taxi trips. 

When you run the new code, you'll get the following histogram:

![Histogram Of Trip Duration](../img/tripduration-histogram-trim.png)
{ .img-fluid .mb-4 }

Much better! This is a dataset column we can confidently train a machine learning model on. 

Now, we are finally ready to calculate the confusion matrix. 

{{< /encrypt >}}

# Plot The Pearson Correlation Matrix

It's very easy to calculate and plot the correlation matrix for the New York TLC dataset, because we can use the **CorrelationUtils** helper class from the previous lab. The class is completely reusable and will work on any dataset.

Let's see if our AI agent is smart enough to import code from another project.

{{< encrypt >}}

#### Import The CorrelationUtils Helper Class

For the next prompt, you'll need the raw url of the CorrelationUtils class you created in the previous lab. We will ask the agent to import the class into our current project.

I pushed the class to a repository on Codeberg, so my url is: https://codeberg.org/mdft/ml-mlnet-csharp/raw/branch/main/CaliforniaHousing/CorrelationUtils.cs. Here is the prompt I used:

"Copy the entire CorrelationUtils module from this repository: codeberg.org/mdft/ml-mlnet-fsharp/raw/branch/main/CaliforniaHousing/CorrelationUtils.fs. Add the module to this project."
{ .prompt }

You can see that I'm referring the agent to my CorrelationUtils class stored on Codeberg. You will have to substitute the url with your own (or you can use my class, that's fine too).

The module needs the BetterConsoleTables and MathNet.Numerics packages, so let's install them too:

"Add the NuGet packages BetterConsoleTables and MathNet.Numerics"
{ .prompt }

That should fix any Intellisense errors in your code. Now we can simply add calls to `CalculateCorrelationMatrix` and `PlotCorrelationMatrix`. 

#### Calculate and Plot the Correlation Matrix

In your main program class, locate the following code snippet:

```fsharp
// get column names, skip row id and non-numeric columns
let columnNames = 
    typeof<TaxiTripWithDuration>.GetProperties()
    |> Array.filter (fun p -> p.Name <> "RowID" && 
                             (p.PropertyType = typeof<float32> || p.PropertyType = typeof<int>))
    |> Array.map (fun p -> p.Name)
```

And then put this code right underneath it:

```fsharp
// Calculate the correlation matrix
let matrix = CorrelationUtils.CalculateCorrelationMatrix<TaxiTripWithDuration>(taxiTrips, columnNames)

// plot correlation matrix
let plot = CorrelationUtils.PlotCorrelationMatrix(matrix, columnNames)

// Save the plot to a file
plot.SavePng("correlation-heatmap.png", 900, 800)
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

# Plot The Scatterplot Matrix

In data science, there is a visualisation type called the scatterplot matrix which shows a the relationship of every dataset column to every other column. The visualisation can be used to quickly determine of there is a linear or semi-linear relationship between a feature and the label. 

In the previous lesson, we discovered that **FareAmount** is the best candidate for the label and that **TripDuration**, **TripDistance** and **RateCodeID** are strongly correlated with the label.

In this lesson, we're going to generate a scatterplot matrix that shows the relationhip between each of these four dataset columns. 

{{< encrypt >}}

#### Create a Scatterplot Matrix

We'll start by setting up our top-level code just like with the Pearson correlation matrix. Add the following code to the main program method:

```fsharp
// column names to plot
let scatterPlotColumns = [| "TripDuration"; "TripDistance"; "RatecodeID"; "FareAmount" |]

// plot scatterplot matrix
let smplot = PlotScatterplotMatrix<TaxiTripWithDuration>(taxiTrips, scatterPlotColumns)

// Save the plot to a file
smplot.SavePng("scatterplot-matrix.png", 1900, 1200)
```

The `PlotScatterplotMatrix` method is going to assemble a `Multiplot` using individual scatter plots, just like we did with the histogram grid. So we can simply copy and paste code from the `HistogramUtils` class and tweak it a little, like this:

```fsharp
let PlotScatterplotMatrix<'T> (data: 'T list) (columnNames: string[]) =
    let grid = Multiplot()
    grid.RemovePlot(grid.GetPlot(0)) // remove default plot
    let size = columnNames.Length
    grid.Layout <- MultiplotLayouts.Grid(columns = size, rows = size)

    for rowName in columnNames do
        for colName in columnNames do
            let plot = PlotScatter<'T>(data, colName, rowName)
            grid.AddPlot(plot) |> ignore

    grid
```

This will loop through the column names and create a grid of scatterplots with each plot showing a unique pair of columns. 

Now we just need to implement the `PlotScatter` method. But we have this method already, because we used it in the California Housing lab to plot a graph of **MedianIncome** versus **MedianHouseValue**. 

If you don't have that code anymore, prompt your AI agent to recreate the code or use my example for reference:

```fsharp
let PlotScatter<'T> (data: 'T list) (colName: string) (rowName: string) =
    // get column values
    let colProp = typeof<'T>.GetProperty(colName)
    let colValues = data |> List.map (fun h -> Convert.ToDouble(colProp.GetValue(h))) |> Array.ofList

    // get row values
    let rowProp = typeof<'T>.GetProperty(rowName)
    let rowValues = data |> List.map (fun h -> Convert.ToDouble(rowProp.GetValue(h))) |> Array.ofList

    // generate scatterplot
    let plt = Plot()
    plt.Add.ScatterPoints(colValues, rowValues) |> ignore
    plt.XLabel(colName)
    plt.YLabel(rowName)

    plt
```

And that should get you this scatterplot matrix:

![Scatterplot Matrix](../img/scatterplot-matrix.png)
{ .img-fluid .mb-4 }

If you look at the two plots from the left in the bottom row, you can see that there is indeed a linear relationship between **FareAmount**, **TripDistance** and **TripDuration**. But the outlier, a trip with a fare of $350, distorts the graph and makes the relationship difficult to see. 

So let's get rid of the outlier. And we can use the same code to eliminate all trips with negative fares as well.

#### Filter The FareAmount

Filtering data in ML.NET is very easy, all you need is to call the `FilterRowsByColumn` method to filter a dataview by a given column. In fact, you've done that already in your code when you filtered the trip duration to anything up to 60 minutes. 

Locate the line in your code that filters by **TripDuration**:

```csharp
// Filter out taxi trips longer than 60 minutes
var filteredData = mlContext.Data.FilterRowsByColumn(transformedData, "TripDuration", upperBound: 60);
```

And simply add this:

```fsharp
// Keep fares between $0 and $100
let filteredData2 = mlContext.Data.FilterRowsByColumn(filteredData, "FareAmount", lowerBound = 0.0, upperBound = 100.0)
```

And with that, your scatterplot matrix should now look like this:

![Scatterplot Matrix](../img/scatterplot-matrix-2.png)
{ .img-fluid .mb-4 }

There are three linear relationships clearly visible in the matrix:

- **FareAmount** by **TripDuration**
- **FareAmount** by **TripDistance**
- **TripDuration** by **TripDistance**

Also interesting is the statistical artefact at a fare of $50, visible in the graphs as a horizontal or vertical line. It means taxi drivers often charge $50 for a trip, even though by distance or duration the fare should actually be different (usually lower). 

#### Create a Utility Class

We will reuse this scatterplot code in later lessons, so this is a good moment to preserve your work.

In Visual Studio Code, select the code that declares the `PlotScatterplotMatrix` and `PlotScatter` methods. Then press CTRL+I to launch the in-line AI prompt window, and type the following prompt:

"Move all of this code to a separate utility module called ScatterUtils."
{ .prompt }

This will produce a new module file called `ScatterUtils`, with all of the code for creating and plotting the scatterplot matrix. We can now use this module in other projects.

If you get stuck or want to save some time, feel free to download my completed ScatterUtils class from Codeberg and use it in your own project:

https://codeberg.org/mdft/ml-mlnet-csharp/src/branch/main/TaxiFarePrediction/ScatterUtils.cs


#### Summary

We have used the Pearson correlation matrix to identify the features most strongly correlated with the label, and we generated a scatterplot matrix to verify that the relationships between these features and the label are indeed linear (with some noise added).

We're now ready to implement the data transformations and build the machine learning pipeline. 

{{< /encrypt >}}

# Design And Build The Transformation Pipeline

Now let's start designing the ML.NET data transformation pipeline. This is the sequence of feature engineering steps that will transform the dataset into something suitable for a machine learning algorithm to train on.

After completing the previous lessons, you should have a pretty good idea which feature engineering steps are needed to get this dataset ready for machine learning training.

You're already performing these transformations:

{{< encrypt >}}

- Add a new column with the trip duration
- Remove trips with a duration > 60 minutes
- Remove trips with a negative fare or > 100 dollars

Here are some additional steps you could consider:

- Normalize features
- Remove trips with a distance > 15 miles
- Remove trips with 0 passengers
- Remove trips with tips > 15 dollars
- One-hot encode rate code ID
- One-hot encode payment type

And the correlation matrix showed that the columns **VendorID**, **PassengerCount**, **PULocationID** and **DOLocationID** are very weakly correlated with the label, so you could consider leaving them out of the training data.

Which steps will you choose?

Write down all feature engineering steps you want to perform on the New York TLC dataset, in order.
{ .homework }

#### Implement The Transformation Pipeline

Now let's ask Copilot to implement our chosen data transformation steps with an ML.NET machine learning pipeline. Enter the following prompt in the Copilot panel:

"Implement the following data transformations by building a machine learning pipeline in F#:<br>- [your first transformation step]<br>- [your second transformation step]<br>- ..."
{ .prompt }

You should now have a nice data transformation pipeline that prepares your dataset for machine learning training. Let's take a look at the code.

#### Filter outliers

If you decided to remove outliers, your code should look like this:

```fsharp
// Filter outliers
let mutable filteredData = mlContext.Data.FilterRowsByColumn(transformedData, "TripDuration", upperBound = 60.0f)
filteredData <- mlContext.Data.FilterRowsByColumn(filteredData, "FareAmount", lowerBound = 0.01f, upperBound = 100.0f)
filteredData <- mlContext.Data.FilterRowsByColumn(filteredData, "TripDistance", upperBound = 15.0f)
filteredData <- mlContext.Data.FilterRowsByColumn(filteredData, "TipAmount", upperBound = 15.0f)
filteredData <- mlContext.Data.FilterByCustomPredicate<TaxiTripWithDuration>(filteredData, 
    Func<TaxiTripWithDuration, bool>(fun row -> row.inherited.PassengerCount >= 1))
```

This code uses `FilterRowsByColumn` to filter all columns of type `float`, and `FilterByCustomPredicate` to filter **PassengerCount** to exclude trips with zero passengers.

### Normalize Features

If you decided to normalize any features in the dataset, it will look like this:

```fsharp
// Build ML pipeline with data transformations
let mlPipeline =
    mlContext.Transforms.Concatenate(
        "NumericFeatures",
        [| "TripDistance"; "TripDuration" |])
    // Normalize numeric features
    .Append(mlContext.Transforms.NormalizeMinMax(
        outputColumnName = "NormalizedFeatures",
        inputColumnName = "NumericFeatures"))
```

This code uses `Concatenate` to combine all numeric features (just **TripDistance** and **TripDuration** in my case) into a new combined feature called **NumericFeatures**. The `NormalizeMinMax` method then normalizes these features into a new **NormalizedFeatures** column.

#### One-Hot Encode Categories

If you decided to one-hot encode **RatecodeID** and **PaymentType**, you'll see the following code:

```fsharp
// One-hot encode RatecodeID
.Append(mlContext.Transforms.Categorical.OneHotEncoding(
    outputColumnName = "RatecodeIDEncoded",
    inputColumnName = "RatecodeID"))

// One-hot encode PaymentType
.Append(mlContext.Transforms.Categorical.OneHotEncoding(
    outputColumnName = "PaymentTypeEncoded",
    inputColumnName = "PaymentType"))
    
// Combine all features into a single feature vector
.Append(mlContext.Transforms.Concatenate(
    "Features", 
    "NormalizedFeatures", 
    "RatecodeIDEncoded", 
    "PaymentTypeEncoded"))
```
The `OneHotEncoding` methods perform one-hot encoding on **RatecodeID** and **PaymentType**, and **Concatenate** combines the encoded features and the **NormalizedFeatures** column set up earlier into one new column called **Features**.

These code examples are reference implementations of common data transformations in ML.NET using F#. Compare the output of your AI agent with this code, and correct your agent if needed. Note how F# uses `let` for immutable bindings and `let mutable` when mutation is required.
{ .tip }

And finally, you'll see some code to actually perform the transformations and get access to the transformed data:

```fsharp
// Apply the feature engineering transformations
let model = mlPipeline.Fit(filteredData)
let transformedDataWithFeatures = model.Transform(filteredData)
```

This code calls `Fit` to generate a machine learning model that implements the pipeline. The `Transform` method then uses this model to transform the original dataview into a new transformed dataview with all data transformations applied. 

Now we're ready to add a regression learning algorithm to the machine learning pipeline, so that we can train the model on the data and calculate the regression metrics. 

{{< /encrypt >}}

# Train A Regression Model

We're going to continue with the code we wrote in the previous lab. That C# application set up an ML.NET pipeline to load the New York TLC dataset and clean up the data using several feature engineering techniques.

So all we need to do is append a few command to the end of the pipeline to train and evaluate a regression model on the data.

{{< encrypt >}}

#### Split The Dataset

But first, we need to split the dataset into two partitions: one for training and one for testing. The training partition is typically a randomly shuffled subset of around 80% of all data, with the remaining 20% reserved for testing.

Open the Copilot panel and type the following prompt:

"Split the transformed data into two partitions: 80% for training and 20% for testing."
{ .prompt }

You should get the following code:

```fsharp
// Split the data into training (80%) and testing (20%) datasets
let dataSplit = mlContext.Data.TrainTestSplit(transformedDataWithFeatures, testFraction = 0.2)
let trainingData = dataSplit.TrainSet
let testingData = dataSplit.TestSet
```

The `TrainTestSplit` method splits a dataset into two parts, with the `testFraction` argument specifying how much data ends up in the second part.

#### Train The Model Using SDCA

Now let's add a machine learning algorithm to the pipeline.

"Create a regression pipeline that uses the SDCA algorithm to train a model on the 80% training data partition."
{ .prompt }

You should now see the SDCA algorithm at the end of your pipeline:

```fsharp
// Train model with SDCA algorithm
.Append(mlContext.Regression.Trainers.Sdca(
    labelColumnName = "FareAmount",
    featureColumnName = "Features"))
```

And the `Fit` and `Transform` code should now look like this:

```fsharp
// Train the model on the data
let model = mlPipeline.Fit(trainingData)
```

This code trains the model on the 80% training data partition.

In the next lesson, we'll calculate the prediction evaluation metrics to find out how good the model is at predicting taxi fares.

{{< /encrypt >}}

# Evaluate The Results

Now let's evaluate the quality of the model by comparing the predictions made on the 20% test data to the actual fare amounts, and calculate the regression evaluation metrics.

So imagine you take a taxi trip in New York city an you use your model to predict the fare beforehand. What kind of prediction error would you consider acceptable?

{{< encrypt >}}

Determine the minimum mean absolute error or root mean square error values you deem acceptable. This will be the target your model needs to beat.
{ .homework }

#### Calculate Evaluation Metrics

Enter the following prompt:

"Use the trained model to create predictions for the test set, and then calculate evaluation metrics for these predictions and print them."
{ .prompt }

That should create the following code:

```fsharp
// Use the trained model to create predictions for the test set
Console.WriteLine("Evaluating model on test data...")
let predictions = model.Transform(testingData)

// Display the model evaluation metrics
let metrics = mlContext.Regression.Evaluate(predictions, labelColumnName = "FareAmount")
Console.WriteLine()
Console.WriteLine("**** Model Metrics ****")
Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:F3}")
Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError:F3}")
Console.WriteLine($"R-Squared: {metrics.RSquared:F3}")
```

This code calls `Transform` to set up predictions for every single taxi trip in the test partition. The `Evaluate` method then compares these predictions to the actual fare amounts and automatically calculates these metrics:

- **RSquared**: this is the coefficient of determination, a common evaluation metric for regression models. It tells you how well your model explains the variance in the data, or how good the predictions are compared to simply predicting the mean.
- **RootMeanSquaredError**: this is the root mean squared error or RMSE value. It’s the go-to metric in the field of machine learning to evaluate models and rate their accuracy. RMSE represents the length of a vector in n-dimensional space, made up of the error in each individual prediction.
- **MeanSquaredError**: this is the mean squared error, or MSE value. Note that RMSE and MSE are related: RMSE is the square root of MSE.
- **MeanAbsoluteError**: this is the mean absolute prediction error.

Note that both RMSE and MAE are expressed in dollars. They can both be interpreted as a kind of 'average error' value, but the RMSE will respond much more strongly to large prediction errors. Therefore, if RMSE > MAE, it means the model struggles with some predictions and generates relatively large errors. 

If you used the same transformations as I did, you should get the following output:

![Regression Model Evaluation](../img/evaluate.jpg)
{ .img-fluid .mb-4 }

Let's analyze my results:

The R-squared value is **0.992**. This means that the model explains approximately 99% of the variance in the fare amount. This is an exceptionally high level of explanatory power, suggesting that the model captures nearly all of the underlying patterns in the data. But this may be a sign of **overfitting**, where the model has simply memorized the entire dataset.

The mean absolute error (MAE) is **$0.425**. Given that NYC taxi fares typically range from around $2.50 to $50 or more for most trips, this level of error is extremely low. This means that on average, the model's predictions deviate from the actual fares by less than fifty cents.

The root mean squared error (RMSE) is **$0.541**. The RMSE penalizes larger errors more heavily than the MAE, and this value suggests that most predictions are very close to the true fare values, with almost no large deviations. 

So how did your model do?

Compare your model with the target you set earlier. Did it make predictions that beat the target? Are you happy with the predictive quality of your model? Can you explain what each regression metric means for the quality of your predictions? 
{ .homework }

#### Conclusion

Being able to generate fare predictions with an average error of only 42 cents is really good. It means that almost every taxy trip fare can be fully explained from its duration, distance covered, rate code and other relevant factors. The machine learning model discovered this pattern during training, and is applying the pattern to make near-perfect predictions for every fare.

Unfortunately we've given ourselves a very easy goal here. The full TLC dataset covers more than 8 million trips, but we are working with a fraction of that data. Our dataset holds 10,000 trips from shortly after midnight, on December 1st 2018. This is a very easy dataset to work with, and we may be looking at a situation where the SDCA algorithm is memorizing each trip. We won't know anything for sure until we run the app again on all 8 million trips.

We'll do that shortly, but first, let's add a prediction engine to the machine learning app to make a few ad-hoc fare predictions.

{{< /encrypt >}}

# Make A Prediction

To wrap up, let’s use the model to make a prediction.

We're going to invent a fake taxi trip in New York City. I'm going to get into a cab at Times Square and take a trip to Washington Square Park. The trip covers 2.3 miles and takes 12 minutes. What's the fare I should expect to pay?

We will ask our AI agent to write code that prompts us for all the properties of a single taxi trip, and then we'll use the machine learning model to predict what the fare amount will be.

{{< encrypt >}}

#### Make A Prediction

Enter the following prompt:

"Add code to prompt the user for all the properties of a single taxi trip, and then use the model to generate a prediction of the fare amount. Ask only for the trip duration, trip distance, rate code ID and payment type."
{ .prompt }

The agent will create a new class `TaxiTripFarePrediction` with a property labelled `Score` to hold the generated prediction:

```fsharp
// Class to hold prediction
[<CLIMutable>]
type TaxiTripFarePrediction = {
    [<ColumnName("Score")>]
    PredictedFareAmount: float32
}
```

And then it will add code like this to make the prediction:

```fsharp
// Create input data
let tripData = { inherited = Unchecked.defaultof<TaxiTrip>; TripDuration = 0.0f }

// Get user input
Console.Write("Trip Duration (minutes): ")
match System.Single.TryParse(Console.ReadLine()) with
| true, tripDuration -> tripData.TripDuration <- tripDuration
| _ -> ()

// ... (similar for other properties)

// Create a prediction engine
let predictionEngine = mlContext.Model.CreatePredictionEngine<TaxiTripWithDuration, TaxiTripFarePrediction>(model)

// Make prediction
let prediction = predictionEngine.Predict(tripData)

// Display prediction
Console.WriteLine($"Predicted Fare Amount: ${prediction.PredictedFareAmount:F2}")
```

The `CreatePredictionEngine` method sets up a prediction engine. The two type arguments are the input data record type and the record type to hold the prediction.

With the prediction engine set up, a call to `Predict` is all you need to make a single prediction. The prediction value is then available in the `PredictedFareAmount` field.

Let's try this for the fake trip I took earlier. Here is the data you need to enter:

- Trip duration = 12 minutes
- Trip distance = 2.3 miles
- Rate code ID = 1 (standard rate)
- Payment type = 1 (credit card)

And this is the output I get:

![Using The Model To Make A Prediction](../img/prediction.jpg)
{ .img-fluid .mb-4 }

I get a predicted fare amount of **$10.70**.

What prediction did you get? Try changing the input data to see how this affects the predicted fare amount. Do the changes in prediction value make sense to you?
{ .homework }

Next, let's load the full dataset of 8 million trips and re-run the app to discover the actual regression metrics and prediction accuracy. 

{{< /encrypt >}}

# Load The Full Dataset

So far, we have been working with a subset of the New York TLC dataset. The subset contains the first 10,000 taxi trips made in the early hours of the morning on December 1st, 2018. This is not a representative subset of all the trips made in December, but it allowed us to quickly design a data transformation pipeline and train a regression model on the data.

Now, let's download the full dataset. 

{{< encrypt >}}

The New York City Taxi and Limousine Commission (TLC) website has a page where you can access all [TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) in Parquet format. Please [download the full dataset for December 2018 with this link](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2018-12.parquet).

This is a file named **yellow_tripdata_2018-12.parquet**. It's about 112 MB in size and holds roughly 8.1 million taxi trip records. Copy this file into your project folder.

#### Load The Parquet File

Now let's alter the app so that we can choose if we want to load the parquet file with all taxi trips or the CSV file with only the first 10,000 trips. Enter the following prompt:

"Add code that asks the user if they want to load the Taxi-Trips.csv file with the first 10,000 trips or the yellow_tripdata_2018-12.parquet file with all 8.1 million trips. Use the Parquet.NET library to load the parquet file into a list of TaxiTrip objects and then use LoadFromEnumerable to convert the list to a dataview for machine learning."
{ .prompt }

This is not an easy refactor because the ML.NET library does not support parquet loading yet, even though there is a NuGet package for it. My agent repeatedly got stuck trying to use a nonexistent `LoadFromParquetFile` method. Instead, you have to use the **Parquet.NET** library to load the file and then convert the list of trips back to a dataview using `LoadFromEnumerable`.

We want to see code like this:

```fsharp
let parquetTrips = 
    ParquetSerializer.DeserializeAsync<TaxiTrip>("yellow_tripdata_2018-12.parquet")
        .GetAwaiter()
        .GetResult()
data <- mlContext.Data.LoadFromEnumerable(taxiTrips)
```

The `ParquetSerializer.DeserializeAsync` method will load the file and produce a list of `TaxiTrip` objects (called with `GetAwaiter` and `GetResults` so that the synchronous calling thread will block while the data is loading), and then `LoadFromEnumerable` will convert the list to a dataview. 

Unfortunately, this code will not work. Here's what happens when you try to run the app:

![Parquet Conversion Error](../img/parquet-error.jpg)
{.img-fluid .mb-4}

The error message refers to the fact that the **VendorID** column in the parquet file has a different data type and nullability than the corresponding property in the `TaxiTrip` class. The schema of the data and the class structure are out of sync.

So let's inspect the schema of the parquet file data to find out what's going on. The [Parquet.NET project](https://github.com/aloneguid/parquet-dotnet) provides a nice little tool called 'Floor' that can visualize a parquet file. Here is the schema of the TLC dataset according to Floor: 

![Parquet File Schema](../img/parquet-schema.jpg)
{.img-fluid .mb-4}

You can see that many fields have different names, everything is nullable (the repetition type is OPTIONAL) and the data types are twice as wide as expected (`double` instead of `float`, and `long` instead of `int`).

What we need is a modified `TaxiTrip` class that takes this new schema into account. Something like this:

```fsharp
[<CLIMutable>]
type ParquetTaxiTrip = {
    VendorID: Nullable<int64>
    tpep_pickup_datetime: Nullable<DateTime>
    tpep_dropoff_datetime: Nullable<DateTime>
    passenger_count: Nullable<float>
    trip_distance: Nullable<float>
    RatecodeID: Nullable<float>
    store_and_fwd_flag: string
    PULocationID: Nullable<int64>
    DOLocationID: Nullable<int64>
    payment_type: Nullable<int64>
    fare_amount: Nullable<float>
    extra: Nullable<float>
    mta_tax: Nullable<float>
    tip_amount: Nullable<float>
    tolls_amount: Nullable<float>
    improvement_surcharge: Nullable<float>
    total_amount: Nullable<float>
    congestion_surcharge: Nullable<int>
    airport_fee: Nullable<int>
}
```

Note that every property is now nullable and that most datatypes are `Int64` or `double`. This class exactly matches the schema of the data in the parquet file. 

To load the data, all we need to do is pass the correct type to the `DeserializeAsync` method, like this:

```fsharp
let parquetTrips = 
    ParquetSerializer.DeserializeAsync<ParquetTaxiTrip>("yellow_tripdata_2018-12.parquet")
        .GetAwaiter()
        .GetResult()
```

This code works and will load all 8.1 million trips. But now we have a new problem: we cannot convert the list of trips to a dataview, because the ML.NET library does not support nullable `double` or `Int64` properties at all. 

We can fix this by manually converting every `ParquetTaxiTrip` to a `TaxiTrip`, like this:

```fsharp
// Convert data to TaxiTrip list
let taxiTrips = 
    parquetTrips 
    |> Seq.map (fun p -> 
        {
            RowID = 0
            VendorID = p.VendorID.GetValueOrDefault(0L) |> int
            PickupDateTime = p.tpep_pickup_datetime.GetValueOrDefault(DateTime.MinValue)
            DropoffDateTime = p.tpep_dropoff_datetime.GetValueOrDefault(DateTime.MinValue)
            PassengerCount = p.passenger_count.GetValueOrDefault(0.0) |> int
            TripDistance = p.trip_distance.GetValueOrDefault(0.0) |> float32
            RatecodeID = p.RatecodeID.GetValueOrDefault(0.0) |> int
            StoreAndFwdFlag = p.store_and_fwd_flag ?? ""
            PULocationID = p.PULocationID.GetValueOrDefault(0L) |> int
            DOLocationID = p.DOLocationID.GetValueOrDefault(0L) |> int
            PaymentType = p.payment_type.GetValueOrDefault(0L) |> int
            FareAmount = p.fare_amount.GetValueOrDefault(0.0) |> float32
            Extra = p.extra.GetValueOrDefault(0.0) |> float32
            MtaTax = p.mta_tax.GetValueOrDefault(0.0) |> float32
            TipAmount = p.tip_amount.GetValueOrDefault(0.0) |> float32
            TollsAmount = p.tolls_amount.GetValueOrDefault(0.0) |> float32
            ImprovementSurcharge = p.improvement_surcharge.GetValueOrDefault(0.0) |> float32
            TotalAmount = p.total_amount.GetValueOrDefault(0.0) |> float32
        })
    |> List.ofSeq

// Convert list to IDataView
data <- mlContext.Data.LoadFromEnumerable(taxiTrips)
```

This is not very efficient code, but it's good enough for now. 

#### Train And Evaluate The Model

At this point, we are done. The app can now load all 8.1 million taxi trips from the parquet file, convert them to a dataview, and then use the existing code to kick off a machine learning training run and report the regression metrics. 

Run your app and have it load the taxi trips from the parquet file. Then wait for model training to finish and note the new regression metrics. What has happened to the predictive quality of the model? 
{ .homework }

Here's what I got:

![Training a Regression Model on all Taxi Trips](../img/evaluate-parquet.jpg)
{.img-fluid .mb-4}

The R-squared value is **0.872** which means that the model explains 87% of the variance in the fare amount. This is quite a bit lower than the previous result of 0.992, but still a good result. It suggests that the previous model trained on 10,000 trips was indeed overfitting, and we are now looking at a more realistic result. 

The mean absolute error (MAE) is **$2.818**. It increased sevenfold compared to the previous MAE of 0.425. In this dataset we have a lot more variety: long trips, airport trips, surcharges, and possibly mislabeled fares. This naturally increases prediction difficulty. It’s likely the model is not handling outliers, long-distance trips, or rare cases well.

The root mean squared error (RMSE) is **$4.565**. The fact that the RMSE is twice as large as the MAE indicates that there are outliers where the model makes large errors in its prediction. This suggests that we may have to adjust our data filters for the new dataset. 

#### Conclusion 

All in all, this was quite a step backwards. And this is not surprising. We designed the data transformation pipeline specifically for a dataset of only 10,000 trips. Even worse, all of these trips were early in the morning on December 1st, 2018. We made many assumptions based off the histograms, correlation matrix and scatterplot grid and baked those assumptions into the design of the machine learning pipeline.

But now we have a dataset that covers the full month of December. The patterns in this dataset might be completely different, and we'll have to revisit all of our previous assumptions to check if they are still valid for this much larger dataset. 

Be very careful when you design a data transformation pipeline for a partial subset of a dataset. The histograms and the correlation matrix can change dramatically when loading the full dataset, and you may have to revist all your prior assumptions about the data.
{ .tip }

In the next lesson, we'll quickly regenerate the histogram grid, the Pearson correlation matrix and the scatterplot grid to see if the data transformation pipeline need to be changed. 

{{< /encrypt >}}

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
Console.WriteLine("Generating histograms...")
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

```csharp
// plot scatterplot matrix
var scatterPlotColumns = new string[] { "TripDuration", "TripDistance", "RatecodeID", "FareAmount" };
var smplot = ScatterUtils.PlotScatterplotMatrix<TaxiTripWithDuration>(taxiTrips, scatterPlotColumns);

// Save the plot to a file
smplot.SavePng("scatterplot-matrix.png", 1900, 1200);
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

# Improve Your Results

There are many factors that influence the quality of your model predictions, including how you process the dataset, which regression algorithm you pick, and how you configure the training hyperparameters.

Here are a couple of things you could do to improve your model:

{{< encrypt >}}

- Split the pickup datetime into separate hour, day of week and weekend columns
- Analyze the trip distance and trip duration columns and calculate a new 'on-time' column
- Filter on one specific ratecode ID and determine the prediction accuracy per ratecode
- Bin and one-hot encode trip distance to create a new column called 'long-distance'
- Bin and one-hot encode trip duration to create a new column called 'long-duration'
- Try a different regression learning algorithm.
- Use different hyperparameter values for your learning algorithm.

Experiment with different data processing steps and regression algorithms. Document your best-performing machine learning pipeline for this dataset, and write down the corresponding regression evaluation metrics.
{ .homework }

How close can you make your predictions to the actual fare amounts? 

{{< /encrypt >}}

# Hall Of Fame

Would you like to be famous? You can [submit your best-performing model](mailto:mark@mdfteurope.com) for inclusion in this hall of fame, which lists the best regression evaluation scores for the New York TLC dataset. I've added my own results as a baseline, using the transformations I mentioned in the lab. 

Can you beat my score?

| Rank | Name | Algorithm | Transformations |   MAE   |  RMSE   |
|------|------|-----------|-----------------|---------|---------|
|  1   | Mark | SDCA      | As mentioned in lab | $2.197 | $3.407 |

I will periodically collect new submissions and merge them into the hall of fame. I'll share the list in my courses and on social media. If you make the list, you'll be famous!

# Recap

Congratulations on finishing the lab. Here's what you have learned.

{{< encrypt >}}

You learned how to analyze the **New York TLC dataset**, both automatically by having an AI agent scan the data for you, and manually by inspecting the data by hand. You wrote code to generate a **histogram of every feature** in the dataset, and how to analyze these histograms to identify outliers to filter out.

You learned how to calculate the **trip duration** from the original pickup and dropoff dates and times. You also generated a histogram of the trip duration to identify any outliers to filter. You also calculated the **Pearson correlation matrix** for every feature and label, and used the matrix to identify features that are **strongly correlated** to the label.

You wrote code to generate the **scatterplot matrix**, and used it to learn how the strongly correlated features are related to the label. You also used the matrix to identify outliers and statistical artefacts in the dataset, and set up **data filters** for the New York TLC dataset. 

You built a **machine learning pipeline** and trained and evaluated a regression model on the dataset. Then you analyzed the regression metrics to determine the quality of the predictions. You discovered that the model might be **overfitting**. 

You learned how to load the full dataset as a **parquet file**, and feed the data into the existing dataview in your application. You regenerated the histogram grid, correlation matrix and scatterplot matrix to determine if your assumptions about the data are still valid for the full dataset. You then decided to **adjust your data filters and transformations** or leave them unchanged.

You completed the lab by experimenting with different data processing steps and regression algorithms to find the best-performing model. 

{{< /encrypt >}}

# Conclusion

This concludes the two regression modules in this lab. I hope you enjoyed training models on the California Housing and New York Taxi datasets. You now have hands-on experience building two F# apps that train models on a dataset. 

The New York TLC dataset is a nice example of a large training dataset. We have 8.1 million taxi trips in the month of December alone. If you wanted to train your model on all of 2018, you would have to deal with roughly 100 million rows of data. Very large datasets are common in machine learning. Computer vision models are routinely trained on 10 million images, and contemporary large language models are trained on pretty much the entire Internet!

If you need to produce other kinds of numerical predictions in the future, feel free to just copy and paste the code from the labs. The steps to build a regression pipeline are the same every time, all you need to tweak are the data processing steps, the learning algorithm and the hyperparameters.

I hope that you're starting to realize that machine learning applications are actually very simple. With just a few hundred lines of F# code, you can process a dataset, train a model, evaluate the metrics, and then start generating predictions. F#'s functional programming features make the code more concise and expressive than equivalent C# implementations.

