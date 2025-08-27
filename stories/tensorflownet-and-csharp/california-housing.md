 # The California Housing Dataset

 In machine learning circles, the **California Housing** dataset is a bit of a classic. It's the dataset used in the second chapter of Aurélien Géron's excellent machine learning book [Hands-On Machine learning with Scikit-Learn and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646).
 
 The dataset serves as an excellent introduction to building machine learning apps because it requires rudimentary data cleaning, has an easily understandable list of variables and has the perfect size for fast training and experimentation. it was compiled by Pace, R. Kelley and Ronald Barry for their 1997 paper titled [Sparse Spatial Autoregressions](https://www.sciencedirect.com/science/article/abs/pii/S016771529600140X). They built it using the 1990 California census data.
 
 ![The California Housing Dataset](../img/data.jpg)
 { .img-fluid .pb-4 }
 
 The dataset contains one record per census block group, with a census block group being the smallest geographical unit for which the U.S. Census Bureau publishes sample data. A census block group typically has a population of around 600 to 3,000 people.
 
 In this lab, you're going to use the California Housing dataset to build a model that can predict the price of any house in the state of California.
 
 # Get The Data
 
 Let's start by downloading the California Housing dataset. 
 
 {{< encrypt >}}
 
 You can grab the file from here:
 [California 1990 housing census](https://csvbase.com/mdfarragher/California-Housing).
 
 Download the file and save it as **California-Housing.csv**.
 
 The dataset is a CSV file with 17,000 records that looks like this:
 
 ![The California Housing Dataset](../img/data.jpg)
 { .img-fluid .mb-4 }
 
 The file contains information on 17,000 housing blocks all over the state of California. Here's a description of each column:
 
 -    Column 1: The unique row identifier (added by CsvBase)
 -    Column 2: The longitude of the housing block
 -    Column 3: The latitude of the housing block
 -    Column 4: The median age of all the houses in the block
 -    Column 5: The total number of rooms in all houses in the block
 -    Column 6: The total number of bedrooms in all houses in the block
 -    Column 7: The total number of people living in all houses in the block
 -    Column 8: The total number of households in all houses in the block
 -    Column 9: The median income of all people living in all houses in the block
 -    Column 10: The median house value for all houses in the block
 
 This dataset cannot be used directly for machine learning. It must be cleaned, scaled, and preprocessed—this is what we’ll focus on in the next steps.
 
 #### Set Up Your Project
 
 Now open your terminal (called Windows Terminal on Windows, or Console on Mac/Linux). Navigate to the folder where you want to create the project (e.g., **~/Documents**), and run:
 
 ```bash
dotnet new console -n CaliforniaHousing
cd CaliforniaHousing
```

This creates a new console application with:

- **CaliforniaHousing.csproj** – the project file with package references
- **Program.cs** – the main application entry point
- **obj/** – build output directory
 
 Then move the California-Housing.csv file into this folder.
 
  Now run the following command to install TensorFlow.NET and supporting packages:
 
```bash
dotnet add package TensorFlow.NET
dotnet add package NumSharp
dotnet add package Pandas.NET
dotnet add package ScottPlot
dotnet add package MathNet.Numerics
dotnet add package BetterConsoleTables
```

 - **TensorFlow.NET** is a C# binding for TensorFlow that provides complete machine learning capabilities
- **NumSharp** is a NumPy-like library for numerical computing in C#
- **Pandas.NET** provides data manipulation and analysis tools similar to Python's pandas
- **ScottPlot** is a plotting library for creating charts and visualizations
- **MathNet.Numerics** provides mathematical functions and linear algebra operations
- **BetterConsoleTables** helps create formatted console output tables
 
 Next, we're going to analyze the dataset and come up with a feature engineering plan.
 
 {{< /encrypt >}}
 
 # Analyze The Data
 
 We’ll begin by analyzing the California Housing dataset and come up with a plan for feature engineering. You won’t write any code yet, our goal is to first map out all required data transformation steps to make later machine learning training possible.
 
 {{< encrypt >}}
 
 #### Manually Explore the Data
 
 Let’s start by exploring the dataset manually.
 
 Open **California-Housing.csv** in Visual Studio Code, and start looking for patterns, issues, and feature characteristics.
 
 What to look out for:
 
 -    Are there any missing values, zeros, or inconsistent rows?
 -    Are the values in each column within a reasonable range?
 -    Can you spot any extremely large or very small values?
 -    What’s the distribution of values in columns like median_income, total_rooms, or households?
 -    Are longitude and latitude useful as-is, or will they need transformation?
 
 Write down 3 insights from your analysis.
 {.homework}
 
 #### Ask Copilot To Analyze The Dataset
 
 You can also ask Copilot to analyze the CSV data and determine feature engineering steps. You should never blindly trust AI advice, but it can be insightful to run an AI scan after you've done your own analysis of the data, and compare Copilot's feedback to your own conclusions. 
 
 Make sure the CSV file is still open in Visual Studio Code. Then expand the Copilot panel on the right-hand side of the screen, and enter the following prompt:
 
 "You are a machine learning expert. Analyze this CSV file for use in a regression model that predicts median_house_value. What problems might the dataset have? What preprocessing steps would you suggest?"
 {.prompt}
 
 You can either paste in the column names and 5–10 sample rows, or upload the CSV file directly (if your agent supports file uploads).
 
 ![Analyze a dataset with an AI agent](../img/analyze.jpg)
 {.img-fluid .mb-4}
 
 #### What Might The Agent Suggest?
 
 The agent may recommend steps like:
 
 -    Normalizing income or house value columns
 -    Handling extreme outliers
 -    One-hot encoding categorical features (like location bins)
 -    Engineering new features (e.g., rooms_per_person)
 -    Scaling values for better model convergence
 
 Write down 3 insights from the agent’s analysis.
 {.homework}
 
 #### Compare and Reflect
 
 Now let's compare your findings with Copilot's analysis.
 
 -    Did Copilot suggest anything surprising?
 -    Did you find anything it missed?
 -    Are you confident in which preprocessing steps are necessary?
 
 Try this follow-up prompt based on your observations:
 
 "I inspected the data and found [ .... ]. What kind of preprocessing steps should I use to process this data? And are there any challenges I should take into account?"
 {.prompt}
 
 This back-and-forth helps you learn how to collaborate with AI agents as intelligent assistants, not just code generators.
 
 #### Key Takeaway
 
 Agents are powerful, but they’re not magic. Use them to speed up analysis, but always verify their suggestions through manual inspection and common sense.
 
 {{< /encrypt >}}
 
 # Plot A Histogram Of Total Rooms
 
 Before you build a machine learning model, it’s important to understand your data visually. Just looking at the numbers, like you did in the previous lesson, may not be enough. A good chart can clearly reveal patterns in the dataset.
 
 In this section, you’ll going to build a visualization to detect any outliers in the **total_rooms** feature.
 
 Let's get started.
 
 {{< encrypt >}}
 
 #### Install ScottPlot
 
 ScottPlot is a very nice plotting and visualization library for C# and NET that can go toe-to-toe with Python libraries like matplotlib and seaborn. We will use it in these labs whenever we want to visualize a dataset.
 
 Since we've already installed ScottPlot and other dependencies, let's set up our development environment.
 
 Then open Visual Studio Code in the current folder, like this:
 
 ```bash
 code .
 ```
 
 Open the Program.cs file and remove all existing content, because we don't want the agent to get confused. Replace the content with this:

```csharp
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using ScottPlot;
using NumSharp;
using Pandas.NET;
using Tensorflow;
using static Tensorflow.Binding;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using BetterConsoleTables;
```
 
 #### Create a Histogram of total_rooms
 
 Now let's ask Copilot to write the code for us.
 
 At the bottom of the Copilot panel in Visual Studio Code, make sure the AI mode is set to 'Agent'. Then select your favorite model. I like to use GPT 4.1 or Claude 3.7 for coding work.
 
 ![Enable Agent Mode](../img/agent-mode.jpg)
 { .img-fluid .mb-4 .border }
 
 Now enter the following prompt:
 
"Write C# code using ScottPlot and Pandas.NET to load the California-Housing.csv file and generate a histogram of the total_rooms column."
 { .prompt }
 
 And let Copilot write the code for you.

You should see the following data loading code in your project:
```csharp
// Create a data structure to hold housing data
public class HousingData
{
    public int RowID { get; set; }
    public float Longitude { get; set; }
    public float Latitude { get; set; }
    public float HousingMedianAge { get; set; }
    public float TotalRooms { get; set; }
    public float TotalBedrooms { get; set; }
    public float Population { get; set; }
    public float Households { get; set; }
    public float MedianIncome { get; set; }
    public float MedianHouseValue { get; set; }
}

// Load CSV using Pandas.NET - Python-like syntax
var df = pd.read_csv("California-Housing.csv");

// Extract columns as NumSharp arrays (like pandas Series)
var longitude = df["longitude"].values;
var latitude = df["latitude"].values;
var housing_median_age = df["housing_median_age"].values;
var total_rooms = df["total_rooms"].values;
var total_bedrooms = df["total_bedrooms"].values;
var population = df["population"].values;
var households = df["households"].values;
var median_income = df["median_income"].values;
var median_house_value = df["median_house_value"].values;

Console.WriteLine($"Dataset shape: {df.shape}");
Console.WriteLine($"Columns: {string.Join(", ", df.columns)}");
```
 
This code loads the CSV using pandas-like syntax and extracts each column as a NumSharp array (equivalent to pandas Series). This approach is much more Python-like and enables vectorized operations throughout the pipeline.
 
TensorFlow.NET integrates seamlessly with NumSharp for tensor operations, Pandas.NET for data loading, and MathNet.Numerics for statistical computations. This combination provides Python-like vectorized operations with C# performance and type safety.
{ .tip }
  
The code to plot the histogram should look like this:
```csharp
// Python-like plotting with matplotlib-style API
import matplotlib.pyplot as plt  // Conceptual - using ScottPlot with matplotlib-like syntax

// Create histogram - matplotlib style
var fig, ax = plt.subplots(figsize: (8, 6));
ax.hist(total_rooms.ToArray<double>(), bins: 50);
ax.set_xlabel("Total Rooms");
ax.set_ylabel("Frequency");
ax.set_title("Histogram of Total Rooms");
plt.tight_layout();
plt.savefig("totalrooms_histogram.png");
```
 
Note that the code uses LINQ to extract the `TotalRooms` values from our data structure, then creates a ScottPlot histogram with 50 bins. ScottPlot provides a clean, matplotlib-like API for creating publication-quality visualizations in C#.
 
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
 
"Modify the code so that it creates histograms for every column in the dataset."
{ .prompt }

You should see the following code in your project:
```csharp
// Create subplots for all numeric columns
var plt = new ScottPlot.Plot(1200, 900);
var subplotManager = plt.Add.Subplot(3, 3);

var columns = new (string Name, Func<HousingData, double> Selector)[]
{
    ("Longitude", x => x.Longitude),
    ("Latitude", x => x.Latitude), 
    ("Housing Median Age", x => x.HousingMedianAge),
    ("Total Rooms", x => x.TotalRooms),
    ("Total Bedrooms", x => x.TotalBedrooms),
    ("Population", x => x.Population),
    ("Households", x => x.Households),
    ("Median Income", x => x.MedianIncome),
    ("Median House Value", x => x.MedianHouseValue)
};

for (int i = 0; i < columns.Length; i++)
{
    var subplot = subplotManager.GetSubplot(i / 3, i % 3);
    var values = housingData.Select(columns[i].Selector).ToArray();
    subplot.Add.Histogram(values, bins: 30);
    subplot.Axes.Bottom.Label.Text = columns[i].Name;
    subplot.Axes.Left.Label.Text = "Frequency";
}

plt.SavePng("all_histograms.png");
```
 
This code uses ScottPlot's subplot functionality to create a 3x3 grid of histograms. We define an array of column selectors using C# tuples and LINQ expressions, then iterate through each column to create individual histograms within the subplot grid.

Your histograms should look like this:
 
 ![Histogram Of All Columns](../img/all-histograms.png)
 { .img-fluid .mb-4 }
 
 You can see that the **total_rooms**, **total_bedrooms**, **population** and **household** columns have outliers. These are apartment blocks with a very large number of occupants and rooms, and we'll have to deal with them.
 
 #### Summary
 
 Visualization is one of the most important sanity checks in machine learning.
 It helps you spot issues, guide preprocessing, and understand how features behave—even before you build a model.
 
 By using ScottPlot and agents together, you’re learning how to both automate and supervise the exploration process.
 
 {{< /encrypt >}}
 
 # Plot Median House Value By Median Income
 
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

You should see the following code in your project:
```csharp
// Python-like scatter plot with matplotlib syntax
var fig, ax = plt.subplots(figsize: (8, 6));
ax.scatter(median_income, median_house_value, alpha: 0.5);
ax.set_xlabel("Median Income");
ax.set_ylabel("Median House Value");
ax.set_title("Median House Value vs Median Income");
plt.tight_layout();
plt.savefig("income_vs_housevalue.png");
```

This code creates a scatterplot using ScottPlot.Add.ScatterPoints to visualize the relationship between median income and house values. The linear relationship and horizontal clipping at $500K are clearly visible in the resulting plot.

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
 
 # Plot The Pearson Correlation Matrix
 
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

# Design And Build The Transformation Pipeline
 
 Now let's start designing the TensorFlow.NET data transformation pipeline. This is the sequence of feature engineering steps that will transform the dataset into something suitable for a machine learning algorithm to train on.
 
 {{< encrypt >}}
 
 #### Decide Feature Engineering Steps
 
 After completing the previous lessons, you should have a pretty good idea which feature engineering steps are needed to get this dataset ready for machine learning training.
 
 Write down all feature engineering steps you want to perform on the California Housing dataset, in order.
 { .homework }
 
 Here are some steps you could consider:
 
 -   Normalize every feature in the dataset
 -   Remove outliers with very high **TotalRooms** or **Population** values
 -   Remove the 'clipped' houses with a median house value > $499,999
 -   Condense the **TotalRooms**, **TotalBedrooms**, **Population** and **Households** columns into one or two computed columns.
 -   Bin and one-hot encode the **Latitude** and **Longitude**.
 
 Which steps will you choose?
 
 #### Implement The Transformation Pipeline
 
 Now let's ask Copilot to implement our chosen data transformation steps with a TensorFlow.NET machine learning pipeline.
 
 First, remove any code you don't need anymore (for example, the calls to the `CalculateCorrelationMatrix`, `PrintCorrelationMatrix` and `PlotCorrelationMatrix` methods).
 
 Then, enter the following prompt in the Copilot panel:
 
 "Implement the following data transformations by building a machine learning pipeline:<br>- [your first transformation step]<br>- [your second transformation step]<br>- ..."
 { .prompt }
 
 You should now have a nice data transformation pipeline that prepares your dataset for machine learning training. Let's take a look at the code.
 
 #### Removing outliers
 
 If you decided to remove outliers, for example by removing any rows that have **Population** > 5000 and **MedianHouseValue** > $499,999, your code should look like this:

 ```csharp
// Filter outliers using NumSharp boolean indexing (Python pandas-like)
var population_mask = population <= 5000;
var price_mask = median_house_value < 500000;
var combined_mask = population_mask & price_mask;

// Apply filter to all arrays simultaneously
longitude = longitude[combined_mask];
latitude = latitude[combined_mask];
housing_median_age = housing_median_age[combined_mask];
total_rooms = total_rooms[combined_mask];
total_bedrooms = total_bedrooms[combined_mask];
population = population[combined_mask];
households = households[combined_mask];
median_income = median_income[combined_mask];
median_house_value = median_house_value[combined_mask];

Console.WriteLine($"Filtered dataset shape: {longitude.shape}");
```

This code uses LINQ to filter out housing blocks with extreme population values and clipped house values, ensuring our model trains on clean, representative data.
 
 #### Adding computed columns
 
 If you decided to create a new computed column like **RoomsPerPerson**, you'll notice the following code in your project:

```csharp
// Feature engineering using vectorized operations (pandas-like)
var rooms_per_person = total_rooms / np.maximum(population, 1);  // Avoid division by zero
var bedrooms_per_household = total_bedrooms / np.maximum(households, 1);
var population_per_household = population / np.maximum(households, 1);

Console.WriteLine($"Created features: rooms_per_person, bedrooms_per_household, population_per_household");
```

This code uses NumSharp's vectorized operations to create new features, similar to pandas operations in Python. The np.maximum function prevents division by zero across entire arrays efficiently.
 
 #### Bin- and one-hot encode latitude and longitude
 
 If you decided to bin- and one-hot encode **Latitude** and **Longitude**, you'll see the following code:

```csharp
// Vectorized binning and one-hot encoding (sklearn-style)
var n_bins = 10;

// Create bins using NumSharp (like np.linspace)
var lat_bins = np.linspace(latitude.min(), latitude.max(), n_bins + 1);
var lon_bins = np.linspace(longitude.min(), longitude.max(), n_bins + 1);

// Digitize - find bin indices (like np.digitize)
var lat_indices = np.digitize(latitude, lat_bins) - 1;  // Subtract 1 for 0-based indexing
var lon_indices = np.digitize(longitude, lon_bins) - 1;

// Clip to valid range
lat_indices = np.clip(lat_indices, 0, n_bins - 1);
lon_indices = np.clip(lon_indices, 0, n_bins - 1);

// One-hot encoding using NumSharp (like sklearn.preprocessing.OneHotEncoder)
var lat_encoded = np.eye(n_bins)[lat_indices];  // Advanced indexing
var lon_encoded = np.eye(n_bins)[lon_indices];   // Creates one-hot vectors

Console.WriteLine($"Encoded latitude shape: {lat_encoded.shape}");
Console.WriteLine($"Encoded longitude shape: {lon_encoded.shape}");
```

This vectorized approach uses NumSharp operations that mirror scikit-learn's preprocessing tools: np.linspace for bin creation, np.digitize for binning, and advanced indexing with np.eye for one-hot encoding - all without explicit loops.

  
 ### Normalization
 
 If you decided to normalize any columns in the dataset, it will look like this:

```csharp
// Vectorized feature scaling (sklearn StandardScaler/MinMaxScaler style)
var feature_matrix = np.column_stack(new[] { 
    longitude, latitude, housing_median_age, total_rooms, 
    total_bedrooms, population, households, median_income 
});

// Min-max scaling (equivalent to sklearn.preprocessing.MinMaxScaler)
var feature_min = feature_matrix.min(axis: 0, keepdims: true);
var feature_max = feature_matrix.max(axis: 0, keepdims: true);
var scaled_features = (feature_matrix - feature_min) / (feature_max - feature_min);

// Handle outliers with clipping
scaled_features = np.clip(scaled_features, 0.0f, 1.0f);

Console.WriteLine($"Scaled features shape: {scaled_features.shape}");
```

This code applies min-max normalization to scale all features to a [0,1] range. We cap extreme outliers to prevent them from skewing the normalization, which helps with model convergence during training.

 
 These code examples are reference implementations of common data transformations in TensorFlow.NET. Compare the output of your AI agent with this code, and correct your agent if needed.
 { .tip }
 
 To perform the transformations and get access to the transformed data, you'll need code like this:

```csharp
// Final data transformation for TensorFlow.NET
var finalData = normalizedData.Select(x => new
{
    Features = new float[] 
    {
        x.NormalizedLongitude,
        x.NormalizedLatitude,
        x.NormalizedAge,
        x.NormalizedIncome,
        x.NormalizedRooms,
        x.NormalizedBedrooms,
        x.NormalizedPopulation,
        x.NormalizedHouseholds
    }.Concat(x.LatitudeEncoded).Concat(x.LongitudeEncoded).ToArray(),
    Target = x.MedianHouseValue
}).ToList();

Console.WriteLine($"Final feature vector size: {finalData.First().Features.Length}");
```

This code combines all normalized features and one-hot encoded location vectors into a single feature array suitable for TensorFlow.NET training. Each record now has a standardized feature vector and target value.

 
 #### Test The Code
 
 My Claude 3.7 agent added a bit of extra code after the pipeline to output a sample row from the transformed data. My run looked like this:
 
 ![Pipeline Run Output](../img/pipeline-run.png)
 { .img-fluid .mb-4 }
 
 You can see that I decided to remove outliers by getting rid of all rows with a population larger than 5000. There were 265 housing blocks matching that condition in the dataset. The new computed column **RoomsPerPerson** has a numeric range from 0.0019 to 1.0, this is because I normalized all columns, including this one. And in the sample row, you can clearly see that the latitude and longitude values have been one-hot encoded into 10-element numerical vectors.
 
 Everything seems to be working.
 
 #### Summary
 
 In this lesson, you put on your data scientist hat and decided which data transformation steps to apply to the dataset. The AI agent then generated the corresponding TensorFlow.NET pipeline code for you.
 
 Coming up with the correct data transformations for any given dataset requires deep domain knowledge and a fair bit of intuition, and this is not something an agent can reliably do for you. Don't fall into the trap of asking the agent to come up with the transformations, because this is your job!
 
 So always make a plan first, based on your analysis of the dataset. Then prompt the agent and ask it to follow your plan.
 
 To wrap this lab up, let's see if we can create a cross product of the latitude and longitude vectors.
 
 {{< /encrypt >}}
 
 # Cross Latitude And Longitude
 
 Now let's perform one final data transformation: we're going to calculate the cross product of the encoded latitude and longitude, to create a new 100-element vector of zeroes and ones. We're layering a 10x10 grid over the state of California and placing a single '1' value in the grid to indicate the location of the housing block.
 
 Let's get started.
 
 {{< encrypt >}}
 
 #### Cross The Latitude and Longitude
 
 Open the Copilot panel in Visual Studio Code and enter the following prompt:
 
 "Add a step to the transformation pipeline to calculate a vector cross product of LatitudeEncoded and LongitudeEncoded, creating a new vector with 100 elements."
 { .prompt }
 
My AI agent implemented the cross product like this:

```csharp
// Calculate cross product of latitude and longitude encodings  
var crossedData = encodedData.Select(x => new 
{
    x.Original,
    LocationCross = x.LatitudeEncoded
        .SelectMany((lat, i) => x.LongitudeEncoded.Select((lon, j) => lat * lon))
        .ToArray()
}).ToList();

Console.WriteLine("Sample cross products:");
for (int i = 0; i < 3; i++)
{
    var sample = crossedData[i].LocationCross;
    var nonZeroIndex = Array.IndexOf(sample, 1.0f);
    Console.WriteLine($"Record {i}: Non-zero at index {nonZeroIndex}");
}
```

This code creates a 100-element cross product vector by multiplying each element of the latitude encoding with each element of the longitude encoding. This creates a 10x10 grid representation where exactly one cell has value 1.0, representing the housing block's location.
 

 You should get something like the following output:
 
 ![Cross Of Latitude And Longitude](../img/cross-console.png)
 { .img-fluid .mb-4 }
 
 You can see from the three sampled test rows that the cross product is a 100-element one-hot encoded vector, and that we have only a single '1' value in each vector.
 
 #### Summary
 
 Vector crossing is a handy trick for dealing with latitude and longitude features. Instead of training a machine learning model on two separate 10-element vectors, we now train on a single 100-element vector.
 
 This gives a machine learning model the freedom to treat each grid cell independently from all others. For example, a model could learn that housing blocks in San Francisco are very expensive, but if you travel a couple of miles east, the price drops rapidly. The model will be able to optimize predictions for these two regions independently.
  
 {{< /encrypt >}}
 
 # Recap
 
Congratulations on finishing the lab. Here's what you have learned.

{{< encrypt >}}

You learned how to analyze the **California Housing dataset**, both automatically by having an AI agent scan the data for you, and manually by inspecting the data by hand. You also learned how to prompt an AI agent to quickly generate **dataset visualizations** with ScottPlot, like feature histograms and correlation heatmaps.

You learned that the California Housing dataset needs a lot of preprocessing before we can use it for machine learning training. The median house values have been **clipped** at $500,000 and there are extreme **outliers** with a very high population and total number of rooms. You also discovered that many columns are strongly **correlated** with each other.

You learned how to prompt an AI agent to generate all the code to set up a **data transformation pipeline**. You learned what reference implementations of common data transformations look like in Microsoft.ML, and are now able to supervise the output of your AI agent for future work.

The dataset contains latitude and longitude columns. You learned how to **bin-, one-hot encode- and cross** them to create a 10x10 grid overlaying the state of California.

{{< /encrypt >}}

# Conclusion

This concludes the section on feature engineering. 

You now have some hands-on experience scrubbing, scaling, transforming, binning, one-hot encoding and crossing feature data. You'll use these skills in later labs to optimize your machine learning predictions.

The exact sequence of data transformation steps has a huge impact on the accuracy of machine learning predictions. This is why feature engineering is such an important step in data science.

As you practice with more and more datasets, you will slowly build an intuition for choosing the right transformation step for each feature column in your data.

But until then, just remember to always normalize your data and one-hot encode anything that looks like a category!

# Train A Regression Model

We're going to continue with the code we wrote in the previous lab. That C# application set up a data transformation pipeline to load the California Housing dataset and clean up the data using several feature engineering techniques.

So all we need to do is append a few command to the end of the pipeline to train and evaluate a regression model on the data.

{{< encrypt >}}

#### Split The Dataset

But first, we need to split the dataset into two partitions: one for training and one for testing. The training partition is typically a randomly shuffled subset of around 80% of all data, with the remaining 20% reserved for testing.

We do this, because sometimes a machine learning model will memorize all the labels in a dataset, instead of learning the subtle patterns hidden in the data itself. When this happens, the model will produce excellent predictions for all the data it has been trained on, but struggle with data it has never seen before.

By keeping 20% of our data hidden from the model, we can check if this unwanted process of memorization (called **overfitting**) is actually happening.

So let's split our data into an 80% partition for training and a 20% partition for testing.

Open the code from the previous lesson in Visual Studio Code. Keep the data transformation pipeline intact, but remove any other code you don't need anymore.

Then open the Copilot panel and type the following prompt:

"Split the data into two partitions: 80% for training and 20% for testing."
{ .prompt }

You should get the following code:

```csharp
// Shuffle the data randomly
var random = new Random(42);
var shuffledData = filteredData.OrderBy(x => random.Next()).ToList();

// Split into 80% training and 20% testing
var splitIndex = (int)(shuffledData.Count * 0.8);
var trainingData = shuffledData.Take(splitIndex).ToList();
var testingData = shuffledData.Skip(splitIndex).ToList();

Console.WriteLine($"Training samples: {trainingData.Count}");
Console.WriteLine($"Testing samples: {testingData.Count}");
```

This code uses LINQ to randomly shuffle the data and then splits it into training (80%) and testing (20%) partitions. We set a random seed for reproducible results.


#### Add A Machine Learning Algorithm

Now let's add a machine learning algorithm to the pipeline.

"Add code to set up a linear regression algorithm using TensorFlow.NET."
{ .prompt }

That should produce the following code:

```csharp
// Train-test split using NumSharp (sklearn-style)
var n_samples = scaled_features.shape[0];
var train_size = (int)(n_samples * 0.8);

// Shuffle and split
var indices = np.arange(n_samples);
np.random.shuffle(indices);

var train_idx = indices[:train_size];
var test_idx = indices[train_size:];

// Create train/test sets
var X_train = scaled_features[train_idx];
var y_train = median_house_value[train_idx].reshape(-1, 1);
var X_test = scaled_features[test_idx];
var y_test = median_house_value[test_idx].reshape(-1, 1);

var n_features = X_train.shape[1];

// Create TensorFlow model (Python TensorFlow style)
var X = tf.placeholder(tf.float32, shape: (None, n_features), name: "X");
var y = tf.placeholder(tf.float32, shape: (None, 1), name: "y");

// Model parameters (Python TensorFlow style)
var W = tf.Variable(tf.random.normal((n_features, 1), stddev: 0.1), name: "weights");
var b = tf.Variable(tf.zeros((1,)), name: "bias");

// Linear model: predictions = X @ W + b (Python @ operator style)
var predictions = tf.add(tf.matmul(X, W), b, name: "predictions");

// Loss function: MSE (Python TensorFlow style)
var loss = tf.reduce_mean(tf.square(predictions - y), name: "mse_loss");

// Optimizer (Python style)
var learning_rate = 0.01;
var optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss);
```

This code sets up a linear regression model using TensorFlow.NET. We define placeholders for input features (X) and target values (Y), create weight and bias variables, and set up the linear model with mean squared error loss and gradient descent optimization.


Be careful when you run this prompt! My AI agent generated code that included all features, including **MedianHouseValue**, **Latitude**, **Longitude**, **TotalRooms**, **TotalBedrooms**, **Population**, **Househoulds** and the encoded latitude and longitude.

This is obviously wrong, as the location cross product replaces all other latitude and longitude columns, and **RoomsPerPerson** replaces all other room- and person-related columns.

Even worse, did you notice the **MedianHouseValue** column in that list? This is the label that we're trying to predict. If we train a model on the label itself, the model can simply ignore all other features and output the label directly. This is like asking the model to make a prediction, and then giving it the actual answer it is supposed to predict. 

Always be vigilant. AI agents can easily make mistakes like this, because they do not understand the meaning of each dataset column. Your job as a data scientist is to make sure that the generated code does not contain any bugs.
{ .tip }

Linear regression with gradient descent is a fundamental machine learning algorithm that finds the optimal weights by iteratively minimizing the cost function. TensorFlow.NET implements this efficiently using automatic differentiation. If you're interested in the mathematical foundations, you can read more about linear regression on Wikipedia:

https://en.wikipedia.org/wiki/Linear_regression


#### Train A Machine Learning Model

Now let's train a machine learning model using our data transformation pipeline and the regression algorithm:

"Train a machine learning model on the training set using the regression algorithm."
{ .prompt }

That will produce the following code:

```csharp
// Training session (Python TensorFlow style)
using var sess = tf.Session();
sess.run(tf.global_variables_initializer());

// Training hyperparameters
var epochs = 1000;
var print_every = 100;

// Training loop (Python style)
Console.WriteLine("Training started...");
for (int epoch in range(1, epochs + 1))  // Python range style
{
    // Forward pass and backpropagation
    var feed_dict = new FeedDict {
        [X] = X_train,
        [y] = y_train
    };
    
    var (_, loss_val) = sess.run((optimizer, loss), feed_dict);
    
    // Print progress (Python style)
    if (epoch % print_every == 0)
    {
        Console.WriteLine($"Epoch {epoch}/{epochs}, Loss: {loss_val:.4f}");
    }
}

Console.WriteLine("Training completed!");
```
 
This training loop uses Python-style syntax with FeedDict for cleaner feed management, Adam optimizer (more common in Python), and Python-style string formatting for progress reporting.


#### Summary

In this lesson, you completed the machine learning pipeline you built in the previous lesson, by adding a machine learning algorithm. Then you split the data into a training and testing set, and trained a model on the training set.

In the next lesson, we'll calculate the prediction evaluation metrics to find out how good the model is at predicting house prices.

{{< /encrypt >}}

# Evaluate The Results

Now let's evaluate the quality of the model by having it generate predictions (called **scoring**) for the remaining 20% of data in the test partition. Then we'll compare those predictions to the actual median house values, and calculate the regression evaluation metrics.

So imagine you are a realtor in California selling houses. What kind of prediction accuracy would you consider acceptable?

{{< encrypt >}}

Determine the minimum mean absolute error or root mean square error values you deem acceptable. This will be the target your model needs to beat.
{ .homework }

#### Calculate Evaluation Metrics

Enter the following prompt:

"Use the trained model to create predictions for the test set, and then calculate evaluation metrics for these predictions and print them."
{ .prompt }

That should create the following code:

```csharp
// Prepare test data
var testX = np.zeros((testingData.Count, numFeatures), dtype: np.float32);
var testY = np.zeros((testingData.Count, 1), dtype: np.float32);

for (int i = 0; i < testingData.Count; i++)
{
    var record = testingData[i];
    testX[i, 0] = record.Longitude;
    testX[i, 1] = record.Latitude;
    testX[i, 2] = record.HousingMedianAge;
    testX[i, 3] = record.TotalRooms;
    testX[i, 4] = record.TotalBedrooms;
    testX[i, 5] = record.Population;
    testX[i, 6] = record.Households;
    testX[i, 7] = record.MedianIncome;
    
    testY[i, 0] = record.MedianHouseValue;
}

// Make predictions using TensorFlow operations
var predictions = sess.run(pred, new FeedItem(X, testX));

// Calculate evaluation metrics
// Convert to NumSharp arrays for vectorized operations (Python-like)
var y_true = np.array(testY.ToArray<float>());
var y_pred = np.array(predictions.ToArray<float>());

// Calculate metrics using vectorized operations (sklearn-like)
var mae = np.mean(np.abs(y_true - y_pred));
var mse = np.mean(np.power(y_true - y_pred, 2));
var rmse = np.sqrt(mse);

// Calculate R-squared (sklearn-like)
var y_mean = np.mean(y_true);
var ss_res = np.sum(np.power(y_true - y_pred, 2));
var ss_tot = np.sum(np.power(y_true - y_mean, 2));
var r2 = 1.0 - (ss_res / ss_tot);

Console.WriteLine($"Mean Absolute Error: ${mae:F0}");
Console.WriteLine($"Root Mean Squared Error: ${rmse:F0}");
Console.WriteLine($"R-Squared: {r2:F4}");
```

This code evaluates the trained TensorFlow.NET model on the test set by calculating predictions and computing regression metrics including MAE, RMSE, and R-squared. These metrics help assess how well the model generalizes to unseen data.


This code calculates the following metrics:

- **RSquared**: this is the coefficient of determination, a common evaluation metric for regression models. It tells you how well your model explains the variance in the data, or how good the predictions are compared to simply predicting the mean.
- **RootMeanSquaredError**: this is the root mean squared error or RMSE value. It’s the go-to metric in the field of machine learning to evaluate models and rate their accuracy. RMSE represents the length of a vector in n-dimensional space, made up of the error in each individual prediction.
- **MeanSquaredError**: this is the mean squared error, or MSE value. Note that RMSE and MSE are related: RMSE is the square root of MSE.
- **MeanAbsoluteError**: this is the mean absolute prediction error.

Note that both RMSE and MAE are expressed in dollars. They can both be interpreted as a kind of 'average error' value, but the RMSE will respond much more strongly to large prediction errors. Therefore, if RMSE > MAE, it means the model struggles with some predictions and generates relatively large errors. 

You should get the following output:

![Regression Model Evaluation](../img/evaluate.jpg)
{ .img-fluid .mb-4 }

Let's analyze my results:

The R-squared value is **0.6013**. This means the model is able to explain ~60% of the variance in housing prices. This is not bad, it's capturing a majority of the pattern in the data, but still leaves ~40% unexplained. This indicates a moderate-to-strong fit, and for real-world housing data, we could consider this a reasonable result.

The mean absolute error (MAE) is **$42,760**. So on average, the model's predictions are off by about $42k. That's not bad at all, given that the most expensive house in our dataset is $500k.

The mean squared error (MSE) is **~3.65 billion**. Large errors are penalized much more heavily in this metric because of the squaring, so this large number indicates the presence of some high-error predictions.

The root mean squared error (RMSE) is **~$60,439**. This metric is similar to MAE, but gives more weight to large errors. The fact that the RMSE > MAE suggests that the model occasionally makes big errors in its predictions.

So how did your model do?

Compare your model with the target you set earlier. Did it make predictions that beat the target? Are you happy with the predictive quality of your model? Feel free to experiment with other learning algorithms to try and get a better result.
{ .homework }

#### Summary

Evaluation is an essential step in machine learning, because this is where we check if the predictions our model is making are any good. If our model cannot beat the target we set, we'll have to go back to the drawing board and tweak the data transformations or pick a different learning algorithm.

A target of $50k is perfectly reasonable for housing, it means you'll accept a 10% prediction error for houses worth $500k. And the SDCA learning algorithm combined with all the data transformations we discussed was able to beat that target.

Let's continue working on our machine learning app and add a couple more features.

{{< /encrypt >}}

# Save And Load The Model

When you have a machine learning model with good prediction quality, you may want to save the model to a file so that you can easily use it later.

Saving a model will export all of the internal model weights, which represent the knowledge the model has gathered during the training. These weights are just a series of numbers, and saving these numbers to a file safeguards this knowledge and makes it available for later use.

{{< encrypt >}}

When we want to use a model to make predictions, we can simply set up a blank machine learning model, and then load knowledge into it by importing the weights back in to the model. This bypasses the entire training process, which is great because training a large model can sometimes take weeks or months!

Let's enhance our app with some simple code to save the weights of the fully trained model to a file.

#### Save The Model

Open Visual Studio Code and enter this prompt in the Copilot panel:

"Add code to save the fully trained model to a file."
{ .prompt }

Your agent should generate the following code:

```csharp
// Save the trained model
var saver = tf.train.Saver();
var savePath = saver.save(sess, "./housing_model.ckpt");
Console.WriteLine($"Model saved to: {savePath}");

// Also save model metadata
using (var writer = new StreamWriter("model_info.txt"))
{
    writer.WriteLine($"Feature count: {numFeatures}");
    writer.WriteLine($"Training samples: {trainingData.Count}");
    writer.WriteLine($"Test samples: {testingData.Count}");
    writer.WriteLine($"Final training cost: {sess.run(cost, new FeedItem(X, trainX), new FeedItem(Y, trainY))}");
}
```

This code saves the trained TensorFlow model using tf.train.Saver, which creates checkpoint files containing the model's weights and graph structure. We also save metadata about the training process for future reference.

The model is saved in TensorFlow's checkpoint format (.ckpt), which includes the model weights, biases, and computational graph structure. This format allows for efficient loading and continued training if needed.

There's a special universal file format called ONNX that you can use to transfer knowledge between machine learning models running on different platforms. So let's modify our app to use the ONNX format instead. 

#### Save The Model In ONNX Format

Enter the following prompt:

"Add code to save the fully trained model to a file in the ONNX format."
{ .prompt }

 The generated code should look like this:

```csharp
// Save model in ONNX format (requires tf2onnx converter)
// Note: This typically requires Python's tf2onnx package
// For demonstration, we show the conceptual approach:

// First save as TensorFlow SavedModel format
var builder = new SavedModelBuilder("./saved_model");
builder.add_meta_graph_and_variables(sess, new[] { "serve" });
builder.save();

Console.WriteLine("Model saved in SavedModel format");
Console.WriteLine("To convert to ONNX, use: python -m tf2onnx.convert --saved-model ./saved_model --output housing_model.onnx");
```

This code saves the model in TensorFlow's SavedModel format, which can then be converted to ONNX using external tools. The ONNX format enables cross-platform deployment and inference.

ONNX (Open Neural Network Exchange) is excellent for deploying models across different frameworks and platforms. However, the conversion process often requires additional tools and may not support all TensorFlow operations.
{ .tip }


#### Loading The Model

Let's add some code to load the model from a file. We can also have the app ask us if we want to train the model or simply load it from a file directly.

Enter the following prompt:

"Add code to load the model from a file. When the app starts, ask the user if they want to train a model and save it to a file, or load the model from a file and use it to generate predictions."
{ .prompt }

This will add a query to your app, and based on your decision, it will either train the model and save it, or load the model and evaluate it.

The code to load a model from a file looks like this:

```csharp
// Load previously saved model
public static Session LoadModel(string modelPath)
{
    var sess = tf.Session();
    var saver = tf.train.Saver();
    
    // Restore the model
    saver.restore(sess, modelPath);
    Console.WriteLine($"Model loaded from: {modelPath}");
    
    return sess;
}

// Usage:
if (File.Exists("./housing_model.ckpt.meta"))
{
    Console.WriteLine("Loading existing model...");
    var loadedSess = LoadModel("./housing_model.ckpt");
    // Use loadedSess for predictions
}
else
{
    Console.WriteLine("No existing model found. Please train a model first.");
}
```

This code demonstrates how to load a previously saved TensorFlow model using tf.train.Saver.restore(). The loaded session contains all the trained weights and can immediately be used for making predictions without retraining.

Let's test the code. Here's what I get when I choose to train the model:

![Training And Saving The Model](../img/save-model.jpg)
{ .img-fluid .mb-4 }

And this is what I get when I ask the app to load the model from a file:

![Loading The Model](../img/load-model.jpg)
{ .img-fluid .mb-4 }

#### Summary

Saving the fully trained machine learning model is a great trick if you want to use the model to generate lots of predictions. Instead of having to train your model every time your app starts up, you can simply load all the knowledge into a blank model instantly.

We do this all the time in machine learning, because large complex models often require weeks (or even months) to train.

Do be careful with the ONNX format. It's great that we can transfer knowledge between models running on different platforms, but many data transformations (like custom mappings) are not supported. You would not be able to save and then reload the California Housing model in this format.

{{< /encrypt >}}

# Make A Prediction

To wrap up, let’s use the model to make a prediction.

We're going to invent a fake housing block in San Francisco, in the middle of the Mission district. The block has 2500 rooms, 1000 bedrooms, houses 500 people and 150 households. The apartments are 10 years old on average, and the normalized median income in that neighborhood is 2.0.

For how much could you sell an apartment in that housing block?

We will ask our AI agent to write code that prompts us for all the properties of a single housing block, and then we'll use the machine learning model to predict what the median house value will be for any apartment in the block.

{{< encrypt >}}

#### Make A Prediction

Enter the following prompt:

"Add code to prompt the user for all the properties of a single housing block, and then use the model to generate a prediction of the median house value."
{ .prompt }

The agent will generate the following code:

```csharp
// Interactive prediction function
public static void MakePrediction(Session sess, Tensor X, Tensor pred)
{
    Console.WriteLine("\nEnter housing block details:");
    
    Console.Write("Longitude: ");
    var longitude = float.Parse(Console.ReadLine());
    
    Console.Write("Latitude: ");
    var latitude = float.Parse(Console.ReadLine());
    
    Console.Write("Housing Median Age: ");
    var age = float.Parse(Console.ReadLine());
    
    Console.Write("Total Rooms: ");
    var rooms = float.Parse(Console.ReadLine());
    
    Console.Write("Total Bedrooms: ");
    var bedrooms = float.Parse(Console.ReadLine());
    
    Console.Write("Population: ");
    var population = float.Parse(Console.ReadLine());
    
    Console.Write("Households: ");
    var households = float.Parse(Console.ReadLine());
    
    Console.Write("Median Income: ");
    var income = float.Parse(Console.ReadLine());
    
    // Create input tensor
    var inputData = np.array(new float[,] {
        { longitude, latitude, age, rooms, bedrooms, population, households, income }
    });
    
    // Make prediction
    var prediction = sess.run(pred, new FeedItem(X, inputData));
    var predictedValue = prediction[0, 0];
    
    Console.WriteLine($"\nPredicted median house value: ${predictedValue:F0}");
}

// Usage:
MakePrediction(sess, X, pred);
```

This code creates an interactive prediction function that prompts the user for housing block characteristics and uses the trained TensorFlow model to predict the median house value. It demonstrates how to format input data and run inference with the trained model.


Let's try this for the fake apartment I introduced earlier. Here is the data you need to enter:

- Longitude = 37.760653
- Latitude = -122.418725
- HousingMedianAge = 10
- TotalRooms = 2500
- TotalBedrooms = 1000
- Population = 500
- Households = 150
- MedianIncome = 2.0

And this is the output I get:

![Using The Model To Make A Prediction](../img/prediction.jpg)
{ .img-fluid .mb-4 }

I get a predicted median house value of **$143,265**.

What prediction did you get? Try changing the input data to see how this affects the predicted house value. Do the changes in prediction value make sense to you?
{ .homework }

#### Summary

Making predictions with the fully trained machine learning model is super easy. All you need to do is set up a prediction engine and then feed data into it. The engine will perform the exact same data transformations you used when training the model, and this ensures that the predictions make sense.

Using a machine learning model to generate predictions is called inference, and it requires a lot less compute capacity than training the model did. The inference compute load is often thousands of times smaller than the training load.

So it makes perfect sense to run a model in production, where you can quickly initialize it by importing the weights from a file. If over time the data changes significantly and the prediction quality starts dropping (this is called **data drift**), you can re-train the model on a separate compute platform and copy the new weight file back to production. 

{{< /encrypt >}}

# Improve Your Results

There are many factors that influence the quality of your model predictions, including how you process the dataset, which regression algorithm you pick, and how you configure the training hyperparameters.

Here are a couple of things you could do to improve your model:

{{< encrypt >}}

- Bin the latitude and longitude into more than 10 bins, thus creating a finer grid over the state of California.
- Use a different learning algorithm.
- Use different hyperparameter values for your learning algorithm.
- Eliminate outliers with a large number of rooms.
- Eliminate outliers with very small populations.
- Try to bin other columns, for example replacing **HousingMedianAge** with a three-element one-hot encoding vector with columns "Young", "Median" and "Old".

Experiment with different data processing steps and regression algorithms. Document your best-performing machine learning pipeline for this dataset, and write down the corresponding regression evaluation metrics.
{ .homework }

How close can you make your predictions to the actual house prices? Feel free to try out different approaches. This is how you build valuable machine learning skills!

{{< /encrypt >}}

# Hall Of Fame

Would you like to be famous? You can [submit your best-performing model](mailto:mark@mdfteurope.com) for inclusion in this hall of fame, which lists the best regression evaluation scores for the California Housing dataset. I've added my own results as a baseline, using the transformations I mentioned in the lab. 

Can you beat my score?

| Rank | Name | Platform | Algorithm | Transformations |   MAE   |  RMSE   |
|------|------|----------|-----------|-----------------|---------|---------|
|  1   | Mark | TensorFlow.NET | Linear Regression | As mentioned in lab | $42,760 | $60,439 |

I will periodically collect new submissions and merge them into the hall of fame. I'll share the list in my courses and on social media. If you make the list, you'll be famous!

# Recap-2

Congratulations on finishing the lab. Here's what you have learned.

{{< encrypt >}}

You learned to **split the dataset** into separate parts for training and testing to prevent the machine learning model from memorizing the median house values for every single housing block in the dataset. You used TensorFlow.NET with NumSharp arrays to prepare and process the California housing data, converting it into tensor format suitable for machine learning.

You used **TensorFlow.NET** to create a linear regression model with placeholders, variables, and computational graphs. You trained the model using gradient descent optimization and mean squared error loss. You learned how to create TensorFlow sessions, run training loops, and monitor the cost function during training. You also learned about saving models in TensorFlow's native format and the **ONNX** format for cross-platform compatibility.

And finally, you learned how you can use the trained TensorFlow model to **generate predictions** by feeding new data through the computational graph using the trained weights and biases.

You completed the lab by experimenting with different data processing steps and regression algorithms to find the best-performing model. 

{{< /encrypt >}}

# Conclusion-2

You now have hands-on experience building a C# app that trains a regression model on a dataset, and then using the fully trained model to generate predictions. This specific dataset, California Housing, required a lot of preprocessing and has features that only weakly correlate with the label. This makes predicting accurate house prices quite challenging. Nevertheless, the best mean absolute error you can achieve is around $28,000.

I hope you also noticed that you cannot simply keep increasing the number of longitude and latitude bins. There's an optimum, and once you go beyond that point, the predictions start degrading in quality again. Knowing where a house is with centimeter-level accuracy is not helpful at all, so we always want to work with a specific level of uncertainty to help our models make better predictions.

This was a fun lab, right?
