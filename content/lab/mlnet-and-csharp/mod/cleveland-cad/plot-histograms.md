---
title: "Plot The Histogram Matrix"
type: "lesson"
layout: "default"
sortkey: 40
---

In this lesson, you’ll going to generate the feature histogram grid that contains a histogram for every feature in the dataset. Like before, we're going to use our famous **HistogramUtils** class from the previous lab modules.

Let's get started.

#### Install Utility Classes And Dependencies

Let's see if Copilot can do the whole thing for us. We'll ask the agent to download the class from the Git repository and install any required dependencies for us. And while we're at it, we might as well ask for the **CorrelationUtils** and **ScatterplotUtils** classes too. 

You'll need the raw urls of the utility files hosted in the repository where you've stored all the code you created in the previous lab. We'll copy these urls into the prompt and ask the agent to import the classes directly into the current project.

I pushed my utility classes to a repository on Codeberg. Here is the prompt I used:

"Copy the HistogramUtils, CorrelationUtils and ScatterplotUtils classes from these repository urls and add them to the project. Install any required NuGet package dependencies to ensure that the code works: <br> - codeberg.org/mdft/ml-mlnet-csharp/raw/branch/main/TaxiFarePrediction/CorrelationUtils.cs <br> - codeberg.org/mdft/ml-mlnet-csharp/raw/branch/main/TaxiFarePrediction/HistogramUtils.cs <br> - codeberg.org/mdft/ml-mlnet-csharp/raw/branch/main/TaxiFarePrediction/ScatterUtils.cs"
{ .prompt }

This worked like a charm. 

#### Load The Heart-Disease.csv File

Now let's ask Copilot to write the code for loading the CSV file and generating a data class that represents one record from the file. 

At the bottom of the Copilot panel in Visual Studio Code, make sure the AI mode is set to 'Agent'. Then select your favorite model (I used Claude 3.7 Sonnet while preparing this lab).

Enter the following prompt:

"Write C# code to load the Heart-Disease.csv file, using the LoadFromTextFile method in ML.NET. Also create a data class that represents one record from the file, that can be used with the CreateEnumerable method in ML.NET to create a list of patient records."
{ .prompt }

And let Copilot write the code for you.

You should see the following data loading code in your project:

```csharp
// Create ML.NET context
var mlContext = new MLContext();

// Load data from the CSV file
var dataView = mlContext.Data.LoadFromTextFile<HeartData>(
    path: "Heart-Disease.csv",
    hasHeader: true,
    separatorChar: ',');

// Convert to enumerable for processing
var heartDataList = mlContext.Data.CreateEnumerable<HeartData>(
    dataView, reuseRowObject: false).ToList();
```

This code uses `LoadFromTextFile` to load the CSV file into a dataview, and `CreateEnumerable` to convert the loaded data into an enumeration of `HeartData` instances. 

This is what the `HeartData` class looks like:

```csharp
// Data class that represents one record from the Heart-Disease.csv file
public class HeartData
{
    [LoadColumn(0), ColumnName("csvbase_row_id")] public float RowID { get; set; }
    [LoadColumn(1), ColumnName("Age")] public float Age { get; set; }
    [LoadColumn(2), ColumnName("Sex")] public float Sex { get; set; }
    [LoadColumn(3), ColumnName("Cp")] public float ChestPainType { get; set; }
    [LoadColumn(4), ColumnName("TrestBps")] public float RestingBloodPressure { get; set; }
    [LoadColumn(5), ColumnName("Chol")] public float Cholesterol { get; set; }
    [LoadColumn(6), ColumnName("Fbs")] public float FastingBloodSugar { get; set; }
    [LoadColumn(7), ColumnName("RestEcg")] public float RestingECG { get; set; }
    [LoadColumn(8), ColumnName("Thalac")] public float MaxHeartRate { get; set; }
    [LoadColumn(9), ColumnName("Exang")] public float ExerciseInducedAngina { get; set; }
    [LoadColumn(10), ColumnName("OldPeak")] public float STDepression { get; set; }
    [LoadColumn(11), ColumnName("Slope")] public float Slope { get; set; }
    [LoadColumn(12), ColumnName("Ca")] public float NumMajorVessels { get; set; }
    [LoadColumn(13), ColumnName("Thal")] public float Thalassemia { get; set; }
    [LoadColumn(14), ColumnName("Diag")] public float Diagnosis { get; set; }
}
```

Note the use of `LoadColumn` attributes that specify the CSV column indices for each property. My agent also added `ColumnName` attributes that map each property to their corresponding CSV column names.

#### Generate The Histogram Matrix

To generate the histogram matrix, we'll use the same code as always:

```csharp
// get column names, skip row id and non-numeric columns
var columnNames = (from p in typeof(HeartData).GetProperties()
                   where p.Name != "RowID"
                            && (p.PropertyType == typeof(float)
                            || p.PropertyType == typeof(int))
                   select p.Name).ToArray();

// calculate the histogram grid
Console.WriteLine("Generating histograms...");
var grid = HistogramUtils.PlotAllHistograms<HeartData>(heartDataList, columnNames, columns: 4, rows: 4);

// save the grid
grid.SavePng("histograms.png", 1900, 1280);
```

Homework: add code to generate the histogram matrix. Then run your app and examine the histograms. What do you notice? Write down your conclusions.  
{ .homework }

Here's what I got:

![Histogram Grid For Full Dataset](../img/histograms.png)
{.img-fluid .mb-4}

And these are my conclusions:

- The patient with a **Cholesterol** level of 564 is indeed an outlier and should be removed. We could also consider removing the cluster of patients with cholesterol levels around 400. 

- The dataset is unbalanced with almost twice as many men as women. We need to _undersample_ the men to remove the bias in the **Sex** column, or the model will struggle to diagnose cardiovascular disease in women.  

- The **Diag** column is more or less balanced, but only if we treat this as a _binary_ classification problem where we predict if patients are healthy (Diag = 0) or sick (Diag > 0).

- The histograms for the **NumMajorVessels** and **Thalassemia** columns are empty, probably because these columns contain missing values that ML.NET is not parsing correctly during load. We are going to have to replace these missing values. 

In the next lesson, we'll deal with these issues by replacing the missing values in the **NumMajorVessels** and **Thalassemia** columns, and modifying the **Diag** column to only contain a 0 (patient is healthy) or a 1 (patient has a cardiovascular disease).

Then we'll generate the histogram matrix again and check if everything is ok. 
