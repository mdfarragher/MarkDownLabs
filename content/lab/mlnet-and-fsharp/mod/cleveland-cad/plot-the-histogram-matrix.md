---
title: "Plot The Histogram Matrix"
type: "lesson"
layout: "default"
sortkey: 40
---

# Plot The Histogram Matrix

In this lesson, you'll going to generate the feature histogram grid that contains a histogram for every feature in the dataset. Like before, we're going to use our famous `HistogramUtils` class from the previous lab modules.

Let's get started.

{{< encrypt >}}

#### Install Utility Classes And Dependencies

Let's see if Copilot can do the whole thing for us. We'll ask the agent to download the class from the Git repository and install any required dependencies for us. And while we're at it, we might as well ask for the **CorrelationUtils** and **ScatterplotUtils** classes too. 

You'll need the raw urls of the utility files hosted in the repository where you've stored all the code you created in the previous lab. We'll copy these urls into the prompt and ask the agent to import the classes directly into the current project.

I pushed my utility classes to a repository on Codeberg. Here is the prompt I used:

"Copy the HistogramUtils, CorrelationUtils and ScatterplotUtils modules from these repository urls and add them to the project. Install any required NuGet package dependencies to ensure that the code works: <br> - codeberg.org/mdft/ml-mlnet-fsharp/raw/branch/main/TaxiFarePrediction/CorrelationUtils.fs <br> - codeberg.org/mdft/ml-mlnet-fsharp/raw/branch/main/TaxiFarePrediction/HistogramUtils.fs <br> - codeberg.org/mdft/ml-mlnet-fsharp/raw/branch/main/TaxiFarePrediction/ScatterUtils.fs"
{ .prompt }

This worked like a charm. 

#### Load The Heart-Disease.csv File

Now let's ask Copilot to write the code for loading the CSV file and generating a data class that represents one record from the file. 

At the bottom of the Copilot panel in Visual Studio Code, make sure the AI mode is set to 'Agent'. Then select your favorite model (I used Claude 3.7 Sonnet while preparing this lab).

Enter the following prompt:

"Write F# code to load the Heart-Disease.csv file, using the LoadFromTextFile method in ML.NET. Also create a record type that represents one record from the file, that can be used with the CreateEnumerable method in ML.NET to create a list of patient records."
{ .prompt }

And let Copilot write the code for you.

You should see the following data loading code in your project:

```fsharp
// Create ML.NET context
let mlContext = MLContext()

// Load data from the CSV file
let dataView = mlContext.Data.LoadFromTextFile<HeartData>(
    path = "Heart-Disease.csv",
    hasHeader = true,
    separatorChar = ',')

// Convert to enumerable for processing
let heartDataList = 
    mlContext.Data.CreateEnumerable<HeartData>(dataView, reuseRowObject = false)
    |> List.ofSeq
```

This code uses `LoadFromTextFile` to load the CSV file into a dataview, and `CreateEnumerable` to convert the loaded data into a sequence of `HeartData` instances, which is then converted to a list using F# pipeline operators. 

This is what the `HeartData` record type looks like:

```fsharp
open Microsoft.ML.Data

// Record type that represents one record from the Heart-Disease.csv file
[<CLIMutable>]
type HeartData = {
    [<LoadColumn(0); ColumnName("csvbase_row_id")>] RowID: float32
    [<LoadColumn(1); ColumnName("Age")>] Age: float32
    [<LoadColumn(2); ColumnName("Sex")>] Sex: float32
    [<LoadColumn(3); ColumnName("Cp")>] ChestPainType: float32
    [<LoadColumn(4); ColumnName("TrestBps")>] RestingBloodPressure: float32
    [<LoadColumn(5); ColumnName("Chol")>] Cholesterol: float32
    [<LoadColumn(6); ColumnName("Fbs")>] FastingBloodSugar: float32
    [<LoadColumn(7); ColumnName("RestEcg")>] RestingECG: float32
    [<LoadColumn(8); ColumnName("Thalac")>] MaxHeartRate: float32
    [<LoadColumn(9); ColumnName("Exang")>] ExerciseInducedAngina: float32
    [<LoadColumn(10); ColumnName("OldPeak")>] STDepression: float32
    [<LoadColumn(11); ColumnName("Slope")>] Slope: float32
    [<LoadColumn(12); ColumnName("Ca")>] NumMajorVessels: float32
    [<LoadColumn(13); ColumnName("Thal")>] Thalassemia: float32
    [<LoadColumn(14); ColumnName("Diag")>] Diagnosis: float32
}
```

Note the use of `LoadColumn` attributes that specify the CSV column indices for each record field. My agent also added `ColumnName` attributes that map each field to their corresponding CSV column names.

#### Generate The Histogram Matrix

To generate the histogram matrix, we'll use the same code as always:

```fsharp
open System.Reflection

// get column names, skip row id and non-numeric columns
let columnNames = 
    typeof<HeartData>.GetProperties()
    |> Array.filter (fun p -> p.Name <> "RowID" && 
                             (p.PropertyType = typeof<float32> || p.PropertyType = typeof<int>))
    |> Array.map (fun p -> p.Name)

// calculate the histogram grid
Console.WriteLine("Generating histograms...")
let grid = HistogramUtils.PlotAllHistograms<HeartData>(heartDataList, columnNames, columns = 4, rows = 4)

// save the grid
grid.SavePng("histograms.png", 1900, 1280)
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

{{< /encrypt >}}