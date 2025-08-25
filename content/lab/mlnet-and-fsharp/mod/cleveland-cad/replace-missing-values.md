---
title: "Replace Missing Values"
type: "lesson"
layout: "default"
sortkey: 50
---

# Replace Missing Values

The Cleveland CAD dataset has two columns with missing values:
- **NumMajorVessels**
- **Thalassemia** 

You can easily spot them in the Heart-Disease.csv datafile as '?' characters where the values should be.

{{< encrypt >}}

![Missing Values In The Cleveland CAD Dataset](../img/missing.jpg)
{ .img-fluid .mb-4 }

You can see from the scrollbar on the right that there are 6 records in total with missing values. These records are causing the columns to be loaded as `string` instead of `float` data, and this is why the histograms did not appear in the previous lesson.

ML.NET has built-in support for missing data. When a column value is empty, it will be loaded as a `float.NaN` value and we can then decide how to treat these values. But unfortunately, ML.NET does not understand the meaning of the '?' characters in the Cleveland CAD dataset. 

We'll have to write some custom code to deal with them instead. 

#### Parse Missing Values

Let's start by building a data transformation pipeline that uses a `CustomMapping` to convert the `string` values to `float`, and represent missing values as `float.NaN`: 

Enter the following prompt:

"Build an ML.NET pipeline that processes the dataset and parses the Ca and Thal string columns to float32. If the column value is '?', store it as float.NaN."
{ .prompt }

The agent will make a number of changes to your code. First, it will modify and rename the `HeartData` record type to accommodate the **NumMajorVessels** and **Thalassemia** fields as strings:

```fsharp
// Input record type that represents one record from the Heart-Disease.csv file
// This type handles the raw CSV data with potential missing values as '?'
[<CLIMutable>]
type HeartDataInput = {
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
    [<LoadColumn(12); ColumnName("Ca")>] NumMajorVessels: string
    [<LoadColumn(13); ColumnName("Thal")>] Thalassemia: string
    [<LoadColumn(14); ColumnName("Diag")>] Diagnosis: float32
}
```

Note that the **NumMajorVessels** and **Thalassemia** fields are now of type `string`. 

Next, the agent will create a new `HeartData` record type with **NumMajorVessels** and **Thalassemia** fields of type `float32`:

```fsharp
// Record type for transformed data with properly parsed numeric values
[<CLIMutable>]
type HeartData = {
    RowID: float32
    Age: float32
    Sex: float32
    ChestPainType: float32
    RestingBloodPressure: float32
    Cholesterol: float32
    FastingBloodSugar: float32
    RestingECG: float32
    MaxHeartRate: float32
    ExerciseInducedAngina: float32
    STDepression: float32
    Slope: float32
    NumMajorVessels: float32
    Thalassemia: float32
    Diagnosis: float32
}
```

And finally, the agent will add an ML.NET data transformation pipeline to your code with a `CustomMapping` that converts the '?' characters to `float.NaN` values:

```fsharp
// Define a transformation pipeline to handle the '?' values and convert to float.NaN
let pipeline = mlContext.Transforms.CustomMapping<HeartDataInput, HeartData>(
    (fun input output -> 
        // Copy all the direct float values
        output.RowID <- input.RowID
        output.Age <- input.Age
        output.Sex <- input.Sex
        output.ChestPainType <- input.ChestPainType
        output.RestingBloodPressure <- input.RestingBloodPressure
        output.Cholesterol <- input.Cholesterol
        output.FastingBloodSugar <- input.FastingBloodSugar
        output.RestingECG <- input.RestingECG
        output.MaxHeartRate <- input.MaxHeartRate
        output.ExerciseInducedAngina <- input.ExerciseInducedAngina
        output.STDepression <- input.STDepression
        output.Slope <- input.Slope
        output.Diagnosis <- input.Diagnosis
        
        // Handle NumMajorVessels: convert '?' to NaN, or parse the value
        if String.IsNullOrEmpty(input.NumMajorVessels) || input.NumMajorVessels = "?" then
            output.NumMajorVessels <- float32 nan
        else
            output.NumMajorVessels <- float32 input.NumMajorVessels

        // Handle Thalassemia: convert '?' to NaN, or parse the value
        if String.IsNullOrEmpty(input.Thalassemia) || input.Thalassemia = "?" then
            output.Thalassemia <- float32 nan
        else
            output.Thalassemia <- float32 input.Thalassemia),
    "CustomMappingForHeartData")

// Apply the transformation pipeline
let transformedData = pipeline.Fit(rawDataView).Transform(rawDataView)

// Convert to enumerable for processing
let heartDataList = 
    mlContext.Data.CreateEnumerable<HeartData>(transformedData, reuseRowObject = false)
    |> List.ofSeq
```

Note that `CreateEnumerable` now operates on the `transformedData` dataview, which holds the patient data with the missing values correctly parsed. 

#### Replace Missing Values With Defaults

We're still not able to calculate the histogram matrix, because the Scottplot library cannot calculate a histogram from data that contains NaN values. So we're going to have to replace them with something else. 

Let's replace the missing data with benign values for healthy patients:

- **NumMajorVessels** represents the number of major vessels (0–3) colored by fluoroscopy. For a healthy patient, all vessels would be clearly visible on an X-ray so we'll use a default value of 3.
- **Thalassemia** represents the Thallium heart scan results: 3 = normal, 6 = fixed defect, 7 = reversible defect. We will use a default value of 3.

We can enter these defaults directly into the custom mapping code:

```fsharp
// Handle NumMajorVessels
if String.IsNullOrEmpty(input.NumMajorVessels) || input.NumMajorVessels = "?" then
    output.NumMajorVessels <- 3.0f // default for healthy patient
else
    output.NumMajorVessels <- float32 input.NumMajorVessels

// Handle Thalassemia
if String.IsNullOrEmpty(input.Thalassemia) || input.Thalassemia = "?" then
    output.Thalassemia <- 3.0f // default for healthy patient
else
    output.Thalassemia <- float32 input.Thalassemia
```

#### Change Diagnosis To Binary Classification

While we're at it, let's also change the label column to a binary classification value, with a 0 indicating the patient is healthy, and a 1 indicating the patient has cardiovascular disease. 

We'll make the following change to the CustomMapping code:

```fsharp
// Change diagnosis to binary classification
output.Diagnosis <- if input.Diagnosis = 0.0f then 0.0f else 1.0f
```

With these changes, the histogram matrix now looks like this:

![Histogram Grid For Full Dataset](../img/histograms-2.png)
{.img-fluid .mb-4}

A couple of observations:

- The **Diag** column is now nicely balanced, with roughly the same number of sick and healthy patients. 

- The **NumMajorVessels** histogram shows that the median value is zero. For the majority of patients, no vessels are colored by fluoroscopy at all. The 'all-healthy' value of 3 is actually an outlier, and we should consider using a default of zero instead.

- For the **Thalassemia** column, a value of 3 is the most common. This seems to be a good default for missing values. 

We're now ready to calculate the Pearson correlation matrix. We will do that in the next lesson. 

{{< /encrypt >}}