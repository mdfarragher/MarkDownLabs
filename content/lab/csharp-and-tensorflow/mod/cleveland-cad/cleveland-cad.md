# The Cleveland CAD Dataset

Coronary artery disease (CAD) involves the reduction of blood flow to the heart muscle due to build-up of plaque in the arteries of the heart. It is the most common form of cardiovascular disease, and the only reliable way to detect it right now is through a costly and invasive procedure called coronary angiography. This procedure represents the gold standard for detecting CAD, but unfortunately comes with increased risk to CAD patients.

So it would be really nice if we had some kind of reliable and non-invasive alternative to replace the current gold standard.

Other less invasive diagnostic tools do exist. Doctors have proposed using electrocardiograms, thallium scintigraphy and fluoroscopy of coronary calcification. However the diagnostic accuracy of these tests only ranges between 35%-75%.

We can try to build a machine learning model that trains on the results of these non-invasive tests and combines them with other patient attributes to generate a CAD diagnosis. If the model predictions are accurate enough, we can use the model to replace the current invasive procedure and save patient lives.

To train our model, we're going to use the the Cleveland CAD dataset from the University of California UCI. The data contains real-life diagnostic information of 303 anonymized patients and was compiled by Robert Detrano, M.D., Ph.D of the Cleveland Clinic Foundation back in 1988.

![Cleveland CAD Dataset](../img/data.jpg)
{ .img-fluid .pb-4 }

The dataset contains 13 features which include the results of the aforementioned non-invasive diagnostic tests along with other relevant patient information. The label represents the result of the invasive coronary angiogram and indicates the presence or absence of CAD in the patient. A label value of 0 indicates absence of CAD and label values 1-4 indicate the presence of CAD.

Do you think you can develop a medical-grade diagnostic tool for heart disease?

Let's find out!

# Get The Data

Let's start by downloading the Cleveland CAD dataset. 

{{< encrypt >}}

Grab the file from here: [Cleveland CAD dataset](https://csvbase.com/mdfarragher/Heart-Disease).

Download the file and save it as **Heart-Disease.csv**.

The file is a comma-separated text file with 15 columns of data:

- Row ID: a unique row identifier (added by CsvBase)
- Age
- Sex: 1 = male, 0 = female
- Chest Pain Type: 1 = typical angina, 2 = atypical angina , 3 = non-anginal pain, 4 = asymptomatic
- Resting blood pressure in mm Hg on admission to the hospital
- Serum cholesterol in mg/dl
- Fasting blood sugar > 120 mg/dl: 1 = true; 0 = false
- Resting EKG results: 0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes’ criteria
- Maximum heart rate achieved
- Exercise induced angina: 1 = yes; 0 = no
- ST depression induced by exercise relative to rest
- Slope of the peak exercise ST segment: 1 = up-sloping, 2 = flat, 3 = down-sloping
- Number of major vessels (0–3) colored by fluoroscopy
- Thallium heart scan results: 3 = normal, 6 = fixed defect, 7 = reversible defect
- Diagnosis of heart disease: 0 = normal risk, 1-4 = elevated risk

Columns 2-14 are patient diagnostic information, and the last column is the diagnosis: 0 means a healthy patient, and values 1-4 mean an elevated risk of heart disease.

Let's get started.

#### Set Up The Project

Now open your terminal and navigate to the folder where you want to create the project (e.g., **~/Documents**), and run:

```bash
dotnet new console -o HeartDisease
cd HeartDisease
```

This creates a new C# console application with:

- **Program.cs** – your main program file
- **HeartDisease.csproj** – your project file

Then move the Heart-Disease.csv file into this folder.

Now run the following command to install the Microsoft.ML machine learning library:

```bash
dotnet add package Microsoft.ML
```

Next, we're going to analyze the dataset and come up with a feature engineering plan.

{{< /encrypt >}}

# Analyze The Data

We’ll begin by analyzing the Cleveland CAD dataset and come up with a plan for feature engineering. Our goal is to map out all required data transformation steps in advance to make later machine learning training as successful as possible.

{{< encrypt >}}

#### Manually Explore the Data

Let’s start by exploring the dataset manually.

Open **Heart-Disease.csv** in Visual Studio Code, and start looking for patterns, issues, and feature characteristics.

What to look out for:

-    Are there any missing values, zeros, or inconsistent rows?
-    Are the values in each column within a reasonable range?
-    Can you spot any extremely large or very small values?
-    Is there any bias in columns like **Age** or **Sex**?
-    Do we have balanced populations of sick and healthy patients?

Write down 3 insights from your analysis.
{.homework}

#### Ask Copilot To Analyze The Dataset

You can also ask Copilot to analyze the CSV data and determine feature engineering steps. You should never blindly trust AI advice, but it can be insightful to run an AI scan after you've done your own analysis of the data, and compare Copilot's feedback to your own conclusions. 

Make sure the CSV file is still open in Visual Studio Code. Then expand the Copilot panel on the right-hand side of the screen, and enter the following prompt:

"You are a machine learning expert. Analyze this CSV file for use in a classification model that predicts Diag. What problems might the dataset have? What preprocessing steps would you suggest?"
{.prompt}

You can either paste in the column names and 5–10 sample rows, or upload the CSV file directly (if your agent supports file uploads).

![Analyze a dataset with an AI agent](../img/analyze.jpg)
{.img-fluid .mb-4}

#### What Might The Agent Suggest?

The agent may recommend steps like:

-    Normalize data columns
-    Handle outliers (like the patient with a **Chol** value of 564)
-    One-hot encode categorical columns like **Sex** and **Cp**
-    Handle missing values in the **Thal** and **Ca** columns
-    Balance the population of sick and healthy patients through sampling
-    Bin and one-hot encode the **Age** column into age groups

Write down 3 insights from the agent’s analysis.
{.homework}

Next, we'll generate a couple of histograms to see if we can find outliers in any of the data columns. 

{{< /encrypt >}}

# Plot The Histogram Matrix

In this lesson, you’ll going to generate the feature histogram grid that contains a histogram for every feature in the dataset. Like before, we're going to use our famous `HistogramUtils` class from the previous lab modules.

Let's get started.

{{< encrypt >}}

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

"Write C# code to load the Heart-Disease.csv file, using the LoadFromTextFile method in Tensorflow.NET. Also create a data class that represents one record from the file, that can be used with the CreateEnumerable method in Tensorflow.NET to create a list of patient records."
{ .prompt }

And let Copilot write the code for you.

You should see the following data loading code in your project:

```csharp
// Create Tensorflow.NET context
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

- The histograms for the **NumMajorVessels** and **Thalassemia** columns are empty, probably because these columns contain missing values that Tensorflow.NET is not parsing correctly during load. We are going to have to replace these missing values. 

In the next lesson, we'll deal with these issues by replacing the missing values in the **NumMajorVessels** and **Thalassemia** columns, and modifying the **Diag** column to only contain a 0 (patient is healthy) or a 1 (patient has a cardiovascular disease).

Then we'll generate the histogram matrix again and check if everything is ok. 

{{< /encrypt >}}

# Replace Missing Values

The Cleveland CAD dataset has two columns with missing values:
- **NumMajorVessels**
- **Thalassemia** 

You can easily spot them in the Heart-Disease.csv datafile as '?' characters where the values should be.

{{< encrypt >}}

![Missing Values In The Cleveland CAD Dataset](../img/missing.jpg)
{ .img-fluid .mb-4 }

You can see from the scrollbar on the right that there are 6 records in total with missing values. These records are causing the columns to be loaded as `string` instead of `float` data, and this is why the histograms did not appear in the previous lesson.

Tensorflow.NET has built-in support for missing data. When a column value is empty, it will be loaded as a `float.NaN` value and we can then decide how to treat these values. But unfortunately, Tensorflow.NET does not understand the meaning of the '?' characters in the Cleveland CAD dataset. 

We'll have to write some custom code to deal with them instead. 

#### Parse Missing Values

Let's start by building a data transformation pipeline that uses a `CustomMapping` to convert the `string` values to `float`, and represent missing values as `float.NaN`: 

Enter the following prompt:

"Build an Tensorflow.NET pipeline that processes the dataset and parses the Ca and Thal string columns to float. If the column value is '?', store it as float.NaN."
{ .prompt }

The agent will make a number of changes to your code. First, it will modify and rename the `HeartData` class to accomodate the **NumMajorVessels** and **Thalassemia** columns as strings:

```csharp
// Input data class that represents one record from the Heart-Disease.csv file
// This class handles the raw CSV data with potential missing values as '?'
public class HeartDataInput
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
    [LoadColumn(12), ColumnName("Ca")] public string? NumMajorVessels { get; set; }
    [LoadColumn(13), ColumnName("Thal")] public string? Thalassemia { get; set; }
    [LoadColumn(14), ColumnName("Diag")] public float Diagnosis { get; set; }
}
```

Note that the **NumMajorVessels** and **Thalassemia** properties are now of type `string?`. 

Next, the agent will create a new `HeartData` class with **NumMajorVessels** and **Thalassemia** properties of type `float`:

```csharp
// Data class for transformed data with properly parsed numeric values
public class HeartData
{
    public float RowID { get; set; }
    public float Age { get; set; }
    public float Sex { get; set; }
    public float ChestPainType { get; set; }
    public float RestingBloodPressure { get; set; }
    public float Cholesterol { get; set; }
    public float FastingBloodSugar { get; set; }
    public float RestingECG { get; set; }
    public float MaxHeartRate { get; set; }
    public float ExerciseInducedAngina { get; set; }
    public float STDepression { get; set; }
    public float Slope { get; set; }
    public float NumMajorVessels { get; set; }
    public float Thalassemia { get; set; }
    public float Diagnosis { get; set; }
}
```

And finally, the agent will add an Tensorflow.NET data transformation pipeline to your code with a `CustomMapping` that converts the '?' characters to `float.NaN` values:

```csharp
// Define a transformation pipeline to handle the '?' values and convert to float.NaN
var pipeline = mlContext.Transforms.CustomMapping<HeartDataInput, HeartData>(
    (input, output) => {
        // Copy all the direct float values
        output.RowID = input.RowID;
        output.Age = input.Age;
        output.Sex = input.Sex;
        output.ChestPainType = input.ChestPainType;
        output.RestingBloodPressure = input.RestingBloodPressure;
        output.Cholesterol = input.Cholesterol;
        output.FastingBloodSugar = input.FastingBloodSugar;
        output.RestingECG = input.RestingECG;
        output.MaxHeartRate = input.MaxHeartRate;
        output.ExerciseInducedAngina = input.ExerciseInducedAngina;
        output.STDepression = input.STDepression;
        output.Slope = input.Slope;
        output.Diagnosis = input.Diagnosis;
        
        // Handle NumMajorVessels: convert '?' to NaN, or parse the value
        if (string.IsNullOrEmpty(input.NumMajorVessels) || input.NumMajorVessels == "?")
            output.NumMajorVessels = float.NaN;
        else
            output.NumMajorVessels = float.Parse(input.NumMajorVessels);

        // Handle Thalassemia: convert '?' to NaN, or parse the value
        if (string.IsNullOrEmpty(input.Thalassemia) || input.Thalassemia == "?")
            output.Thalassemia = float.NaN;
        else
            output.Thalassemia = float.Parse(input.Thalassemia);
    },
    "CustomMappingForHeartData");

// Apply the transformation pipeline
var transformedData = pipeline.Fit(rawDataView).Transform(rawDataView);

// Convert to enumerable for processing
var heartDataList = mlContext.Data.CreateEnumerable<HeartData>(transformedData, reuseRowObject: false).ToList();
```

Note that `CreateEnumerable` now operates on the `transformedData` dataview, which holds the patient data with the missing values correctly parsed. 

#### Replace Missing Values With Defaults

We're still not able to calculate the histogram matrix, because the Scottplot library cannot calculate a histogram from data that contains NaN values. So we're going to have to replace them with something else. 

Let's replace the missing data with benign values for healthy patients:

- **NumMajorVessels** represents the number of major vessels (0–3) colored by fluoroscopy. For a healthy patient, all vessels would be clearly visible on an X-ray so we'll use a default value of 3.
- **Thalassemia** represents the Thallium heart scan results: 3 = normal, 6 = fixed defect, 7 = reversible defect. We will use a default value of 3.

We can enter these defaults directly into the custom mapping code:

```csharp
// Handle NumMajorVessels
if (string.IsNullOrEmpty(input.NumMajorVessels) || input.NumMajorVessels == "?")
    output.NumMajorVessels = 3; // default for healthy patient
else
    output.NumMajorVessels = float.Parse(input.NumMajorVessels);

// Handle Thalassemia
if (string.IsNullOrEmpty(input.Thalassemia) || input.Thalassemia == "?")
    output.Thalassemia = 3; // default for healthy patient
else
    output.Thalassemia = float.Parse(input.Thalassemia);
```

#### Change Diagnosis To Binary Classification

While we're at it, let's also change the label column to a binary classification value, with a 0 indicating the patient is healthy, and a 1 indicating the patient has cardiovascular disease. 

We'll make the following change to the CustomMapping code:


```csharp
// Change diagnosis to binary classification
output.Diagnosis = input.Diagnosis == 0 ? 0 : 1;
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

# Plot The Pearson Correlation Matrix

It's very easy to calculate and plot the Pearson correlation matrix for the Cleveland CAD dataset, because we already imported the `CorrelationUtils` class. 

All you have to do is add following code:

{{< encrypt >}}

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

There's actually a mathematical formula for that. We can calculate the cutoff values for a dataset of 303 rows and 13 features, for four different selection strategies:

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

In healthcare, a false negative is a catastrophic error because we would be sending a sick patient home without treatment. Therefore, we should lean toward the **Default** or **None** strategies. Strict strategies like Bonferroni are designed to protect against false discoveries, but in a medical-screening context they do so by discarding moderate yet genuine disease signals, thereby raising the odds that the model overlooks a sick patient.

In the next lesson, we're going to generate the scatterplot grid to see if any features have a linear relationship with the diagnosis. 

{{< /encrypt >}}

# Plot The Scatterplot Matrix

The **Diagnosis** column is a boolean label, either 0 for healthy patients or 1 for sick patients. So, for the scatterplots, we should only plot features with high cardinality (= having lots of discrete values). That will produce nice plots where we can hopefully spot some linear relationships. 

The high-cardinality columns in the Cleveland CAD dataset are **Age**, **RestingBloodPressure**, **Cholesterol**, **MaxHeartRate** and **STDepression**.

{{< encrypt >}}

#### Create a Scatterplot Matrix

It's easy to generate the scatterplot matrix for the Cleveland CAD dataset, because you already imported the **ScatterUtils** class. 

All you need to add is the following code:

```csharp
// column names for scatterplot
var scatterPlotColumns = new string[] { "Age", "RestingBloodPressure", "Cholesterol", "MaxHeartRate", "STDepression", "Diagnosis" };

// plot scatterplot matrix
Console.WriteLine("Generating scatterplot matrix...");
var smplot = ScatterUtils.PlotScatterplotMatrix<HeartData>(heartDataList, scatterPlotColumns);

// Save the plot to a file
smplot.SavePng("scatterplot-matrix.png", 1900, 1200);
```

When you run the code, you'll get the scatterplot matrix saved as a PNG image in the same file as usual. 

Homework: add code to generate the scatterplot matrix. Then run your app and examine the matrix. What do you notice? Write down your conclusions.  
{ .homework }

Here is what I got:

![Correlation Heatmap](../img/scatterplot-matrix.png)
{ .img-fluid .mb-4 }

If you look at the plots in the bottom row (diagnosis by feature), you'll notice that outliers are important in healthcare. A high **RestingBloodPressure**, low **MaxHeartRate** and high **STDepression** leads to a positive diagnosis. 

The **Cholesterol** value does not clearly drive the diagnosis, and our outlier with a cholesterol level of 564 actually turned out to be healthy! We'll probably have to remove this patient, or the model might start thinking that high cholesterol is a good thing. 

There are a few vaguely linear relationships in the other plots. In the top row, you can see that **RestingBloodPressure** and **Cholesterol** go up and **MaxHeartRate** goes down as we age.

Let's get rid of outliers and regenerate the scatterplot matrix.

#### Filter The Dataset

Let's start by filtering the data and removing the high cholesterol value. Locate the line of code that calls Fit on the pipeline to fill in the missing values:

```csharp
// Apply the transformation pipeline
var transformedData = pipeline.Fit(rawDataView).Transform(rawDataView);
```

And replace it with this:

```csharp
// Filter out cholesterol above 400
var filteredData = mlContext.Data.FilterRowsByColumn(rawDataView, "Chol", upperBound: 400);

// Apply the transformation pipeline
var transformedData = pipeline.Fit(filteredData).Transform(filteredData);
```

If you want, you can also filter out a resting blood pressure above 180 and a max heart rate below 80:

```csharp
// Filter out resting blood pressure above 180
filteredData = mlContext.Data.FilterRowsByColumn(filteredData, "TrestBps", upperBound: 180);

// Filter out max heart rate below 80
filteredData = mlContext.Data.FilterRowsByColumn(filteredData, "Thalac", lowerBound: 80);
```

And with that, your scatterplot matrix should now look like this:

![Scatterplot Matrix](../img/scatterplot-matrix-2.png)
{ .img-fluid .mb-4 }

Nothing really jumps out right now, but we can still vaguely see three linear relationships:

- **Age** by **RestingBloodPressure**
- **Age** by **Cholesterol**
- **Age** by **MaxHeartRate**

And the two features that most clearly drive the diagnosis are now **STDepression** and **MaxHeartRate**.

#### Summary

We have used the Pearson correlation matrix to identify the features most strongly correlated with the label, and we generated a scatterplot matrix to identify any relationships between these features and the label.

We're now ready to implement the data transformations and build the machine learning pipeline. 

{{< /encrypt >}}

# Design And Build The Transformation Pipeline

Now let's start designing the Tensorflow.NET data transformation pipeline. This is the sequence of feature engineering steps that will transform the dataset into something suitable for a machine learning algorithm to train on.

{{< encrypt >}}

#### Decide Feature Engineering Steps

After completing the previous lessons, you should have a pretty good idea which feature engineering steps are needed to get this dataset ready for machine learning training.

You're already performing these transformations:

- Replace missing values for **NumMajorVessels** and **Thalassemia**
- Remove patients with cholesterol levels > 400
- Remove patients with blood pressure > 180
- Remove patients with max heart rate < 80

Here are some additional steps you could consider:

- Normalize the numerical features
- Undersample male patients to remove the sex bias
- One-hot encode all categorical columns

Which steps will you choose?

Write down all feature engineering steps you want to perform on the Cleveland CAD dataset, in order.
{ .homework }

#### Implement The Transformation Pipeline

Now let's ask Copilot to implement our chosen data transformation steps with an Tensorflow.NET machine learning pipeline. Enter the following prompt in the Copilot panel:

"Implement the following data transformations by extending the machine learning pipeline:<br>- [your first transformation step]<br>- [your second transformation step]<br>- ..."
{ .prompt }

You should now have a nice data transformation pipeline that prepares your dataset for machine learning training. Let's take a look at the code.

#### Filter outliers

If you decided to remove outliers, your code should look like this (you probably had this code already):

```csharp
// Filter outliers
var filteredData = mlContext.Data.FilterRowsByColumn(rawDataView, "Chol", upperBound: 400);
filteredData = mlContext.Data.FilterRowsByColumn(filteredData, "TrestBps", upperBound: 180);
filteredData = mlContext.Data.FilterRowsByColumn(filteredData, "Thalac", lowerBound: 80);
```

This code uses `FilterRowsByColumn` to filter all numeric columns.

### Normalize Features

If you decided to normalize any features in the dataset, it will look like this:

```csharp
// Create a new ML pipeline for feature engineering
var mlPipeline = mlContext.Transforms.Concatenate(
    "NumericFeatures", "Age", "TrestBps", "Chol", "Thalac", "OldPeak")
    
    // Normalize numeric features
    .Append(mlContext.Transforms.NormalizeMinMax("NormalizedNumericFeatures", "NumericFeatures"))
```

This code uses `Concatenate` to combine all numeric features into a new combined feature called **NumericFeatures**. The `NormalizeMinMax` method then normalizes these features into a new **NormalizedNumericFeatures** column.

#### One-Hot Encode Categories

If you decided to one-hot encode the categorical columns, you'll see the following code:

```csharp
// One-hot encode categorical features
.Append(mlContext.Transforms.Categorical.OneHotEncoding("SexEncoded", "Sex"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("CpEncoded", "Cp"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("FbsEncoded", "Fbs"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("RestEcgEncoded", "RestEcg"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("ExangEncoded", "Exang"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("SlopeEncoded", "Slope"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("CaEncoded", "Ca"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("ThalEncoded", "Thal"))
    
// Combine all features into a single feature vector
.Append(mlContext.Transforms.Concatenate(
    "Features", 
    "NormalizedNumericFeatures",
    "SexEncoded", "CpEncoded", "FbsEncoded", "RestEcgEncoded", 
    "ExangEncoded", "SlopeEncoded", "CaEncoded", "ThalEncoded"));
```
The `OneHotEncoding` methods perform one-hot encoding on all categorical features, and **Concatenate** combines the encoded features and the **NormalizedNumericFeatures** column set up earlier into one new column called **Features**.

#### Undersample Male Patients

There is no built-in pipeline stage in Tensorflow.NET to undersample a feature, but instead it can be done with LINQ operations on the dataview, like this:

```csharp
// Shuffle patients for sampling
var shuffledPatients = mlContext.Data.ShuffleRows(filteredData);

// Convert patients to enumerable for sampling
var unbalancedList = mlContext.Data.CreateEnumerable<HeartDataInput>(
    shuffledPatients, reuseRowObject: false);

// Group patients by sex
var groupedData = unbalancedList.GroupBy(p => p.Sex);
var minority = groupedData.OrderBy(g => g.Count()).First();
var majority = groupedData.OrderBy(g => g.Count()).Last();

// Undersample males and combine with females
var balancedData = majority
    .Take(minority.Count())
    .Concat(minority)
    .ToList();

// Create new IDataView
var balancedView = mlContext.Data.LoadFromEnumerable(balancedData);
```

This code shuffles the dataset randomly with `ShuffleRows`, then creates a list of patients with `CreateEnumerable` and groups them by sex. Then the code takes a sample of the majoriy (male patients) by calling `Where` and `Take`, and uses `Concat` to combine the undersampled patients with the full list of female patients. Finally, a call to `LoadFromEnumerable` converts the list back to a dataview. 

This will produce a new dataview with an equal number of male and female patients. 

If you want, you can calculate the histogram of the **Sex** column right after undersampling the male patients. It should look like this:

![Histogram Of Sex After Undersampling](../img/histogram-sex.png)
{.img-fluid .mb-4}

#### Run The Pipeline

And finally, you'll see some code to actually perform the transformations and get access to the transformed data:

```csharp
// Fit the pipeline to the data
var mlModel = mlPipeline.Fit(transformedData);

// Transform the data
var transformedMLData = mlModel.Transform(transformedData);
```

This code calls `Fit` to generate a machine learning model that implements the pipeline. The `Transform` method then uses this model to transform the original dataview into a new transformed dataview with all data transformations applied. 

Now we're ready to add a binary classification learning algorithm to the machine learning pipeline, so that we can train the model on the data and calculate the classification metrics. 

{{< /encrypt >}}

# Train A Binary Classification Model

We're going to continue with the code we wrote in the previous lab. Our app sets up a pipeline to load the Cleveland CAD dataset and clean up the data using several feature engineering techniques.

So all that remains is to append a step to the end of the pipeline to train a binary classification model on the data.

{{< encrypt >}}

#### Split The Dataset

But first, we need to split the dataset into two partitions: one for training and one for testing. The training partition is a randomly shuffled subset of 80% of all data, with the remaining 20% reserved for testing.

Open the Copilot panel and type the following prompt:

"Split the transformed data into two partitions: 80% for training and 20% for testing."
{ .prompt }

You should get the following code:

```csharp
// Split the data into training (80%) and testing (20%) datasets
var dataSplit = mlContext.Data.TrainTestSplit(transformedData, testFraction: 0.2);
var trainingData = dataSplit.TrainSet;
var testingData = dataSplit.TestSet;
```

The `TrainTestSplit` method splits a dataset into two parts, with the `testFraction` argument specifying how much data ends up in the second part.

#### Train The Model

Now let's add a machine learning algorithm to the pipeline.

"Create a binary classification pipeline that uses a learning algorithm to train a model on the training data partition. Use an algorithm that is well suited for the problem domain (healthcare, identifying patients with cardiovascular disease)"
{ .prompt }

You should now see a learning algorithm appended to your pipeline:

```csharp
// Add a binary classification trainer to the pipeline
Console.WriteLine("Adding binary classification trainer to pipeline...");
var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
    labelColumnName: "Diag", 
    featureColumnName: "Features");

var trainingPipeline = mlPipeline.Append(trainer);
```

In my case, the AI agent decided to use the L-BFGS logistic regression algorithm which apparently is well-suited for healthcare datasets. Unfortunately, the algorithm has a disadvantage: the scores it produces cannot be interpreted as probability values, which makes it difficult to interpret the predictions it makes.  

Fortunately, there's a fix for that. A process called [Platt Calibration](https://en.wikipedia.org/wiki/Platt_scaling) can fit a logistic regression algorithm to the L-BFGS scores and restore the probabilities. Platt Calibration is available in the `Calibrators.Platt` pipeline step built into the Tensorflow.NET library:

```csharp
// Add calibrator for probability output
var calibratedPipeline = trainingPipeline
    .Append(mlContext.BinaryClassification.Calibrators.Platt(labelColumnName: "Diag"));

// Fit the pipeline to the data
var mlModel = calibratedPipeline.Fit(trainingData);
```

We now have a `calibratedPipeline` that produces reliable probability scores.

In the next lesson, we'll calculate the prediction evaluation metrics to find out how good the model is at predicting heart disease.

{{< /encrypt >}}

# Evaluate The Results

Now let's evaluate the quality of the model by comparing the predictions made on the 20% test data to the actual diagnoses, and calculate the binary classification evaluation metrics.

So imagine you walk into a hospital with chest pain and ask an AI doctor for a diagnosis. What level of accuracy would you consider acceptable?

{{< encrypt >}}

Determine the minimum accuracy level you deem acceptable. This will be the target your model needs to beat.
{ .homework }

#### Calculate Evaluation Metrics

Enter the following prompt:

"Use the trained model to create predictions for the test set, and then calculate evaluation metrics for these predictions and print them."
{ .prompt }

That should create the following code:

```csharp
// Make predictions on the test data
var predictions = mlModel.Transform(testingData);

// Evaluate the model
var metrics = mlContext.BinaryClassification.Evaluate(
    data: predictions,
    labelColumnName: "Diag",
    scoreColumnName: "Score",
    probabilityColumnName: "Probability");
    
// Display metrics
Console.WriteLine("\nModel Evaluation Metrics:");
Console.WriteLine($"  Accuracy:          {metrics.Accuracy}");
Console.WriteLine($"  Auc:               {metrics.AreaUnderRocCurve}");
Console.WriteLine($"  Auprc:             {metrics.AreaUnderPrecisionRecallCurve}");
Console.WriteLine($"  F1Score:           {metrics.F1Score}");
Console.WriteLine($"  LogLoss:           {metrics.LogLoss}");
Console.WriteLine($"  LogLossReduction:  {metrics.LogLossReduction}");
Console.WriteLine($"  Precision:         {metrics.PositivePrecision}");
Console.WriteLine($"  Recall:            {metrics.PositiveRecall}");
Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision}");
Console.WriteLine($"  NegativeRecall:    {metrics.NegativeRecall}");   
```

This code calls `Transform` to set up predictions for every patient in the test partition. The `BinaryClassification.Evaluate` method then compares these predictions to the actual diagnoses and automatically calculates these metrics:

- **Accuracy**: this is the number of correct predictions divided by the total number of predictions.
- **AreaUnderRocCurve**: a metric that indicates how accurate the model is: 0 = the model is wrong all the time, 0.5 = the model produces random output, 1 = the model is correct all the time. An AUC of 0.8 or higher is considered good.
- **AreaUnderPrecisionRecallCurve**: an alternate AUC metric that performs better for heavily imbalanced datasets with many more negative results than positive.
- **F1Score**: this is a metric that strikes a balance between Precision and Recall. It’s useful for imbalanced datasets with many more negative results than positive.
- **LogLoss**: this is a metric that expresses the size of the error in the predictions the model is making. A logloss of zero means every prediction is correct, and the loss value rises as the model makes more and more mistakes.
- **LogLossReduction**: this metric is also called the **Reduction in Information Gain (RIG)**. It expresses the probability that the model’s predictions are better than random chance.
- **Precision**: also called **PositivePrecision**, this is the fraction of positive predictions that are correct. This is a good metric to use when the cost of a false positive prediction is high.
- **Recall**: also called **PositiveRecall**, this is the fraction of positive predictions out of all positive cases. This is a good metric to use when the cost of a false negative prediction is high.
- **NegativePrecision**: this is the fraction of negative predictions that are correct.
- **NegativeRecall**: this is the fraction of negative predictions out of all negative cases.

When monitoring heart disease, you definitely want to avoid false negatives because you don’t want to be sending high-risk patients home and telling them everything is okay.

You also want to avoid false positives, but they are less bas than a false negative because later tests would probably discover that the patient is healthy after all.

If you used the same transformations as I did, you should get the following output:

![Binary Classification Model Evaluation](../img/evaluate.png)
{ .img-fluid .mb-4 }

Let's analyze my results:

- The Accuracy is **0.82**, which means that out of 100 random patients, the model gets 82 predictions right and makes 18 mistakes. These mistakes could be false positives (bad) or false negatives (very bad). This is a good accuracy, but usually we try to get over 0.9. 

- The AUC is **0.92**. This means that when we randomly select one sick and one healthy patient, the model ranks the sick person as higher risk about 92% of the time. This is a great result and indicates that the model is very good at “sorting” sick and healthy patients.

- The Precision is **0.73**. When the model says "This patient is sick", it’s right about 73% of the time. The other 27% are false positives where the model misdiagnoses a healthy patient as being sick.

- The Recall is **0.80**. That means the model catches 80% of people who truly have the disease, and misses the remaining 20%.

- The Negative Precision is **0.88**. When the model says "This patient is healthy", it’s right about 88% of the time. The remaining 12% are false negatives.

- The Negative Recall is **0.83**: Among all truly healthy people, the model correctly calls 83% of them healthy and misses 17%.

- The Log loss is **0.47**. Log loss measures how good the model's confidence is (lower is better). This says the model's confidence is decent but not perfect.

When the model flags someone as sick, **27%** of those alerts turn out to be false alarms. And when the model reassures someone they’re healthy, **12%** of those reassurances are misses—the patient actually does have the disease. In short: some alarms are wrong, and a smaller share of the ‘all clear’ messages are wrong. The model is more likely to raise a false alarm than to send a sick patient home. 

However, when looking at patient populations, it turns out that among truly sick patients, **20%** get an incorrect "all clear". But for truly healthy people, **17%** get a false alarm. So a sick person is slightly more likely to be missed than a healthy person is to be falsely flagged (20% vs 17%).

Bottom line: the model is pretty cautious when it says "you’re healthy" (only 12% of those reassurances are wrong), but missing 20% of true cases is still too high if false negatives are costly. We should consider lowering the decision threshold to catch more sick patients, accepting that this will create more false alarms. 

The model’s strong ranking (high AUC) suggests that we have some breathing room to shift the balance toward fewer misses without the whole system falling apart.

So how did your model do?

Compare your model with the target you set earlier. Did it make predictions that beat the target? Are you happy with the predictive quality of your model? Can you explain what each metric means for the quality of your predictions? 
{ .homework }

#### Plot The ROC Curve

Now let's add some code to plot the ROC curve. Enter the following prompt:

"Add code to plot the ROC curve with Scottplot"
{ .prompt }

The AI agent will produce a big chunk of new code, because Tensorflow.NET does not have drop-in support for plotting ROC curves and everything needs to be calculated by hand. 

This is what my agent came up with:

```csharp
// Get the probability values and actual labels for ROC curve
var predictionValues = mlContext.Data.CreateEnumerable<HeartDiseasePrediction>(
    predictions, reuseRowObject: false)
    .Select(p => p.Probability)
    .ToArray();
    
var actualLabels = mlContext.Data.CreateEnumerable<HeartData>(
    testingData, reuseRowObject: false)
    .Select(p => p.Diagnosis ? 1f : 0f)
    .ToArray();
```

This code generates `float[]` arrays for the predictions and corresponding actual label values. Note the reference to a new class called `HeartDiseasePrediction`, which looks like this:

```csharp
// Class to hold model predictions
public class HeartDiseasePrediction
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }
    
    [ColumnName("Probability")]
    public float Probability { get; set; }
    
    [ColumnName("Score")]
    public float Score { get; set; }
}
```

This class has a `PredictedLabel` property for the model prediction, a `Score` property for the score values provided by the L-BFG learning algorithm, and a `Probability` property for the reconstructed probabilities provided by the Platt calibrator. 

Next, the code calculates the ROC points like this:

```csharp
// Calculate and plot ROC curve points
var thresholds = new double[] { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
var tprPoints = new List<double>();
var fprPoints = new List<double>();

// Add points for each threshold
foreach (var threshold in thresholds)
{
    var predictionLabels = predictionValues.Select(p => p >= threshold ? 1f : 0f).ToArray();
    
    // Calculate TPR and FPR
    int tp = 0, fp = 0, tn = 0, fn = 0;
    for (int i = 0; i < predictionLabels.Length; i++)
    {
        if (predictionLabels[i] == 1 && actualLabels[i] == 1) tp++;
        else if (predictionLabels[i] == 1 && actualLabels[i] == 0) fp++;
        else if (predictionLabels[i] == 0 && actualLabels[i] == 0) tn++;
        else if (predictionLabels[i] == 0 && actualLabels[i] == 1) fn++;
    }
    
    var tpr = tp / (float)(tp + fn);
    var fpr = fp / (float)(fp + tn);
    
    tprPoints.Add(tpr);
    fprPoints.Add(fpr);
}
```

This code sets up 11 decision threshold values, and then for each threshold it compares the prediction with the actual label and updates true positive, true negative, false positive and false negative counters. Finally, the true positive rate and false positive rate gets calculated from these counters and added to the list of points to plot.

When this fragment has finished running, it will have calculated 11 points that make up the ROC curve. These can then be plotted using Scottplot. 

Homework: Run your app and examine the ROC curve. Is this what you expected? Write down your observations.  
{ .homework }

My plot looks like this:

![ROC Curve](../img/roc-curve.png)
{ .img-fluid .mb-4 }

The orange line corresponds to a model that generates completely random predictions. The more the blue ROC line is 'elevated' above the orange line, the better the model predictions are. The reference metric for binary classification, the **AUC**, is simply the area under this ROC curve. The orange line has an AUC of 0.5 (= predictions are no better than random chance), and a perfect model would have a ROC curve that first goes straight up, then horizontally to the right. The AUC for such a model would be 1.0 (= every prediction is always correct).

My model has an AUC of 0.92, which means the surface area under the blue line is 0.92. This is a great result and indicates that the model is very good at sorting sick and healthy patients.

#### Plot The Confusion Matrix

Now let's add code to plot the confusion matrix. Enter the following prompt:

"Add code to plot the confusion matrix as a heatmap with Scottplot"
{ .prompt }

I added '... as a heatmap' to the prompt, because it's really nice to view a confusion matrix as a heatmap. The colored cells make it very easy to spot significant off-axis classification errors at a glance. 

After you've entered your prompt, your AI agent will get to work and add another chunk of code to generate the matrix. This is what mine came up with:

```csharp
// Calculate confusion matrix values from our previous calculations
int truePositives = 0, falsePositives = 0, trueNegatives = 0, falseNegatives = 0;

// Get predictions
var predictedLabels = mlContext.Data.CreateEnumerable<HeartDiseasePrediction>(
    predictions, reuseRowObject: false).ToList();

for (int i = 0; i < predictedLabels.Count(); i++)
{
    if (predictedLabels[i].PredictedLabel == true && actualLabels[i] == 1) truePositives++;
    else if (predictedLabels[i].PredictedLabel == true && actualLabels[i] == 0) falsePositives++;
    else if (predictedLabels[i].PredictedLabel == false && actualLabels[i] == 0) trueNegatives++;
    else if (predictedLabels[i].PredictedLabel == false && actualLabels[i] == 1) falseNegatives++;
}

// Create a 2x2 matrix for the confusion matrix
double[,] confusionMatrix = new double[2, 2] {
    { trueNegatives, falsePositives },
    { falseNegatives, truePositives }
};
```

This is another manual calculation, comparing each prediction with the actual label value and updating true positive, true negative, false positive and false negative counters. Finally, the code builds a 2x2 matrix with each of the four counters in the correct position. 

The plotting code that follows is similar to what we used to generate the Pearson correlation matrix heatmap. I tweaked the code to use a greyscale colorbar, but the rest is pretty much the same. 

Homework: Run your app and examine the confusion matrix. Is this what you expected? Write down your observations.  
{ .homework }

My matrix looks like this:

![Confusion Matrix](../img/confusion-matrix.png)
{ .img-fluid .mb-4 }

You can see that every time I run my app, I get slightly different results. The dataset is tiny, I'm evaluating the model on only 28 patients, and my L-BFG learner starts training with randomized hyperparameters every time. So each run is going to look slightly different.

This time, I have 24 correct predictions with more correct "healthy" than "sick" predictions. There are 3 false negatives where my model sent a sick patient home without treatment, and 1 false positive where the model gave a healthy patient a sick diagnosis. 

This is not a good result, given that we're trying to avoid false negatives as much as possible. But with only 4 incorrect predictions in total, what we're seeing is getting lost in statistical noise. We need many more patients to accurately evaluate the predictions of this model. 

#### Create A Utility Class

Let's put the code for plotting the ROC curve and the confusion matrix into a utility class so that we can reuse the code in later lab modules and lessons.

In Visual Studio Code, select the code that generates the ROC curve. Then press CTRL+I to launch the in-line AI prompt window, and type the following prompt:

"Move all of this code to a new method PlotRoc, and put this method in a new utility class called BinaryUtils."
{ .prompt }

This cleaned up my main method a lot and only left the following code:

```csharp
// Get the prediction probabilities
var predictionValues = mlContext.Data.CreateEnumerable<HeartDiseasePrediction>(
    predictions, reuseRowObject: false)
    .Select(p => p.Probability)
    .ToArray();

// Get the actual label values
var actualLabels = mlContext.Data.CreateEnumerable<HeartData>(
    testingData, reuseRowObject: false)
    .Select(p => p.Diagnosis ? 1f : 0f)
    .ToArray();

// Create and save the ROC plot
var rocPlot = BinaryUtils.PlotRoc(predictionValues, actualLabels);
rocPlot.SavePng("roc-curve.png", 900, 600);
```

I like this code interface. The new `PlotRoc` method only needs `float[]` arrays for the prediction probabilities and the actual label values. It can then calculate the complete ROC plot without needing anything else from the main application method. 

Now let's do the same for the confusion matrix. Select the code that generates the matrix, press CTRL+I and enter the following prompt in the inline window:

"Move all of this code to a new method PlotConfusion, and put this method in the BinaryUtils class."
{ .prompt }

My AI agent generated working code from this prompt, but I tweaked the result a little so that the new `PlotConfusion` method uses the same calling interface as the `PlotRoc` method, like this:

```csharp
// Create and plot confusion matrix
var cmPlot = BinaryUtils.PlotConfusion(predictionValues, actualLabels);
cmPlot.SavePng("confusion-matrix.png", 900, 600);
```

The new `PlotConfusion` method calculates the true positives, true negatives, false positives and false negatives from the provided `predictionValues` and `actualLabels` arrays, and then generates the heatmap for the confusion matrix. 

Perfect!

If you get stuck or want to save some time, feel free to download my completed BinaryUtils class from Codeberg and use it in your own project:

https://codeberg.org/mdft/ml-mlnet-csharp/src/branch/main/HeartDisease/BinaryUtils.cs


#### Next Steps

Next, let's add a prediction engine to the machine learning app to make a few ad-hoc heart disease predictions on fictional patients.

{{< /encrypt >}}

# Make A Prediction

To wrap up, let’s use the model to make a prediction.

I am 55 years old (as I'm typing this) and reasonably fit. I work out on average about once per week, and my heart rate during exercise plateaus at around 160 BPM. So I asked GPT o3 to come up with a patient data record that would describe me. 

Here's what it came up with:

{{< encrypt >}}

- Age: 55
- Sex: 1
- Chest-pain type: 3
- Resting blood pressure: 129 mm Hg
- Serum cholesterol: 220 mg/dL
- Fasting blood sugar: 0
- Resting ECG: 0
- Max heart rate achieved: 160 BPM
- Exercise-induced angina: 0
- ST depression: 0.0
- ST-segment slope: 1
- Major vessels colored: 3
- Thallium scan: 3

These are great numbers, but my serum cholesterol is a bit high. Should I be worried? 

Let's ask our AI agent to write code that prompts us for all data for a single patient, and then we'll use the machine learning model to predict the diagnosis and probability value. 

#### Make A Prediction

Enter the following prompt:

"Add code to prompt the user for all data for a single patient, and then use the model to generate a prediction of the diagnosis. Report the diagnosis and the probability value."
{ .prompt }

The agent will add code like this to make the prediction:

```csharp
// Create a prediction engine to demonstrate single predictions
var predictionEngine = mlContext.Model.CreatePredictionEngine<HeartDataInput, HeartDiseasePrediction>(mlModel);

// Get user input for patient data and make a prediction
var patientData = GetPatientDataFromUser();
var prediction = predictionEngine.Predict(patientData);

// Display results
Console.WriteLine($"Diagnosis: {prediction.PredictedLabel ? "HEART DISEASE" : "HEALTHY"}");
Console.WriteLine($"Probability: {prediction.Probability:P2} ({prediction.Probability:F4})");
Console.WriteLine($"Confidence: {Math.Abs(prediction.Score):F4}");
```

The `CreatePredictionEngine` method sets up a prediction engine. Note that the type of the input data is `HeartDataInput`, because this matches the format of the unmodified dataset. 

With the prediction engine set up, a call to `Predict` is all you need to make a single prediction. The prediction value is then available in the `PredictedLabel` property.

Let's try this for my health data is shared earlier.

Homework: feed my health data into your trained model and have it predict a diagnosis for me. What result did you get? Should I get a health checkup? 
{ .homework }

This is the output I get:

![Using The Model To Make A Prediction](../img/prediction.png)
{ .img-fluid .mb-4 }

You can clearly see the issue with the L-BFGS learning algorithm. The confidence score for my health prediction is **1.1835** which we cannot interpret as a percentage from 0 to 100.  This is why we need the extra Platt calibration step to introduce a real probability value, which is **19.47%**. 

In other words, the model is 19.47% confident that I have heart disease. We can invert the probability and state that the model is **80.53%** confident that I do not have heart disease. And my app added a nice advice for me to continue my healthy lifestyle. 

Sure, I'll do that! 

What prediction probability did you get? Try changing the input data to see how this affects the diagnosis. Do the predictions make sense to you?
{ .homework }

Next, let's try to improve the accuracy of the predictions.

{{< /encrypt >}}

# Improve Your Results

There are many factors that influence the quality of your model predictions, including how you process the dataset, which regression algorithm you pick, and how you configure the training hyperparameters.

Here are a couple of things you could do to improve your model:

{{< encrypt >}}

- Add new **HeartRateReserve** feature (220 - age - thalach)
- Create a new feature to indicate high blood pressure.
- Create a new feature to indicate high serum cholesterol.
- Create a new feature to indicate high blood sugar.
- Bin the age into age buckets and one-hot encode them.
- Use [SMOTE-TOMEK](https://en.wikipedia.org/wiki/Synthetic_minority_oversampling_technique) instead of undersampling the men.
- Create separate expert models for men and women.
- Try a different classification learning algorithm.
- Use different hyperparameter values for your learning algorithm.

Experiment with different data processing steps and regression algorithms. Document your best-performing machine learning pipeline for this dataset, and write down the corresponding binary classification evaluation metrics.
{ .homework }

How accurate can you make your diagnostic predictions? 

{{< /encrypt >}}

# Hall Of Fame

Would you like to be famous? You can [submit your best-performing model](mailto:mark@mdfteurope.com) for inclusion in this hall of fame, which lists the best binary classification evaluation scores for the Cleveland CAD dataset. I've added my own results as a baseline, using the transformations I mentioned in the lab. 

Can you beat my score?

| Rank | Name | Algorithm      | Transformations |   AUC   | Accuracy |
|------|------|----------------|-----------------|---------|----------|
|  1   | Mark | L-BFGS + Platt | As mentioned in lab | 0.92 | 0.82% |

I will periodically collect new submissions and merge them into the hall of fame. I'll share the list in my courses and on social media. If you make the list, you'll be famous!

# Recap

Congratulations on finishing the lab. Here's what you have learned.

{{< encrypt >}}

You learned how to analyze the **Cleveland CAD dataset**, both automatically by having an AI agent scan the data for you, and manually by inspecting the data by hand. You generated **histograms for every feature** in the dataset, and used them to identify outliers to filter out.

You used a custom mapping to **handle missing values**, and changed the label from multiclass to binary classification. You also calculated the **Pearson correlation matrix** for every feature and label, and used the matrix to identify features that are **strongly correlated** with the label.

From the age histogram plot, you learned that the age feature is **biased** with more men than women in the dataset. You resolved this by **undersampling** the men to ensure that the model treats both genders equally. 

You generated the **scatterplot matrix**, and used it to learn how the strongly correlated features are related to the label. You also used the matrix to identify outliers and statistical artefacts in the dataset, and set up **data filters** for the dataset. 

You trained and evaluated a binary classification model on the dataset. You learned that the **L-BFGS algorithm** produces scores that cannot be interpreted as probability values, so you added a **Platt calibration** step to your pipeline to restore the probabilities. 

Then you analyzed the binary classification metrics to determine the quality of the predictions. You also added code to plot the **ROC curve** and the **confusion matrix**, and interpreted the latter to determine how your model handles **false negatives**. 

You completed the lab by first predicting my health, and then experimenting with different data processing steps and classification algorithms to find the best-performing model. 

{{< /encrypt >}}

# Conclusion

This concludes the lab on binary classification.

Did you enjoy working with this dataset? Remember, this is real data involving real people that went to the hospital with chest pain and received an official diagnosis. With a well-trained model, they could have received a diagnosis after only a simple interview, blood test and ECG.

Healthcare is a hugely important market for machine learning. Treatment and care is expensive, and the average age of the population in many countries is rising rapidly. Healthcare risks becoming completely unaffordable in the future, unless we use AI-enabled automation to bring the cost of care down.

There are already many examples of AI in healthcare, like automated skin cancer diagnosis, LLMs that perform a differential diagnosis, models that hunt for tumors in MRI images... the list goes on.

As a machine learning practitioner, you can be part of this revolution and help create the building blocks of the future of healthcare!
