---
title: "Check For Class Imbalance"
type: "lesson"
layout: "default"
sortkey: 40
---

# Check For Class Imbalance

In the previous lab module, you checked for class imbalance in the **Sex** column and when you discovered that the Cleveland CAD dataset had many more male patients than female, you undersampled the men to restore the balance. 

Checking for class imbalance is even more important with multiclass classification datasets, because now we have many distinct label classes to predict, and we want to make sure we have the same number of records for each class. This will ensure that the fully trained model is unbiased.

{{< encrypt >}}

So in this section, you'll going to generate code to plot a histogram for the Label column, to check how often each digit appears in the dataset.

Let's get started.

#### Install Utility Classes And Dependencies

Let's see if Copilot can do the whole thing for us. We'll ask the agent to download the histogram utils class from the Git repository and install any required dependencies for us. 

You'll need the raw url of the utility file hosted in the repository where you've stored the code you created in the previous lab. We'll copy the url into the prompt and ask the agent to import the class directly into the current project.

I pushed my utility class to a repository on Codeberg. You can use my repository url in your prompt if you like:

"Copy the HistogramUtils module from this repository url and add it to the project. Install any required NuGet package dependencies to ensure that the code works: <br> - codeberg.org/mdft/ml-mlnet-fsharp/raw/branch/main/HeartDisease/HistogramUtils.fs"
{ .prompt } 

#### Load The Mnist-Handwriting.csv File

Now let's ask Copilot to write the code for loading the CSV file and generating a data class that represents one record from the file. 

Enter the following prompt:

"Write F# code to load the Mnist-Handwriting.csv file, using the LoadFromTextFile method in ML.NET. Also create a record type that represents one record from the file, that can be used with the CreateEnumerable method in ML.NET to create a list of image data records."
{ .prompt }

And let Copilot write the code for you.

You should see the following data loading code in your project:

```fsharp
// Create ML.NET context
let mlContext = MLContext()

// Load data from CSV file using LoadFromTextFile
let dataView = mlContext.Data.LoadFromTextFile<MnistData>(
    path = "Mnist-Handwriting.csv",
    hasHeader = true,
    separatorChar = ',')

// Create an enumerable list of MnistData records
let dataList = 
    mlContext.Data.CreateEnumerable<MnistData>(dataView, reuseRowObject = false)
    |> List.ofSeq
```

The `MnistData` record type looks super interesting:

```fsharp
open Microsoft.ML.Data

// Record type representing one MNIST handwriting record
[<CLIMutable>]
type MnistData = {
    [<LoadColumn(0)>]
    RowID: float32

    [<LoadColumn(1)>]
    Label: float32

    // Load all 784 pixel values (28x28 image)
    [<LoadColumn(2, 785)>]
    [<VectorType(784)>]
    PixelValues: float32[]
}
```

Note the `VectorType` attribute on the `PixelValues` field. Combined with the `LoadColumn` attribute, it specifies that columns 2 .. 785 of the datafile will be loaded into a single `float32[]` array with 784 elements. 

You can combine multiple columns into float32 arrays in ML.NET pipelines, and this is very convenient when we're working with image data. A 28x28 pixel image has 784 individual pixels, and you don't want to be tracking the column name of each of them individually. 

#### Generate The Histogram Of Labels

To generate the histogram of labels, we can simply do this:

```fsharp
// generate the histogram of labels
Console.WriteLine("Generating histograms of labels...")
let hist = HistogramUtils.PlotHistogram<MnistData>(dataList, "Label")

// save the histogram
hist.SavePng("histogram-labels.png", 800, 600)
```

That histogram utility class comes in really handy every time. 

Homework: add code to generate the histogram of labels. Then run your app and examine the plot. What can you say about any class imbalance in this dataset? Write down your conclusions.  
{ .homework }

Here's what I got:

![Histogram Grid For Full Dataset](../img/histogram-labels.png)
{.img-fluid .mb-4}

You can see that there's hardly any class imbalance in the dataset. We have a total of 10,000 images, and each class appears roughly 1,000 times. This is perfect. 

#### Bonus: Plot The First Digit As ASCII Art

When I prompted my AI agent to load the dataset, it also helpfully generated code to display the first handwritten digit as ASCII art on the console. You can copy my F# code if you want, or you can prompt your own agent to write the code for you. 

Here's what I got. My agent created a new function called `VisualizeDigit` that looks like this:

```fsharp
// Helper function to visualize a digit as ASCII art
let VisualizeDigit (pixels: float32[]) =
    for row in 0..27 do
        for col in 0..27 do
            let index = row * 28 + col
            let pixelValue = pixels.[index]
            
            // Convert pixel value to ASCII character
            let displayChar =
                if pixelValue = 0.0f then ' '
                elif pixelValue < 64.0f then '.'
                elif pixelValue < 128.0f then '+'
                elif pixelValue < 192.0f then '*'
                else '#'
            
            Console.Write(displayChar)
        Console.WriteLine()
```

This is cool! And in the main program method, it added the following code:

```fsharp
// Visualize a sample digit (optional)
Console.WriteLine("\nVisualizing first digit (28x28 ASCII art):")
let firstRecord = List.head dataList
VisualizeDigit(firstRecord.PixelValues)
```

So when I run my app, I see this:

![First Digit As ASCII Art](../img/first-digit.png)
{.img-fluid .mb-4}

You can see that the image is clearly a handwritten number 7, which is the first image in the MNIST dataset.

{{< /encrypt >}}