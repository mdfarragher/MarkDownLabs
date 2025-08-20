---
title: "Test A Prediction"
type: "lesson"
layout: "default"
sortkey: 140
---

To wrap up, let’s use the model to test a prediction.

We are going to identify the class pair (predicted versus actual label) that the model struggles with the most, select one sample image from that pair, run a prediction and then show the image as ASCII art. That should give us a clue why the model struggles with that particular pair. 

Let's see if our AI agent can write all code in one go. Enter the following prompt:

{{< encrypt >}}

"Add code to identify the class pair (predicted versus actual label) that the model struggles with the most, select one sample record from that pair, then use the model to generate a prediction for that digit, and finally show the digit as ASCII art for comparison." 
{ .prompt }

#### Find The Most Popular Incorrect Prediction

The agent will add code like this to identify the class pair corresponding to the most popular incorrect prediction:

```csharp
// Get the number of classes
var confusionMatrix = metrics.ConfusionMatrix;
int numClasses = confusionMatrix.NumberOfClasses;

// Find the highest off-diagonal value (most confused pair)
int maxConfusedActual = -1;
int maxConfusedPredicted = -1;
double maxConfusionCount = 0;

for (int actual = 0; actual < numClasses; actual++)
{
    for (int predicted = 0; predicted < numClasses; predicted++)
    {
        if (actual != predicted) // Skip diagonal (correct predictions)
        {
            var predictionIndex = Array.IndexOf(classes, predicted);
            var actualIndex = Array.IndexOf(classes, actual);
            double count = confusionMatrix.GetCountForClassPair(predictionIndex, actualIndex);
            if (count > maxConfusionCount)
            {
                maxConfusionCount = count;
                maxConfusedActual = actual;
                maxConfusedPredicted = predicted;
            }
        }
    }
}
```

This is the same code as before. It uses `Array.IndexOf` to calculate the indices of every class pair, and `GetCountForClassPair` to find the number of predictions for that class pair. The pair for the worst-performing incorrect predictions gets stored in `maxConfusedActual` and `maxConfusedPredicted`.

#### Sample One Incorrect Prediction

Next, we have to select one sample prediction for the worst performing class pair. My agent came up with this, and you'll probably have something similar in your app:

```csharp
// Get predictions with actual labels for analysis
var predictionList = mlContext.Data.CreateEnumerable<MnistPrediction>(predictions, reuseRowObject: false).ToList();
var actualList = mlContext.Data.CreateEnumerable<MnistData>(testData, reuseRowObject: false).ToList();

// Find the first sample from the most confused pair
int sample = -1;
for (int i = 0; i < predictionList.Count; i++)
{
    var prediction = predictionList[i].PredictedLabel;
    var actual = actualList[i].Label;
    if (actual == maxConfusedActual && prediction == maxConfusedPredicted)
    {
        sample = i;
        break;
    }
}
```

This code calls `CreateEnumerable` to create lists of predicted and actual labels, and then searches the lists for a pair that matches the worst performing class pair identified earlier. The code simply grabs the first prediction it can find that matches the pair. 

#### Print The Results

Finally, we can report the outcome like this:

```csharp
Console.WriteLine($"\nFirst image that matches class pair");
Console.WriteLine($"Row ID: {actualList[sample].RowID}");
Console.WriteLine($"Actual Label: {actualList[sample].Label}");
Console.WriteLine($"Predicted Label: {predictionList[sample].PredictedLabel}");

// Show confidence scores for all classes
Console.WriteLine("\nConfidence scores for all classes:");
for (int i = 0; i < predictionList[sample].Score.Length; i++)
{
    var index = Array.IndexOf(classes, i);
    Console.WriteLine($"Class {i}: {predictionList[sample].Score[index]:P2}");
}

// Display the digit as ASCII art
Console.WriteLine($"\nASCII Art Visualization of Digit:");
VisualizeDigit(actualList[sample].PixelValues);
```

This code is quite nice, my AI agent went above and beyond to report everything noteworthy about the sample. I get the actual and predicted label, the confidence scores for all 10 classes and the digit drawn as ASCII art.

Note that the `Score` property with the array of confidence scores is also indexed by key index, so we need another `Array.IndexOf` call to match each score with the correct class value. 

Homework: Inspect your code and make sure you handle class values and key indices correctly everywhere. Then run your app and examine the output. Look at the actual and predicted label, the prediction scores and the ASCII art. Does the incorrect prediction make sense to you? 
{ .homework }

My output looks like this:

![Confusion Matrix](../img/prediction-1.png)
{ .img-fluid .mb-4 }

We already knew that the most popular incorrect prediction is where the model predicts a 9 but the digit was actually a 4. The code samples image 160 which has this exact same class pair. We can see the individual confidence values, with **55.23%** confidence for a '9' and **44.19%** confidence for a '4'. Those values are pretty close together, so the model simply cannot decide between the one or the other digit. 

This is what image 160 looks like:

![Confusion Matrix](../img/prediction-2.png)
{ .img-fluid .mb-4 }

To me this clearly looks like a '4', but I guess the horizontal stroke is too short to make this digit clearly look like the number four. And there's definitely enough sloppy handwriting in the MNIST dataset to confuse the model. 

{{< /encrypt >}}

