---
title: "Test A Prediction"
type: "lesson"
layout: "default"
sortkey: 70
---

# Test A Prediction

To wrap up, let's use the model to test a prediction.

We are going to identify the class pair (predicted versus actual label) that the model struggles with the most, select one sample image from that pair, run a prediction and then show the image as ASCII art. That should give us a clue why the model struggles with that particular pair. 

Let's see if our AI agent can write all code in one go. Enter the following prompt:

{{< encrypt >}}

"Add F# code to identify the class pair (predicted versus actual label) that the model struggles with the most, select one sample record from that pair, then use the model to generate a prediction for that digit, and finally show the digit as ASCII art for comparison." 
{ .prompt }

#### Find The Most Popular Incorrect Prediction

The agent will add code like this to identify the class pair corresponding to the most popular incorrect prediction:

```fsharp
// Get the number of classes
let confusionMatrix = metrics.ConfusionMatrix
let numClasses = confusionMatrix.NumberOfClasses

// Find the highest off-diagonal value (most confused pair)
let mutable maxConfusedActual = -1
let mutable maxConfusedPredicted = -1
let mutable maxConfusionCount = 0.0

for actual in 0 .. numClasses - 1 do
    for predicted in 0 .. numClasses - 1 do
        if actual <> predicted then // Skip diagonal (correct predictions)
            let predictionIndex = Array.IndexOf(classLabels, float32 predicted)
            let actualIndex = Array.IndexOf(classLabels, float32 actual)
            let count = float (confusionMatrix.GetCountForClassPair(predictionIndex, actualIndex))
            if count > maxConfusionCount then
                maxConfusionCount <- count
                maxConfusedActual <- actual
                maxConfusedPredicted <- predicted
```

This code uses `Array.IndexOf` to calculate the indices of every class pair, and `GetCountForClassPair` to find the number of predictions for that class pair. The pair for the worst-performing incorrect predictions gets stored in `maxConfusedActual` and `maxConfusedPredicted`.

#### Sample One Incorrect Prediction

Next, we have to select one sample prediction for the worst performing class pair. My agent came up with this, and you'll probably have something similar in your app:

```fsharp
// Get predictions with actual labels for analysis
let predictionList = 
    mlContext.Data.CreateEnumerable<MnistPrediction>(predictions, reuseRowObject = false) 
    |> List.ofSeq
let actualList = 
    mlContext.Data.CreateEnumerable<MnistData>(testData, reuseRowObject = false) 
    |> List.ofSeq

// Find the first sample from the most confused pair
let mutable sample = -1
for i in 0 .. predictionList.Length - 1 do
    let prediction = predictionList.[i].PredictedLabel
    let actual = actualList.[i].Label
    if actual = float32 maxConfusedActual && prediction = float32 maxConfusedPredicted then
        sample <- i
        break
```

This code calls `CreateEnumerable` to create lists of predicted and actual labels, and then searches the lists for a pair that matches the worst performing class pair identified earlier. The code simply grabs the first prediction it can find that matches the pair.

#### Print The Results

Finally, we can report the outcome like this:

```fsharp
Console.WriteLine($"\nFirst image that matches class pair")
Console.WriteLine($"Row ID: {actualList.[sample].RowID}")
Console.WriteLine($"Actual Label: {actualList.[sample].Label}")
Console.WriteLine($"Predicted Label: {predictionList.[sample].PredictedLabel}")

// Show confidence scores for all classes
Console.WriteLine("\nConfidence scores for all classes:")
for i in 0 .. predictionList.[sample].Score.Length - 1 do
    let index = Array.IndexOf(classLabels, float32 i)
    Console.WriteLine($"Class {i}: {predictionList.[sample].Score.[index]:P2}")

// Display the digit as ASCII art
Console.WriteLine($"\nASCII Art Visualization of Digit:")
VisualizeDigit(actualList.[sample].PixelValues)
```

This code reports everything noteworthy about the sample. You get the actual and predicted label, the confidence scores for all 10 classes and the digit drawn as ASCII art.

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