---
title: "Evaluate The Results"
type: "lesson"
layout: "default"
sortkey: 120
---

Now let's evaluate the quality of the model by comparing the predictions made on the 20% test data to the actual digits, and calculate the classification evaluation metrics.

So imagine you're scanning a stack of printed documents with lots of handwritten numbers. What level of accuracy would you consider acceptable?

Determine the minimum accuracy you deem acceptable for OCR of handwritten digits. This will be the target your model needs to beat.
{ .homework }

#### Calculate Evaluation Metrics

Enter the following prompt:

"Use the trained model to create predictions for the test set, and then calculate evaluation metrics for these predictions and print them."
{ .prompt }

That should create the following code:

```csharp
// Make predictions on test set
var predictions = model.Transform(testData);

// Evaluate the model
var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");
Console.WriteLine($"\nModel Evaluation Results:");
Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy:P2}");
Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy:P2}");
Console.WriteLine($"Log-Loss: {metrics.LogLoss:F4}");
Console.WriteLine($"Log-Loss Reduction: {metrics.LogLossReduction:F4}");
```

This code calls `Transform` to set up predictions for every patient in the test partition. The `MulticlassClassification.Evaluate` method then compares these predictions to the actual diagnoses and automatically calculates these metrics:

- **MicroAccuracy**: this is the average accuracy (=the number of correct predictions divided by the total number of predictions) for every digit in the dataset.
- **MacroAccuracy**: this is calculated by first calculating the average accuracy for each unique prediction value, and then taking the averages of those averages.
- **LogLoss**: this is a metric that expresses the size of the error in the predictions the model is making. A logloss of zero means every prediction is correct, and the loss value rises as the model makes more and more mistakes.
- **LogLossReduction**: this metric is also called the Reduction in Information Gain (RIG). It expresses the probability that the model’s predictions are better than random chance.

We can compare the micro- and macro accuracy to discover if the dataset is biased. In an unbiased set each unique label value will appear roughly the same number of times, and the micro- and macro accuracy values will be close together. 

We already checked for any class imbalance in the MNIST dataset by generating a histogram of labels. The histogram looked fine, with roughly equal populations for each unique digit. So we expect the micro- and macro accuracy values to be close together. 

For the SDCA learning algorithm, you should get something like the following:

![Binary Classification Model Evaluation](../img/evaluate.png)
{ .img-fluid .mb-4 }

Let's analyze my results:

The Macro Accuracy is **91.13%**. This is the average accuracy computed independently for each class, then averaged. It treats all classes equally, even if some digits appear more often than others. This metric is good for judging balanced performance across the entire dataset. 

A value of 91.13% is good. The model is performing well across all digits, with no single class dragging the average down too much.

The Micro Accuracy is **91.34%**. This is the overall accuracy across all predictions, weighting each instance equally. It basically means the "percentage of correct predictions" on the entire dataset, and will be affected by bias if the dataset is class-imbalanced.

A value of 91.34% is good. Over 9 out of 10 predictions are correct, which is solid for a model trained on 10k images. Also note that the micro- and macro accuracies are close together, which means that the dataset does not have a class-imbalance. This is what we expected, as we already checked for class imbalance earlier.

The Log-Loss is **0.3538**. This measures how well the model’s predicted probabilities match the true labels. Lower is better: 0 means perfect confidence and correctness, and higher values mean more wrong or overconfident predictions.

A value of 0.3538 is good. This is fairly low, meaning the probability outputs are reasonably well-calibrated and confident when correct.

Ths Log-Loss Reduction is **0.8460**. This shows how much better the model’s log-loss is compared to a naïve baseline (e.g., guessing by class frequency). A value close to 1 means a big improvement over random guessing.

A value of 0.8460 is very good. The model shows an 84.6% improvement over baseline which shows that it performs far better than random guessing.

So how did your model do?

Compare your model with the target you set earlier. Did it make predictions that beat the target? Are you happy with the predictive quality of your model? Can you explain what each metric means for the quality of your predictions? 
{ .homework }

#### Generate The Confusion Matrix

Printing the confusion matrix is super easy because ML.NET has a built-in method that does everything for us. We don't have to prompt our AI agent for this, because you can do it in a single line of code:

```csharp
// Display confusion matrix
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
```

The multiclass evaluation metrics object has a property `ConfusionMatrix` to access the matrix, and the `GetFormattedConfusionTable` method returns the full table as a string that we can write directly to the console. 

Homework: Add this code to your app, then run the app and examine the confusion matrix. Is this what you expected? Write down your observations.  
{ .homework }

My matrix looks like this:

![Confusion Matrix](../img/confusion-print.png)
{ .img-fluid .mb-4 }

In the matrix, each column holds a prediction and each row corresponds to a ground truth label value. The main diagonal shows all correct predictions, and all the errors are off the diagonal. We also get precison and recall values for each individual class.

But there is a problem with this matrix. What you're seeing on the x- and y-axis are not the class labels, but the class label indices. Remember that `MapValueToKey` step in the machine learning pipeline? It converted every unique class label to a numeric index (the 'Key') and that's what we're seeing in the matrix right now. 

So if you look closely at the image, you can see 13 errors made by the model when predicting the class number 5 which was actually class number 4. But these two classes could be any value, and are not guaranteed to match the digits 5 and 4. 

To generate a confusion matrix with sensible axis labels, we'll need to add some extra code. We'll do that next.

#### Plot The Confusion Matrix

Now let's plot this matrix with ScottPlot, just like we did with the Cleveland CAD dataset. Enter the following prompt:

"Add code to plot the confusion matrix as a heatmap with Scottplot. Use the ConfusionMatrix.GetCountForClassPair method provided by ML.NET"
{ .prompt }

Note that I'm asking for a calculation that uses `GetCountForClassPair`, a built-in ML.NET method that calculates the prediction count for any cell in the matrix. This will (hopefully) prevent the agent from doing the calculation on its own. 

Carefully examine the generated code, for it's very easy for AI agents to make mistakes here. What you want to see is the following code:

```csharp
// Build an array to map classes to key indices
var labelColumn = predictions.Schema["Label"];
var keyValues = new VBuffer<float>(); 
labelColumn.GetKeyValues(ref keyValues);
float[] classLabels = keyValues.DenseValues().ToArray();
```

This code accesses the schema of the **Label** column and uses `GetKeyValues` to obtain the list of key values (stored as a `VBuffer`). Then a call to `DenseValues` converts the vbuffer to a float array.

We now have a `classLabels` array with label values, with array indices that exactly match the key indices the machine learning pipeline is using. 

When AI agents work on multiclass dataset with the ML.NET library, they often confuse class label values and key indices. Be extra attentive and double-check your code. Every call to GetCountForClassPair should provide indices, not label values. 
{ .tip }

With that, we can assemble a confusion matrix like this:

```csharp
// Get the confusion matrix
var confusionMatrix = metrics.ConfusionMatrix;

// Get the number of classes
int numClasses = confusionMatrix.NumberOfClasses;

// Create a 2D array for the confusion matrix data
double[,] matrixData = new double[numClasses, numClasses];

// Fill the matrix using GetCountForClassPair
for (int actualClass = 0; actualClass < numClasses; actualClass++)
{
    for (int predictedClass = 0; predictedClass < numClasses; predictedClass++)
    {
        var predictionIndex = Array.IndexOf(classLabels, predictedClass);
        var actualIndex = Array.IndexOf(classLabels, actualClass);
        var count = confusionMatrix.GetCountForClassPair(predictionIndex, actualIndex);
        matrixData[actualClass, predictedClass] = count;
    }
}
```

The code loops over each possible class pair and uses `Array.IndexOf` to convert the class values to their corresponding key indices. The `GetCountForClassPair` method uses the indices to get the number of predictions for that class pair. It then builds a 2-dimensional matrix with each pair in the correct position. 

The plotting code that follows is similar to what we used to generate the Confusion matrix heatmap for the Cleveland CAD dataset. 

Homework: Run your app and examine the plotted confusion matrix. Examine the most popular incorrect predictions. Do they make sense? Write down your observations. 
{ .homework }

My matrix looks like this:

![Confusion Matrix](../img/confusion-matrix.png)
{ .img-fluid .mb-4 }

The main diagonal contains all correct predictions, and each cell is deep black as expected. The matrix cells off the main diagonal describe incorrect predictions, and from the plot we can quickly see that the most popular mistake is when the model predicts a '9' but the digit is actually a '4'. A second common mistake is when the model predicts a '3' but the digit is actually a '5'. 

These mistakes make sense if we consider the quality of the handwriting in the MNIST dataset. I mean, look at this:

![Confusion Matrix](../img/mnist-examples.png)
{ .img-fluid .mb-4 }

These are a couple of random digits from the dataset. You can see several digits in this example that could either be a '4' or a '9', so the machine learning model does not have an easy task sorting out which is which. 

#### Create A Utility Class

Let's put the code for plotting the confusion matrix into a utility class so that we can reuse the code in later lab modules and lessons.

In Visual Studio Code, select the code that generates the confusion matrix plot. Then press CTRL+I to launch the in-line AI prompt window, and type the following prompt:

"Move all of this code to a new method PlotConfusionMatrix, and put this method in a new utility class called MulticlassUtils."
{ .prompt }

This cleaned up my main method a lot and only left the following code:

```csharp
// Plot confusion matrix as heatmap
var plot = MulticlassUtils.PlotConfusionMatrix(metrics.ConfusionMatrix, classLabels);
plot.SavePng("confusion-matrix.png", 900, 600);
```

If you get stuck or want to save some time, feel free to download my completed MulticlassUtils class from Codeberg and use it in your own project:

https://codeberg.org/mdft/ml-mlnet-csharp/src/branch/main/Mnist/MulticlassUtils.cs


#### Next Steps

Next, let's add a prediction engine to the machine learning app to make a few ad-hoc predictions for random digits in the dataset.
