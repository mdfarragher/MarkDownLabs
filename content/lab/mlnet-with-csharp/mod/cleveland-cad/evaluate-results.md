---
title: "Evaluate The Results"
type: "lesson"
layout: "default"
sortkey: 120
---

Now let's evaluate the quality of the model by comparing the predictions made on the 20% test data to the actual diagnoses, and calculate the binary classification evaluation metrics.

So imagine you walk into a hospital with chest pain and ask an AI doctor for a diagnosis. What level of accuracy would you consider acceptable?

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

The AI agent will produce a big chunk of new code, because ML.NET does not have drop-in support for plotting ROC curves and everything needs to be calculated by hand. 

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

"Move all of this code to a new method PlotRoc, and put this method in a new utility class called EvaluateUtils."
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
var rocPlot = EvaluateUtils.PlotRoc(predictionValues, actualLabels);
rocPlot.SavePng("roc-curve.png", 900, 600);
```

I like this code interface. The new `PlotRoc` method only needs `float[]` arrays for the prediction probabilities and the actual label values. It can then calculate the complete ROC plot without needing anything else from the main application method. 

Now let's do the same for the confusion matrix. Select the code that generates the matrix, press CTRL+I and enter the following prompt in the inline window:

"Move all of this code to a new method PlotConfusion, and put this method in the EvaluateUtils class."
{ .prompt }

My AI agent generated working code from this prompt, but I tweaked the result a little so that the new `PlotConfusion` method uses the same calling interface as the `PlotRoc` method, like this:

```csharp
// Create and plot confusion matrix
var cmPlot = EvaluateUtils.PlotConfusion(predictionValues, actualLabels);
cmPlot.SavePng("confusion-matrix.png", 900, 600);
```

The new `PlotConfusion` method calculates the true positives, true negatives, false positives and false negatives from the provided `predictionValues` and `actualLabels` arrays, and then generates the heatmap for the confusion matrix. 

Perfect!

If you get stuck or want to save some time, feel free to download my completed EvaluateUtils class from Codeberg and use it in your own project:

https://codeberg.org/mdft/ml-mlnet-csharp/src/branch/main/HeartDisease/EvaluateUtils.cs


#### Next Steps

Next, let's add a prediction engine to the machine learning app to make a few ad-hoc heart disease predictions on fictional patients.
