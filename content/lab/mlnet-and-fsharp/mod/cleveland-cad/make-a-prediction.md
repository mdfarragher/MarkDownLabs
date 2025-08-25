---
title: "Make A Prediction"
type: "lesson"
layout: "default"
sortkey: 110
---

# Make A Prediction

To wrap up, let's use the model to make a prediction.

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

```fsharp
// Create a prediction engine to demonstrate single predictions
let predictionEngine = mlContext.Model.CreatePredictionEngine<HeartDataInput, HeartDiseasePrediction>(mlModel)

// Get user input for patient data and make a prediction
let patientData = GetPatientDataFromUser()
let prediction = predictionEngine.Predict(patientData)

// Display results
printfn $"Diagnosis: {if prediction.PredictedLabel then "HEART DISEASE" else "HEALTHY"}"
printfn $"Probability: {prediction.Probability:P2} ({prediction.Probability:F4})"
printfn $"Confidence: {abs prediction.Score:F4}"
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