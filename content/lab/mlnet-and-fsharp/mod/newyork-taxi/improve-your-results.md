---
title: "Improve Your Results"
type: "lesson"
layout: "default"
sortkey: 140
---

There are many factors that influence the quality of your model predictions, including how you process the dataset, which regression algorithm you pick, and how you configure the training hyperparameters.

Here are a couple of things you could do to improve your model:

{{< encrypt >}}

- Split the pickup datetime into separate hour, day of week and weekend columns
- Analyze the trip distance and trip duration columns and calculate a new 'on-time' column
- Filter on one specific ratecode ID and determine the prediction accuracy per ratecode
- Bin and one-hot encode trip distance to create a new column called 'long-distance'
- Bin and one-hot encode trip duration to create a new column called 'long-duration'
- Try a different regression learning algorithm.
- Use different hyperparameter values for your learning algorithm.

Experiment with different data processing steps and regression algorithms. Document your best-performing machine learning pipeline for this dataset, and write down the corresponding regression evaluation metrics.
{ .homework }

How close can you make your predictions to the actual fare amounts? 

{{< /encrypt >}}