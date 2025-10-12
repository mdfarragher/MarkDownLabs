---
title: "Improve Your Results"
type: "lesson"
layout: "default"
sortkey: 50
---

There are many factors that influence the quality of your model predictions, including how you process the dataset, which regression algorithm you pick, and how you configure the training hyperparameters.

Here are a couple of things you could do to improve your model:

{{< encrypt >}}

- Bin the latitude and longitude into more than 10 bins, thus creating a finer grid over the state of California.
- Use a different learning algorithm.
- Use different hyperparameter values for your learning algorithm.
- Eliminate outliers with a large number of rooms.
- Eliminate outliers with very small populations.
- Try to bin other columns, for example replacing **HousingMedianAge** with a three-element one-hot encoding vector with columns "Young", "Median" and "Old".

Experiment with different data processing steps and regression algorithms. Document your best-performing machine learning pipeline for this dataset, and write down the corresponding regression evaluation metrics.
{ .homework }

How close can you make your predictions to the actual house prices? Feel free to try out different approaches. This is how you build valuable machine learning skills!

{{< /encrypt >}}