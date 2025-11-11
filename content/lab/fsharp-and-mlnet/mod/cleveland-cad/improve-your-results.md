---
title: "Improve Your Results"
type: "lesson"
layout: "default"
sortkey: 120
---

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