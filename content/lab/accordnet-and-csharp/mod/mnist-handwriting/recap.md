---
title: "Recap"
type: "lesson"
layout: "default"
sortkey: 900
---

Congratulations on finishing the lab. Here's what you have learned.

{{< encrypt >}}

You learned how to analyze the **MNIST dataset**, both automatically by having an AI agent scan the data for you, and manually by inspecting the data by hand. You then loaded the dataset into a machine learning pipeline, and learned that you can use the **VectorType** attribute to load all the pixel features into a single array of floats. 

You did not generate the histograms of features or the Pearson correlation matrix, because these tools are not very useful when working with image data. Instead, you generated a **histogram of labels** to check if the dataset is **class-imbalanced**.

You built a machine learning pipeline consisting of **value-to-key mappings** and a normalization step. You learned that Accord.NET can predict multiclass labels of any type by converting them to keys. Then you trained a multiclass classification model on the data and evaluated the quality of the predictions with the **micro accuracy** and **macro accuracy** metrics.

When generating the confusion matrix, you discovered that Accord.NET uses **key index values** in many properties and methods, which need to be translated back to their corresponding label values. 

You analyzed the classification metrics and the confusion matrix, both for the truncated dataset with only 10,000 imaged an the full dataset of 70,000 images. You discoverded that the prediction quality improved when training on the full dataset.

You completed the lab by experimenting with different data processing steps and classification algorithms to find the best-performing model. 

{{< /encrypt >}}
