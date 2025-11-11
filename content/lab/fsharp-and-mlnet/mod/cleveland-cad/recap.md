---
title: "Recap"
type: "lesson"
layout: "default"
sortkey: 140
---

# Recap

Congratulations on finishing the lab. Here's what you have learned.

{{< encrypt >}}

You learned how to analyze the **Cleveland CAD dataset**, both automatically by having an AI agent scan the data for you, and manually by inspecting the data by hand. You generated **histograms for every feature** in the dataset, and used them to identify outliers to filter out.

You used a custom mapping to **handle missing values**, and changed the label from multiclass to binary classification. You also calculated the **Pearson correlation matrix** for every feature and label, and used the matrix to identify features that are **strongly correlated** with the label.

From the age histogram plot, you learned that the age feature is **biased** with more men than women in the dataset. You resolved this by **undersampling** the men to ensure that the model treats both genders equally. 

You generated the **scatterplot matrix**, and used it to learn how the strongly correlated features are related to the label. You also used the matrix to identify outliers and statistical artefacts in the dataset, and set up **data filters** for the dataset. 

You trained and evaluated a binary classification model on the dataset. You learned that the **L-BFGS algorithm** produces scores that cannot be interpreted as probability values, so you added a **Platt calibration** step to your pipeline to restore the probabilities. 

Then you analyzed the binary classification metrics to determine the quality of the predictions. You also added code to plot the **ROC curve** and the **confusion matrix**, and interpreted the latter to determine how your model handles **false negatives**. 

You completed the lab by first predicting my health, and then experimenting with different data processing steps and classification algorithms to find the best-performing model. 

{{< /encrypt >}}