---
title: "Recap"
type: "lesson"
layout: "default"
sortkey: 900
---

Congratulations on finishing the lab. Here's what you have learned.

{{< encrypt >}}

You learned how to analyze the **New York TLC dataset**, both automatically by having an AI agent scan the data for you, and manually by inspecting the data by hand. You wrote code to generate a **histogram of every feature** in the dataset, and how to analyze these histograms to identify outliers to filter out.

You learned how to calculate the **trip duration** from the original pickup and dropoff dates and times. You also generated a histogram of the trip duration to identify any outliers to filter. You also calculated the **Pearson correlation matrix** for every feature and label, and used the matrix to identify features that are **strongly correlated** to the label.

You wrote code to generate the **scatterplot matrix**, and used it to learn how the strongly correlated features are related to the label. You also used the matrix to identify outliers and statistical artefacts in the dataset, and set up **data filters** for the New York TLC dataset. 

You built a **machine learning pipeline** and trained and evaluated a regression model on the dataset. Then you analyzed the regression metrics to determine the quality of the predictions. You discovered that the model might be **overfitting**. 

You learned how to load the full dataset as a **parquet file**, and feed the data into the existing dataview in your application. You regenerated the histogram grid, correlation matrix and scatterplot matrix to determine if your assumptions about the data are still valid for the full dataset. You then decided to **adjust your data filters and transformations** or leave them unchanged.

You completed the lab by experimenting with different data processing steps and regression algorithms to find the best-performing model. 

{{< /encrypt >}}
