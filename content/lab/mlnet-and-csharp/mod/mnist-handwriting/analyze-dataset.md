---
title: "Analyze The Data"
type: "lesson"
layout: "default"
sortkey: 30
---

We’ll begin by analyzing the MNIST dataset and come up with a plan for feature engineering. Our goal is to map out all required data transformation steps in advance to make later machine learning training as successful as possible.

{{< encrypt >}}

#### Manually Explore the Data

Let’s start by exploring the dataset manually.

Open **Mnist-Handwriting.csv** in Visual Studio Code, and start looking for patterns, issues, and feature characteristics.

What to look out for:

-    Are there any missing values or inconsistent rows?
-    Are the pixel values in each column all between 0 and 255?
-    Do we have balanced populations of labels?

Write down 3 insights from your analysis.
{.homework}

#### Ask Copilot To Analyze The Dataset

Now expand the Copilot panel in Visual Studio Code and enter the following prompt:

"You are a machine learning expert. Analyze this CSV file for use in a classification model that predicts Label. What problems might the dataset have? What preprocessing steps would you suggest?"
{.prompt}

You can either paste in the column names and 5–10 sample rows, or upload the CSV file directly (if your agent supports file uploads).

![Analyze a dataset with an AI agent](../img/analyze.jpg)
{.img-fluid .mb-4}

#### What Might The Agent Suggest?

The agent may recommend steps like:

-    Normalize pixel columns to a range of 0 .. 1
-    Check for class imbalance and fix with over- or undersampling
-    Remove pixels that are always zero
-    Augment the dataset by translating, rotating and scaling the images

Write down 3 insights from the agent’s analysis.
{.homework}

Next, we'll generate a quick histogram to check if the populations for all classes are balanced. 

{{< /encrypt >}}
