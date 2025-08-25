---
title: "Analyze The Data"
type: "lesson"
layout: "default"
sortkey: 30
---

# Analyze The Data

We'll begin by analyzing the Cleveland CAD dataset and come up with a plan for feature engineering. Our goal is to map out all required data transformation steps in advance to make later machine learning training as successful as possible.

{{< encrypt >}}

#### Manually Explore the Data

Let's start by exploring the dataset manually.

Open **Heart-Disease.csv** in Visual Studio Code, and start looking for patterns, issues, and feature characteristics.

What to look out for:

-    Are there any missing values, zeros, or inconsistent rows?
-    Are the values in each column within a reasonable range?
-    Can you spot any extremely large or very small values?
-    Is there any bias in columns like **Age** or **Sex**?
-    Do we have balanced populations of sick and healthy patients?

Write down 3 insights from your analysis.
{.homework}

#### Ask Copilot To Analyze The Dataset

You can also ask Copilot to analyze the CSV data and determine feature engineering steps. You should never blindly trust AI advice, but it can be insightful to run an AI scan after you've done your own analysis of the data, and compare Copilot's feedback to your own conclusions. 

Make sure the CSV file is still open in Visual Studio Code. Then expand the Copilot panel on the right-hand side of the screen, and enter the following prompt:

"You are a machine learning expert. Analyze this CSV file for use in a classification model that predicts Diag. What problems might the dataset have? What preprocessing steps would you suggest?"
{.prompt}

You can either paste in the column names and 5–10 sample rows, or upload the CSV file directly (if your agent supports file uploads).

![Analyze a dataset with an AI agent](../img/analyze.jpg)
{.img-fluid .mb-4}

#### What Might The Agent Suggest?

The agent may recommend steps like:

-    Normalize data columns
-    Handle outliers (like the patient with a **Chol** value of 564)
-    One-hot encode categorical columns like **Sex** and **Cp**
-    Handle missing values in the **Thal** and **Ca** columns
-    Balance the population of sick and healthy patients through sampling
-    Bin and one-hot encode the **Age** column into age groups

Write down 3 insights from the agent's analysis.
{.homework}

Next, we'll generate a couple of histograms to see if we can find outliers in any of the data columns. 

{{< /encrypt >}}