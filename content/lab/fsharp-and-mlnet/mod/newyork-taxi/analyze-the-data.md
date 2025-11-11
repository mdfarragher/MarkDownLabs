---
title: "Analyze The Data"
type: "lesson"
layout: "default"
sortkey: 30
---

We'll begin by analyzing the New York TLC dataset and come up with a plan for feature engineering. Our goal is to map out all required data transformation steps in advance to make later machine learning training as successful as possible.

{{< encrypt >}}

#### Manually Explore the Data

Let's start by exploring the dataset manually.

Open **Taxi-Trips.csv** in Visual Studio Code, and start looking for patterns, issues, and feature characteristics.

What to look out for:

-    Are there any missing values, zeros, or inconsistent rows?
-    Are the values in each column within a reasonable range?
-    Can you spot any extremely large or very small values?
-    What's the distribution of values in columns like **passenger_count** or **trip_distance**?
-    Are **tpep_pickup_datetime** and **tpep_dropoff_datetime** useful as-is, or will they need transformation?

Write down 3 insights from your analysis.
{.homework}

#### Ask Copilot To Analyze The Dataset

You can also ask Copilot to analyze the CSV data and determine feature engineering steps. You should never blindly trust AI advice, but it can be insightful to run an AI scan after you've done your own analysis of the data, and compare Copilot's feedback to your own conclusions. 

Make sure the CSV file is still open in Visual Studio Code. Then expand the Copilot panel on the right-hand side of the screen, and enter the following prompt:

"You are a machine learning expert. Analyze this CSV file for use in a regression model that predicts total_amount. What problems might the dataset have? What preprocessing steps would you suggest?"
{.prompt}

You can either paste in the column names and 5–10 sample rows, or upload the CSV file directly (if your agent supports file uploads).

![Analyze a dataset with an AI agent](../img/analyze.jpg)
{.img-fluid .mb-4}

#### What Might The Agent Suggest?

The agent may recommend steps like:

-    Normalizing data columns
-    Handling extreme outliers
-    Remove invalid rows, for example with passenger_count = 0 or trip_distance = 0
-    One-hot encoding of categorical features (like RatecodeID and payment_type)
-    Drop features that are tightly correlated with total_amount
-    Converting the pickup and dropoff times to a new trip duration feature
-    Adding new features like pickup_hour, pickup_day_of_week and pickup_weekend_flag

Write down 3 insights from the agent's analysis.
{.homework}

Next, we'll generate a couple of histograms to see if we can find outliers in any of the data columns. 

{{< /encrypt >}}