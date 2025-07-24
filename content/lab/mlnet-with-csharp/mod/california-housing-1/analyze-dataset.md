---
title: "Analyze The Data"
type: "lesson"
layout: "default"
sortkey: 30
---

We’ll begin by analyzing the California Housing dataset and come up with a plan for feature engineering. You won’t write any C# code yet, our goal is to first map out all required data transformation steps to make later machine learning training possible.

#### Manually Explore the Data

Let’s start by exploring the dataset manually.

Open **California-Housing.csv** in Visual Studio Code, and start looking for patterns, issues, and feature characteristics.

What to look out for:

-    Are there any missing values, zeros, or inconsistent rows?
-    Are the values in each column within a reasonable range?
-    Can you spot any extremely large or very small values?
-    What’s the distribution of values in columns like median_income, total_rooms, or households?
-    Are longitude and latitude useful as-is, or will they need transformation?

Write down 3 insights from your analysis.
{.homework}

#### Ask Copilot To Analyze The Dataset

You can also ask Copilot to analyze the CSV data and determine feature engineering steps. You should never blindly trust AI advice, but it can be insightful to run an AI scan after you've done your own analysis of the data, and compare Copilot's feedback to your own conclusions. 

Make sure the CSV file is still open in Visual Studio Code. Then expand the Copilot panel on the right-hand side of the screen, and enter the following prompt:

"You are a machine learning expert. Analyze this CSV file for use in a regression model that predicts median_house_value. What problems might the dataset have? What preprocessing steps would you suggest?"
{.prompt}

You can either paste in the column names and 5–10 sample rows, or upload the CSV file directly (if your agent supports file uploads).

![Analyze a dataset with an AI agent](../img/analyze.jpg)
{.img-fluid .mb-4}

#### What Might The Agent Suggest?

The agent may recommend steps like:

-    Normalizing income or house value columns
-    Handling extreme outliers
-    One-hot encoding categorical features (like location bins)
-    Engineering new features (e.g., rooms_per_person)
-    Scaling values for better model convergence

Write down 3 insights from the agent’s analysis.
{.homework}

#### Compare and Reflect

Now let's compare your findings with Copilot's analysis.

-    Did Copilot suggest anything surprising?
-    Did you find anything it missed?
-    Are you confident in which preprocessing steps are necessary?

Try this follow-up prompt based on your observations:

"I inspected the data and found [ .... ]. What kind of preprocessing steps should I use to process this data? And are there any challenges I should take into account?"
{.prompt}

This back-and-forth helps you learn how to collaborate with AI agents as intelligent assistants, not just code generators.

#### Key Takeaway

Agents are powerful, but they’re not magic. Use them to speed up analysis, but always verify their suggestions through manual inspection and common sense.

