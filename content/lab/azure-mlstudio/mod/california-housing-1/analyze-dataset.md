---
title: "Analyze The Data"
type: "lesson"
layout: "default"
sortkey: 25
---

We’ll begin by analyzing the California Housing dataset and come up with a plan for feature engineering. Our goal is to map out all required data transformation steps to make machine learning training possible.

{{< encrypt >}}

#### Manually Explore the Data

Let’s start by exploring the dataset manually.

Open **California-Housing.csv** in your favorite text editor and start looking for patterns, issues, and feature characteristics.

What to look out for:

-    Are there any missing values, zeros, or inconsistent rows?
-    Are the values in each column within a reasonable range?
-    Can you spot any extremely large or very small values?
-    What’s the distribution of values in columns like median_income, total_rooms, or households?
-    Are longitude and latitude useful as-is, or will they need transformation?

Write down 3 insights from your analysis.
{.homework}

#### Ask Your AI Agent To Analyze The Dataset

Are you using Copilot, ChatGPT or Claude? You can ask your AI agent to analyze the CSV data and determine feature engineering steps. You should never blindly trust AI advice, but it can be insightful to run an AI scan after you've done your own analysis of the data, and compare its feedback to your own conclusions. 

Copy and paste the first 5–10 rows of the dataset into your AI chat window, or upload the CSV file directly if your agent supports file uploads. Then enter the following prompt:

"You are a machine learning expert. Analyze this CSV file for use in a regression model that predicts median_house_value. What problems might the dataset have? What preprocessing steps would you suggest?"
{.prompt}

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

Now let's compare your findings with the AI analysis.

-    Did the agent suggest anything surprising?
-    Did you find anything it missed?
-    Are you confident in which preprocessing steps are necessary?

Try this follow-up prompt based on your observations:

"I inspected the data and found [ .... ]. What kind of preprocessing steps should I use to process this data? And are there any challenges I should take into account?"
{.prompt}

This back-and-forth helps you learn how to collaborate with AI agents as intelligent assistants.

{{< /encrypt >}}

