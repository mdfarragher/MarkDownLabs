---
title: "Analyze The Data"
type: "lesson"
layout: "default"
sortkey: 30
---

We’ll begin by asking a large language model—such as ChatGPT or GitHub Copilot Chat—to analyze the California Housing dataset. You won’t write any code yet. The goal is to see what an AI assistant would suggest based on the raw data.

#### Ask Your AI Agent To Analyze The Dataset

Open Visual Studio Code and make sure the CSV file is open in the editor. Then expand the Copilot panel on the right-hand side of the screen, and enter the following prompt:

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

#### Manually Explore the Data

Let’s verify or challenge the agent’s recommendations by exploring the dataset manually.

Open **California-Housing.csv** in your favorite spreadsheet tool, and start looking for patterns, issues, and feature characteristics.

What to look out for:

-    Are there any missing values, zeros, or inconsistent rows?
-    Are the values in each column within a reasonable range?
-    Can you spot any extremely large or very small values?
-    What’s the distribution of values in columns like median_income, total_rooms, or households?
-    Are longitude and latitude useful as-is, or will they need transformation?

Write down 3 insights from your own analysis.
{.homework}

#### Compare and Reflect

Now compare your findings with the AI's analysis.

-    Did the agent suggest anything surprising?
-    Did you find anything it missed?
-    Are you confident in which preprocessing steps are necessary?

Try this prompt based on your observations:

"I inspected the data and found [ .... ]. What kind of preprocessing steps should I use to process this data? And are there any challenges I should take into account?"
{.prompt}

This back-and-forth helps you learn how to collaborate with agents as intelligent assistants, not just code generators.

#### Key Takeaway

Agents are powerful, but they’re not magic. Use them to speed up analysis, but always verify their suggestions through manual inspection and common sense.

