---
title: "Get The Data"
type: "lesson"
layout: "default"
sortkey: 20
---

Let's start by downloading the California Housing dataset. 

{{< encrypt >}}

You can grab the file from here:
[California 1990 housing census](https://csvbase.com/mdfarragher/California-Housing).

Download the file and save it as **California-Housing.csv**.

The dataset is a CSV file with 17,000 records that looks like this:

![The California Housing Dataset](../img/data.jpg)
{ .img-fluid .mb-4 }

The file contains information on 17,000 housing blocks all over the state of California. Here's a description of each column:

-    Column 1: The unique row identifier (added by CsvBase)
-    Column 2: The longitude of the housing block
-    Column 3: The latitude of the housing block
-    Column 4: The median age of all the houses in the block
-    Column 5: The total number of rooms in all houses in the block
-    Column 6: The total number of bedrooms in all houses in the block
-    Column 7: The total number of people living in all houses in the block
-    Column 8: The total number of households in all houses in the block
-    Column 9: The median income of all people living in all houses in the block
-    Column 10: The median house value for all houses in the block

This dataset cannot be used directly for machine learning. It must be cleaned, scaled, and preprocessed—this is what we'll focus on in the next steps.

#### Set Up Your Project

Now open your terminal (called Windows Terminal on Windows, or Console on Mac/Linux). Navigate to the folder where you want to create the project (e.g., **~/Documents**), and run:

```bash
dotnet new console -lang F# -o CaliforniaHousing
cd CaliforniaHousing
```

This creates a new F# console application with:

- **Program.fs** – your main program file
- **CaliforniaHousing.fsproj** – your project file

Then move the California-Housing.csv file into this folder.

Now run the following command to install additional packages:

```bash
dotnet add package Microsoft.ML
```

**Microsoft.ML** is the Microsoft MLNET machine learning library. We will use it to build all our applications in this course.

Next, we're going to analyze the dataset and come up with a feature engineering plan.

{{< /encrypt >}}