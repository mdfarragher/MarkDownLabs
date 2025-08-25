---
title: "Get The Data"
type: "lesson"
layout: "default"
sortkey: 20
---

Let's start by downloading the Cleveland CAD dataset. 

{{< encrypt >}}

Grab the file from here: [Cleveland CAD dataset](https://csvbase.com/mdfarragher/Heart-Disease).

Download the file and save it as **Heart-Disease.csv**.

The file is a comma-separated text file with 15 columns of data:

- Row ID: a unique row identifier (added by CsvBase)
- Age
- Sex: 1 = male, 0 = female
- Chest Pain Type: 1 = typical angina, 2 = atypical angina , 3 = non-anginal pain, 4 = asymptomatic
- Resting blood pressure in mm Hg on admission to the hospital
- Serum cholesterol in mg/dl
- Fasting blood sugar > 120 mg/dl: 1 = true; 0 = false
- Resting EKG results: 0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes’ criteria
- Maximum heart rate achieved
- Exercise induced angina: 1 = yes; 0 = no
- ST depression induced by exercise relative to rest
- Slope of the peak exercise ST segment: 1 = up-sloping, 2 = flat, 3 = down-sloping
- Number of major vessels (0–3) colored by fluoroscopy
- Thallium heart scan results: 3 = normal, 6 = fixed defect, 7 = reversible defect
- Diagnosis of heart disease: 0 = normal risk, 1-4 = elevated risk

Columns 2-14 are patient diagnostic information, and the last column is the diagnosis: 0 means a healthy patient, and values 1-4 mean an elevated risk of heart disease.

Let's get started.

#### Set Up The Project

Now open your terminal and navigate to the folder where you want to create the project (e.g., **~/Documents**), and run:

```bash
dotnet new console -o HeartDisease
cd HeartDisease
```

This creates a new C# console application with:

- **Program.cs** – your main program file
- **HeartDisease.csproj** – your project file

Then move the Heart-Disease.csv file into this folder.

Now run the following command to install the Accord.NET machine learning library:

```bash
dotnet add package Accord.NET
```

Next, we're going to analyze the dataset and come up with a feature engineering plan.

{{< /encrypt >}}
