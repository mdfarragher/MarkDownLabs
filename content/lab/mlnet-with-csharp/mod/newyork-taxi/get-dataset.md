---
title: "Get The Data"
type: "lesson"
layout: "default"
sortkey: 20
---

Let's start by downloading the New York TLC dataset. Grab the file from here:
[Yellow Taxi Trip Records From December 2018](https://csvbase.com/mdfarragher/Taxi-Trips).

Download the file and save it as **Taxi-Trips.csv**.

This is a truncated dataset with 9,998 records. The original dataset for December 2018 has over 8 million records and is close to 1 GB in size. In this lab, we'll use the smaller dataset to quickly set up our app and experiment with different machine learning pipelines. Later, we'll download the full dataset and test our app on all 8 million trips.

There are a lot of interesting columns in this dataset, for example:

- **tpep_pickup_datetime**: The pickup date and time
- **tpep_dropoff_datetime**: The dropoff date and time
- **passenger_count**: The number of passengers
- **trip_distance**: The trip distance
- **RatecodeID**: The rate code (standard, JFK, Newark, …)
- **payment_type**: The payment type (credit card, cash, …)
- **fare_amount**: The fare amount
- **total_amount**: The total amount (fare plus tip, tolls, tax, etc.)

Let's get started.

#### Set Up The Project

Now open your terminal and navigate to the folder where you want to create the project (e.g., **~/Documents**), and run:

```bash
dotnet new console -o TaxiFarePrediction
cd TaxiFarePrediction
```

This creates a new C# console application with:

- **Program.cs** – your main program file
- **TaxiFarePrediction.csproj** – your project file

Then move the Taxi-Trips.csv file into this folder.

Now run the following command to install the Microsoft.ML machine learning library:

```bash
dotnet add package Microsoft.ML
```

Next, we're going to analyze the dataset and come up with a feature engineering plan.
