---
title: "Load The Full Dataset"
type: "lesson"
layout: "default"
sortkey: 120
---

# Load The Full Dataset

So far, we have been working with a subset of the New York TLC dataset. The subset contains the first 10,000 taxi trips made in the early hours of the morning on December 1st, 2018. This is not a representative subset of all the trips made in December, but it allowed us to quickly design a data transformation pipeline and train a regression model on the data.

Now, let's download the full dataset. 

{{< encrypt >}}

The New York City Taxi and Limousine Commission (TLC) website has a page where you can access all [TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) in Parquet format. Please [download the full dataset for December 2018 with this link](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2018-12.parquet).

This is a file named **yellow_tripdata_2018-12.parquet**. It's about 112 MB in size and holds roughly 8.1 million taxi trip records. Copy this file into your project folder.

#### Load The Parquet File

Now let's alter the app so that we can choose if we want to load the parquet file with all taxi trips or the CSV file with only the first 10,000 trips. Enter the following prompt:

"Add code that asks the user if they want to load the Taxi-Trips.csv file with the first 10,000 trips or the yellow_tripdata_2018-12.parquet file with all 8.1 million trips. Use the Parquet.NET library to load the parquet file into a list of TaxiTrip objects and then use LoadFromEnumerable to convert the list to a dataview for machine learning."
{ .prompt }

This is not an easy refactor because the ML.NET library does not support parquet loading yet, even though there is a NuGet package for it. My agent repeatedly got stuck trying to use a nonexistent `LoadFromParquetFile` method. Instead, you have to use the **Parquet.NET** library to load the file and then convert the list of trips back to a dataview using `LoadFromEnumerable`.

We want to see code like this:

```fsharp
let parquetTrips = 
    ParquetSerializer.DeserializeAsync<TaxiTrip>("yellow_tripdata_2018-12.parquet")
        .GetAwaiter()
        .GetResult()
data <- mlContext.Data.LoadFromEnumerable(taxiTrips)
```

The `ParquetSerializer.DeserializeAsync` method will load the file and produce a list of `TaxiTrip` objects (called with `GetAwaiter` and `GetResults` so that the synchronous calling thread will block while the data is loading), and then `LoadFromEnumerable` will convert the list to a dataview. 

Unfortunately, this code will not work. Here's what happens when you try to run the app:

![Parquet Conversion Error](../img/parquet-error.jpg)
{.img-fluid .mb-4}

The error message refers to the fact that the **VendorID** column in the parquet file has a different data type and nullability than the corresponding property in the `TaxiTrip` class. The schema of the data and the class structure are out of sync.

So let's inspect the schema of the parquet file data to find out what's going on. The [Parquet.NET project](https://github.com/aloneguid/parquet-dotnet) provides a nice little tool called 'Floor' that can visualize a parquet file. Here is the schema of the TLC dataset according to Floor: 

![Parquet File Schema](../img/parquet-schema.jpg)
{.img-fluid .mb-4}

You can see that many fields have different names, everything is nullable (the repetition type is OPTIONAL) and the data types are twice as wide as expected (`double` instead of `float`, and `long` instead of `int`).

What we need is a modified `TaxiTrip` class that takes this new schema into account. Something like this:

```fsharp
[<CLIMutable>]
type ParquetTaxiTrip = {
    VendorID: Nullable<int64>
    tpep_pickup_datetime: Nullable<DateTime>
    tpep_dropoff_datetime: Nullable<DateTime>
    passenger_count: Nullable<float>
    trip_distance: Nullable<float>
    RatecodeID: Nullable<float>
    store_and_fwd_flag: string
    PULocationID: Nullable<int64>
    DOLocationID: Nullable<int64>
    payment_type: Nullable<int64>
    fare_amount: Nullable<float>
    extra: Nullable<float>
    mta_tax: Nullable<float>
    tip_amount: Nullable<float>
    tolls_amount: Nullable<float>
    improvement_surcharge: Nullable<float>
    total_amount: Nullable<float>
    congestion_surcharge: Nullable<int>
    airport_fee: Nullable<int>
}
```

Note that every property is now nullable and that most datatypes are `Int64` or `double`. This class exactly matches the schema of the data in the parquet file. 

To load the data, all we need to do is pass the correct type to the `DeserializeAsync` method, like this:

```fsharp
let parquetTrips = 
    ParquetSerializer.DeserializeAsync<ParquetTaxiTrip>("yellow_tripdata_2018-12.parquet")
        .GetAwaiter()
        .GetResult()
```

This code works and will load all 8.1 million trips. But now we have a new problem: we cannot convert the list of trips to a dataview, because the ML.NET library does not support nullable `double` or `Int64` properties at all. 

We can fix this by manually converting every `ParquetTaxiTrip` to a `TaxiTrip`, like this:

```fsharp
// Convert data to TaxiTrip list
let taxiTrips = 
    parquetTrips 
    |> Seq.map (fun p -> 
        {
            RowID = 0
            VendorID = p.VendorID.GetValueOrDefault(0L) |> int
            PickupDateTime = p.tpep_pickup_datetime.GetValueOrDefault(DateTime.MinValue)
            DropoffDateTime = p.tpep_dropoff_datetime.GetValueOrDefault(DateTime.MinValue)
            PassengerCount = p.passenger_count.GetValueOrDefault(0.0) |> int
            TripDistance = p.trip_distance.GetValueOrDefault(0.0) |> float32
            RatecodeID = p.RatecodeID.GetValueOrDefault(0.0) |> int
            StoreAndFwdFlag = p.store_and_fwd_flag ?? ""
            PULocationID = p.PULocationID.GetValueOrDefault(0L) |> int
            DOLocationID = p.DOLocationID.GetValueOrDefault(0L) |> int
            PaymentType = p.payment_type.GetValueOrDefault(0L) |> int
            FareAmount = p.fare_amount.GetValueOrDefault(0.0) |> float32
            Extra = p.extra.GetValueOrDefault(0.0) |> float32
            MtaTax = p.mta_tax.GetValueOrDefault(0.0) |> float32
            TipAmount = p.tip_amount.GetValueOrDefault(0.0) |> float32
            TollsAmount = p.tolls_amount.GetValueOrDefault(0.0) |> float32
            ImprovementSurcharge = p.improvement_surcharge.GetValueOrDefault(0.0) |> float32
            TotalAmount = p.total_amount.GetValueOrDefault(0.0) |> float32
        })
    |> List.ofSeq

// Convert list to IDataView
data <- mlContext.Data.LoadFromEnumerable(taxiTrips)
```

This is not very efficient code, but it's good enough for now. 

#### Train And Evaluate The Model

At this point, we are done. The app can now load all 8.1 million taxi trips from the parquet file, convert them to a dataview, and then use the existing code to kick off a machine learning training run and report the regression metrics. 

Run your app and have it load the taxi trips from the parquet file. Then wait for model training to finish and note the new regression metrics. What has happened to the predictive quality of the model? 
{ .homework }

Here's what I got:

![Training a Regression Model on all Taxi Trips](../img/evaluate-parquet.jpg)
{.img-fluid .mb-4}

The R-squared value is **0.872** which means that the model explains 87% of the variance in the fare amount. This is quite a bit lower than the previous result of 0.992, but still a good result. It suggests that the previous model trained on 10,000 trips was indeed overfitting, and we are now looking at a more realistic result. 

The mean absolute error (MAE) is **$2.818**. It increased sevenfold compared to the previous MAE of 0.425. In this dataset we have a lot more variety: long trips, airport trips, surcharges, and possibly mislabeled fares. This naturally increases prediction difficulty. It's likely the model is not handling outliers, long-distance trips, or rare cases well.

The root mean squared error (RMSE) is **$4.565**. The fact that the RMSE is twice as large as the MAE indicates that there are outliers where the model makes large errors in its prediction. This suggests that we may have to adjust our data filters for the new dataset. 

#### Conclusion 

All in all, this was quite a step backwards. And this is not surprising. We designed the data transformation pipeline specifically for a dataset of only 10,000 trips. Even worse, all of these trips were early in the morning on December 1st, 2018. We made many assumptions based off the histograms, correlation matrix and scatterplot grid and baked those assumptions into the design of the machine learning pipeline.

But now we have a dataset that covers the full month of December. The patterns in this dataset might be completely different, and we'll have to revisit all of our previous assumptions to check if they are still valid for this much larger dataset. 

Be very careful when you design a data transformation pipeline for a partial subset of a dataset. The histograms and the correlation matrix can change dramatically when loading the full dataset, and you may have to revist all your prior assumptions about the data.
{ .tip }

In the next lesson, we'll quickly regenerate the histogram grid, the Pearson correlation matrix and the scatterplot grid to see if the data transformation pipeline need to be changed. 

{{< /encrypt >}}