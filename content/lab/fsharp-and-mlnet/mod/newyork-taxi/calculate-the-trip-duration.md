---
title: "Calculate The Trip Duration"
type: "lesson"
layout: "default"
sortkey: 50
---

In the New York TLC dataset, we have a set of columns that at first glance appear to be highly correlated: 

- **tpep_pickup_datetime**
- **tpep_dropoff_datetime**
- **trip_distance**

We would normally expect long-duration taxi trips to cover a large distance and short trips to only cover a small distance. So it's reasonable to assume that trip duration and trip distance are strongly positively correlated. 

{{< encrypt >}}

The confusion matrix will tell us how strong the correlation is between trip duration and trip distance, but for that, we first need to calculate the duration. Fortunately we have the trip pickup and dropoff date and time, so this should be easy.

#### Calculate the Trip Duration

Open the Copilot panel and enter the following prompt:

"Create a machine learning pipeline with a custom mapping that adds a new field called TripDuration to the TaxiTrip record type. The duration is the timespan between PickupDateTime and DropoffDateTime in minutes."
{ .prompt }

You may have to prod your AI agent a few times to generate the correct code. We want to see a new machine learning pipeline with the following custom mapping:

```fsharp
// Define a custom mapping to calculate TripDuration
let pipeline = 
    mlContext.Transforms.CustomMapping<TaxiTrip, TaxiTripWithDuration>(
        Action<TaxiTrip, TaxiTripWithDuration>(fun input output ->
            output.RowID <- input.RowID
            output.VendorID <- input.VendorID
            output.PassengerCount <- input.PassengerCount
            output.TripDistance <- input.TripDistance
            output.RatecodeID <- input.RatecodeID
            output.StoreAndFwdFlag <- input.StoreAndFwdFlag
            output.PULocationID <- input.PULocationID
            output.DOLocationID <- input.DOLocationID
            output.PaymentType <- input.PaymentType
            output.FareAmount <- input.FareAmount
            output.Extra <- input.Extra
            output.MtaTax <- input.MtaTax
            output.TipAmount <- input.TipAmount
            output.TollsAmount <- input.TollsAmount
            output.ImprovementSurcharge <- input.ImprovementSurcharge
            output.TotalAmount <- input.TotalAmount
            output.TripDuration <- float32 (input.DropoffDateTime - input.PickupDateTime).TotalMinutes
        ),
        contractName = null)

// Apply the transformation
let transformedData = pipeline.Fit(data).Transform(data)

// Extract all TaxiTrip instances with properties populated
let taxiTripsWithDuration = 
    mlContext.Data.CreateEnumerable<TaxiTripWithDuration>(transformedData, reuseRowObject = false)
    |> Seq.toList
```

Note the last line of the mapping which calculates the trip duration in minutes. Then a call to `Fit` runs the machine learning pipeline and produces `transformedData` (a dataview). And finally, a call to `CreateEnumerable` converts the dataview to a list of `TaxiTripWithDuration` objects.  

Note that we are not registering an assembly for this mapping, so we won't be able to save and load the weights of the fully trained machine learning model. Feel free to add this code yourself if you want maximum flexibility. 
{ .tip }

The `TaxiTripWithDuration` type looks like this:

```fsharp
// Taxi trip with trip duration
[<CLIMutable>]
type TaxiTripWithDuration = {
    mutable RowID: int
    mutable VendorID: int
    mutable PassengerCount: float32
    mutable TripDistance: float32
    mutable RatecodeID: int
    mutable StoreAndFwdFlag: string
    mutable PULocationID: int
    mutable DOLocationID: int
    mutable PaymentType: int
    mutable FareAmount: float32
    mutable Extra: float32
    mutable MtaTax: float32
    mutable TipAmount: float32
    mutable TollsAmount: float32
    mutable ImprovementSurcharge: float32
    mutable TotalAmount: float32
    mutable TripDuration: float32
}
```

With the mapping set up, it's now very easy to calculate a histogram of the new **TripDuration** column. Just add the following code to your main program:

```fsharp
// Plot and save histogram of trip duration
let plot = HistogramUtils.PlotHistogram<TaxiTripWithDuration> taxiTripsWithDuration "TripDuration"
plot.SavePng("tripduration-histogram.png", 600, 400) |> ignore
```

When you run this code, you'll get the following histogram:

![Histogram Of Trip Duration](../img/tripduration-histogram.png)
{ .img-fluid .mb-4 }

And look at that! The histogram has a ton of outliers beyond trip durations longer than 60 minutes. We should definitely consider filtering them out to improve the quality of the fare predictions. 

In fact, let's do that right now.

#### Remove Taxi Trips Longer Than 60 Minutes

We're going to add a filter transformation that removes all taxi trips longer than 60 minutes. Open the Copilot panel and enter the following prompt:

"Use the FilterRowsByColumn method to remove all taxi trips longer than 60 minutes."
{ .prompt }

That should give you the following code (the second line is new):

```fsharp
// Apply the transformation
let transformedData = pipeline.Fit(data).Transform(data)

// Filter out taxi trips longer than 60 minutes
let filteredData = mlContext.Data.FilterRowsByColumn(transformedData, "TripDuration", upperBound = 60.0)

// Extract all TaxiTrip instances with properties populated
let taxiTripsWithDuration = 
    mlContext.Data.CreateEnumerable<TaxiTripWithDuration>(filteredData, reuseRowObject = false)
    |> Seq.toList
```

First, a call to `Fit` runs the pipeline and calculates the trip duration. Then the `FilterRowsByColumn` function filters the dataview by removing all outliers, and finally the `CreateEnumerable` function produces the list of taxi trips. 

When you run the new code, you'll get the following histogram:

![Histogram Of Trip Duration](../img/tripduration-histogram-trim.png)
{ .img-fluid .mb-4 }

Much better! This is a dataset column we can confidently train a machine learning model on. 

Now, we are finally ready to calculate the confusion matrix. 

{{< /encrypt >}}