---
title: "Calculate The Trip Duration"
type: "lesson"
layout: "default"
sortkey: 41
---

In the New York TLC dataset, we have a set of columns that at first glance appear to be highly correlated: 

- **tpep_pickup_datetime**
- **tpep_dropoff_datetime**
- **trip_distance**

We would normally expect long-duration taxi trips to cover a large distance and short trips to only cover a small distance. So it's reasonable to assume that trip duration and trip distance are strongly positively correlated. 

The confusion matrix will tell us how strong the correlation is between trip duration and trip distance, but for that, we first need to calculate the duration. Fortunately we have the trip pickup and dropoff date and time, so this should be easy.


#### Calculate the Trip Duration

Open the Copilot panel and enter the following prompt:

"Create a machine learning pipeline with a custom mapping that adds a new property called TripDuration to the TaxiTrip class. The duration is the timespan between PickupDateTime and DropoffDateTime in minutes."
{ .prompt }

You may have to prod your AI agent a few times to generate the correct code. We want to see a new machine learning pipeline with the following custom mapping:

```csharp
// Define a custom mapping to calculate TripDuration
var pipeline = mlContext.Transforms.CustomMapping<TaxiTrip, TaxiTripWithDuration>(
    (input, output) =>
    {
        output.RowID = input.RowID;
        output.VendorID = input.VendorID;
        output.PickupDateTime = input.PickupDateTime;
        output.DropoffDateTime = input.DropoffDateTime;
        output.PassengerCount = input.PassengerCount;
        output.TripDistance = input.TripDistance;
        output.RatecodeID = input.RatecodeID;
        output.StoreAndFwdFlag = input.StoreAndFwdFlag;
        output.PULocationID = input.PULocationID;
        output.DOLocationID = input.DOLocationID;
        output.PaymentType = input.PaymentType;
        output.FareAmount = input.FareAmount;
        output.Extra = input.Extra;
        output.MtaTax = input.MtaTax;
        output.TipAmount = input.TipAmount;
        output.TollsAmount = input.TollsAmount;
        output.ImprovementSurcharge = input.ImprovementSurcharge;
        output.TotalAmount = input.TotalAmount;
        output.TripDuration = (float)(input.DropoffDateTime - input.PickupDateTime).TotalMinutes;
    },
    contractName: null);

// Apply the transformation
var transformedData = pipeline.Fit(data).Transform(data);

// Extract all TaxiTrip instances with properties populated
var taxiTrips = mlContext.Data.CreateEnumerable<TaxiTripWithDuration>(transformedData, reuseRowObject: false).ToList();
```

Note the last line of the mapping which calculates the trip duration in minutes. Then a call to `Fit` runs the machine learning pipeline and produces `transformedData` (a dataview). And finally, a call to `CreateEnumerable` converts the dataview to a list of `TaxiTripWithDuration` objects.  

Note that we are not registering an assembly for this mapping, so we won't be able to save and load the weights of the fully trained machine learning model. Feel free to add this code yourself if you want maximum flexibility. 
{ .tip }

The `TaxiTripWithDuration` class looks like this:

```csharp
public class TaxiTripWithDuration : TaxiTrip
{
    public float TripDuration { get; set; }
}
```

With the mapping set up, it's now very easy to calculate a histogram of the new **TripDuration** column. Just add the following code to your main program:

```csharp
// Plot and save histogram of trip duration
var plot = HistogramUtils.PlotHistogram<TaxiTripWithDuration>(taxiTrips, "TripDuration");
plot.SavePng("tripduration-histogram.png", 600, 400);
```

When you run this code, you'll get the following histogram:

![Histogram Of Trip Duration](../img/tripduration-histogram.png)
{ .img-fluid .mb-4 }

And look at that! The histogram has a ton of outliers beyond trip durations longer than 60 minutes. We should definitely consider filtering them out to improve the quality of the fare predictions. 

In fact, let's do that right now.

#### Remove Taxi Trips Longer Than 60 Minutes

We're going to add a filter transformation that removes all taxi trips longer than 60 minutes. Open the Copilot panel and enter the following prompt:

"Use the FilterRowsByColumn method to remove all taxi trips longer than 60 mintues."
{ .prompt }

That should give you the following code (the second line is new):

```csharp
// Apply the transformation
var transformedData = pipeline.Fit(data).Transform(data);

// Filter out taxi trips longer than 60 minutes
var filteredData = mlContext.Data.FilterRowsByColumn(transformedData, "TripDuration", upperBound: 60);

// Extract all TaxiTrip instances with properties populated
var taxiTrips = mlContext.Data.CreateEnumerable<TaxiTripWithDuration>(filteredData, reuseRowObject: false).ToList();
```

First, a call to `Fit` runs the pipeline and calculates the trip duration. Then the `FilterRowsByColumn` method filters the dataview by removing all outliers, and finally the `CreateEnumerable` method produces the list of taxi trips. 

When you run the new code, you'll get the following histogram:

![Histogram Of Trip Duration](../img/tripduration-histogram-trim.png)
{ .img-fluid .mb-4 }

Much better! This is a dataset column we can confidently train a machine learning model on. 

Now, we are finally ready to calculate the confusion matrix. 