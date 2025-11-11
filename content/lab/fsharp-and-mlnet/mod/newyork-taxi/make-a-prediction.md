---
title: "Make A Prediction"
type: "lesson"
layout: "default"
sortkey: 110
---

To wrap up, let's use the model to make a prediction.

We're going to invent a fake taxi trip in New York City. I'm going to get into a cab at Times Square and take a trip to Washington Square Park. The trip covers 2.3 miles and takes 12 minutes. What's the fare I should expect to pay?

We will ask our AI agent to write code that prompts us for all the properties of a single taxi trip, and then we'll use the machine learning model to predict what the fare amount will be.

{{< encrypt >}}

#### Make A Prediction

Enter the following prompt:

"Add code to prompt the user for all the properties of a single taxi trip, and then use the model to generate a prediction of the fare amount. Ask only for the trip duration, trip distance, rate code ID and payment type."
{ .prompt }

The agent will create a new class `TaxiTripFarePrediction` with a property labelled `Score` to hold the generated prediction:

```fsharp
// Class to hold prediction
[<CLIMutable>]
type TaxiTripFarePrediction = {
    [<ColumnName("Score")>]
    PredictedFareAmount: float32
}
```

And then it will add code like this to make the prediction:

```fsharp
// Ask for trip details
printfn "\nEnter taxi trip details:"
printf "Pickup date and time: "
let pickupDt = stdin.ReadLine() |> DateTime.Parse
printf "Dropoff date and time: "
let dropoffDt = stdin.ReadLine() |> DateTime.Parse
// ... similar for other properties ...

// Create a trip record
let trip = {
    PickupDateTime = pickupDt
    DropoffDateTime = dropoffDt
    TripDistance = tripDistance
    RatecodeID = rateCodeId
    // ... similar for other properties ...
}

// Create prediction engine
let predictionEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model)

// Get user input and make prediction
let prediction = predictionEngine.Predict(trip)

// Display the prediction
printfn "\nThe predicted fare amount is: $%.2f" prediction.PredictedFareAmount
```

The `CreatePredictionEngine` method sets up a prediction engine. The two type arguments are the input data record type and the record type to hold the prediction.

With the prediction engine set up, a call to `Predict` is all you need to make a single prediction. The prediction value is then available in the `PredictedFareAmount` field.

Let's try this for the fake trip I took earlier. Here is the data you need to enter:

- Pickup date and time = 2018-12-01 00:10:00
- Dropoff date and time = 2018-12-01 00:22:00
- Trip distance = 2.3 miles
- Rate code ID = 1 (standard rate)
- Payment type = 1 (credit card)

And this is the output I get:

![Using The Model To Make A Prediction](../img/prediction.jpg)
{ .img-fluid .mb-4 }

I get a predicted fare amount of **$10.63**.

What prediction did you get? Try changing the input data to see how this affects the predicted fare amount. Do the changes in prediction value make sense to you?
{ .homework }

Next, let's load the full dataset of 8 million trips and re-run the app to discover the actual regression metrics and prediction accuracy. 

{{< /encrypt >}}