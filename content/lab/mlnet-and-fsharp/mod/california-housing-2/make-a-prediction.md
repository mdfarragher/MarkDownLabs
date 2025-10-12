---
title: "Make A Prediction"
type: "lesson"
layout: "default"
sortkey: 40
---

To wrap up, let's use the model to make a prediction.

We're going to invent a fake housing block in San Francisco, in the middle of the Mission district. The block has 2500 rooms, 1000 bedrooms, houses 500 people and 150 households. The apartments are 10 years old on average, and the normalized median income in that neighborhood is 2.0.

For how much could you sell an apartment in that housing block?

We will ask our AI agent to write code that prompts us for all the properties of a single housing block, and then we'll use the machine learning model to predict what the median house value will be for any apartment in the block.

{{< encrypt >}}

#### Make A Prediction

Enter the following prompt:

"Add code to prompt the user for all the properties of a single housing block, and then use the model to generate a prediction of the median house value."
{ .prompt }

The agent will modify the `TransformedHousingData` class and add a `Score` property to hold the generated prediction:

```fsharp
// Type to hold transformed data
[<CLIMutable>]
type TransformedHousingData = {
    mutable longitude: float32
    // ... other properties ...

    mutable Score: float32
}
```

And then it will add code like this to make the prediction:

```fsharp
// Create housing block with data from user
printf "Total rooms: "
let rooms = stdin.ReadLine() |> float32
printf "Total bedrooms: "
let bedrooms = stdin.ReadLine() |> float32
// ... continue for other properties ...

// Create input data
let inputData: HousingData = {
    row_id = 0.0f
    total_rooms = rooms
    total_bedrooms = bedrooms
    // ... set other properties from user input ...
}

// Create prediction engine
let predictionEngine = mlContext.Model.CreatePredictionEngine<HousingData, TransformedHousingData>(model)

// Make prediction
let prediction = predictionEngine.Predict(inputData)
printfn "Predicted house value: $%.2f" prediction.Score
```

The `CreatePredictionEngine` method sets up a prediction engine. The two type arguments are the input data class and the class to hold the transformed data with the prediction.

With the prediction engine set up, a call to `Predict` is all you need to make a single prediction. The prediction value is then available in the `Score` property.

Let's try this for the fake apartment I introduced earlier. Here is the data you need to enter:

- Longitude = 37.760653
- Latitude = -122.418725
- HousingMedianAge = 10
- TotalRooms = 2500
- TotalBedrooms = 1000
- Population = 500
- Households = 150
- MedianIncome = 2.0

And this is the output I get:

![Using The Model To Make A Prediction](../img/prediction.jpg)
{ .img-fluid .mb-4 }

I get a predicted median house value of **$124,180**.

What prediction did you get? Try changing the input data to see how this affects the predicted house value. Do the changes in prediction value make sense to you?
{ .homework }

#### Summary

Making predictions with the fully trained machine learning model is super easy. All you need to do is set up a prediction engine and then feed data into it. The engine will perform the exact same data transformations you used when training the model, and this ensures that the predictions make sense.

Using a machine learning model to generate predictions is called inference, and it requires a lot less compute capacity than training the model did. The inference compute load is often thousands of times smaller than the training load.

So it makes perfect sense to run a model in production, where you can quickly initialize it by importing the weights from a file. If over time the data changes significantly and the prediction quality starts dropping (this is called **data drift**), you can re-train the model on a separate compute platform and copy the new weight file back to production. 

{{< /encrypt >}}