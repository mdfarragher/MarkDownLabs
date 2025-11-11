---
title: "Cross Latitude And Longitude"
type: "lesson"
layout: "default"
sortkey: 80
---

Now let's perform one final data transformation: we're going to calculate the cross product of the encoded latitude and longitude, to create a new 100-element vector of zeroes and ones. We're layering a 10x10 grid over the state of California and placing a single '1' value in the grid to indicate the location of the housing block.

Let's get started.

{{< encrypt >}}

#### Cross The Latitude and Longitude

Open the Copilot panel in Visual Studio Code and enter the following prompt:

"Add a step to the transformation pipeline to calculate a vector cross product of LatitudeEncoded and LongitudeEncoded, creating a new vector with 100 elements."
{ .prompt }

For me, this prompt worked perfectly on the first try. Copilot implemented the cross product like this:

```fsharp
// Compute cross product of LatitudeEncoded and LongitudeEncoded (10x10=100 elements)
.Append(mlContext.Transforms.CustomMapping<TransformedHousingData, TransformedHousingData>(
    (fun input output ->
        
        // Calculate cross product (outer product) of two vectors
        output.location_cross <- Array.create 100 0.0f
        for i in 0 .. input.latitude_encoded.Length - 1 do
            for j in 0 .. input.longitude_encoded.Length - 1 do
                output.location_cross.[i * 10 + j] <-
                    input.latitude_encoded.[i] * input.longitude_encoded.[j]
        
        // Copy all other fields from input to output
        output.housing_median_age <- input.housing_median_age
        output.median_income <- input.median_income
        output.median_house_value <- input.median_house_value
        output.rooms_per_person <- input.rooms_per_person
        output.latitude_encoded <- input.latitude_encoded
        output.longitude_encoded <- input.longitude_encoded
    ),
    "CrossProductMapping"))
```

It's another `CustomMapping` that uses a nested for-loop to manually calculate the vector cross product and store the result in a new field called **location_crossed**. 

The `TransformedHousingData` type has been slightly modified and now looks like this:

```fsharp
// Type to hold transformed data
[<CLIMutable>]
type TransformedHousingData = {
    mutable longitude: float32
    mutable latitude: float32
    mutable housing_median_age: float32
    mutable median_income: float32
    mutable median_house_value: float32

    [<VectorType(10)>] mutable latitude_encoded: float32[]
    [<VectorType(10)>] mutable longitude_encoded: float32[]
    [<VectorType(100)>] mutable location_cross: float32[]
    mutable rooms_per_person: float32
}
```

Note the new `location_cross` field with a `VectorType` identifying it as a 100-element mutable array of floats. 

You should get something like the following output:

![Cross Of Latitude And Longitude](../img/cross-console.png)
{ .img-fluid .mb-4 }

You can see from the sampled test row that the cross product is a 100-element one-hot encoded vector, and that we have only a single '1' value.

#### Summary

Vector crossing is a handy trick for dealing with latitude and longitude features. Instead of training a machine learning model on two separate 10-element vectors, we now train on a single 100-element vector.

This gives a machine learning model the freedom to treat each grid cell independently from all others. For example, a model could learn that housing blocks in San Francisco are very expensive, but if you travel a couple of miles east, the price drops rapidly. The model will be able to optimize predictions for these two regions independently.

Unfortunately, Microsoft.ML has no built-in transformation to calculate a feature cross, so we had to implement it manually using a custom transformation.

{{< /encrypt >}}