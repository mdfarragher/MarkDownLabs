---
title: "Cross Latitude And Longitude"
type: "lesson"
layout: "default"
sortkey: 60
---

Now let's perform one final data transformation: we're going to calculate the cross product of the encoded latitude and longitude, to create a new 100-element vector of zeroes and ones. We're layering a 10x10 grid over the state of California and placing a single '1' value in the grid to indicate the location of the housing block.

Let's get started.

{{< encrypt >}}

#### Cross The Latitude and Longitude

Open the Copilot panel in Visual Studio Code and enter the following prompt:

"Add a step to the transformation pipeline to calculate a vector cross product of LatitudeEncoded and LongitudeEncoded, creating a new vector with 100 elements."
{ .prompt }

For me, this prompt worked perfectly on the first try. Claude implemented the cross product like this:

```csharp
// Compute cross product of LatitudeEncoded and LongitudeEncoded (10x10=100 elements)
.Append(mlContext.Transforms.CustomMapping<CrossProductInput, CrossProductOutput>(
(input, output) => 
{
    // Initialize the arrays if they're null
    output.LatitudeEncoded = input.LatitudeEncoded;
    output.LongitudeEncoded = input.LongitudeEncoded;
    output.LocationCrossProduct = new float[100];
    
    // Calculate cross product (outer product) of two vectors
    for (int i = 0; i < input.LatitudeEncoded?.Length; i++)
        for (int j = 0; j < input.LongitudeEncoded?.Length; j++)
            output.LocationCrossProduct[i * 10 + j] =
                input.LatitudeEncoded![i] * input.LongitudeEncoded![j];
    
    // Copy all other fields from input to output
    output.Longitude = input.Longitude;
    output.Latitude = input.Latitude;
    ....
},
"CrossProductMapping"));
```

It's another `CustomMapping` that uses a nested for-loop to manually calculate the vector cross product.

Note this line:

```csharp
input.LatitudeEncoded![i] * input.LongitudeEncoded![j]
```

The exclamation mark is the **null-forgiving** operator. It informs the C# compiler that `latitudeEncoded` or `longitudeEncoded` will always be initialized, and suppresses null warnings in this expression.

You should get something like the following output:

![Cross Of Latitude And Longitude](../img/cross-console.png)
{ .img-fluid .mb-4 }

You can see from the three sampled test rows that the cross product is a 100-element one-hot encoded vector, and that we have only a single '1' value in each vector.

#### Summary

Vector crossing is a handy trick for dealing with latitude and longitude features. Instead of training a machine learning model on two separate 10-element vectors, we now train on a single 100-element vector.

This gives a machine learning model the freedom to treat each grid cell independently from all others. For example, a model could learn that housing blocks in San Francisco are very expensive, but if you travel a couple of miles east, the price drops rapidly. The model will be able to optimize predictions for these two regions independently.

Unfortunately, Microsoft.ML has no built-in transformation to calculate a feature cross, so we had to implement it manually using a custom transformation.

{{< /encrypt >}}
