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

My AI agent implemented the cross product like this:

```csharp
// Calculate cross product of latitude and longitude encodings  
var crossedData = encodedData.Select(x => new 
{
    x.Original,
    LocationCross = x.LatitudeEncoded
        .SelectMany((lat, i) => x.LongitudeEncoded.Select((lon, j) => lat * lon))
        .ToArray()
}).ToList();

Console.WriteLine("Sample cross products:");
for (int i = 0; i < 3; i++)
{
    var sample = crossedData[i].LocationCross;
    var nonZeroIndex = Array.IndexOf(sample, 1.0f);
    Console.WriteLine($"Record {i}: Non-zero at index {nonZeroIndex}");
}
```

This code creates a 100-element cross product vector by multiplying each element of the latitude encoding with each element of the longitude encoding. This creates a 10x10 grid representation where exactly one cell has value 1.0, representing the housing block's location.


You should get something like the following output:

![Cross Of Latitude And Longitude](../img/cross-console.png)
{ .img-fluid .mb-4 }

You can see from the three sampled test rows that the cross product is a 100-element one-hot encoded vector, and that we have only a single '1' value in each vector.

#### Summary

Vector crossing is a handy trick for dealing with latitude and longitude features. Instead of training a machine learning model on two separate 10-element vectors, we now train on a single 100-element vector.

This gives a machine learning model the freedom to treat each grid cell independently from all others. For example, a model could learn that housing blocks in San Francisco are very expensive, but if you travel a couple of miles east, the price drops rapidly. The model will be able to optimize predictions for these two regions independently.
 
{{< /encrypt >}}