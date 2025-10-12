---
title: "Save And Load The Model"
type: "lesson"
layout: "default"
sortkey: 30
---

When you have a machine learning model with good prediction quality, you may want to save the model to a file so that you can easily use it later.

Saving a model will export all of the internal model weights, which represent the knowledge the model has gathered during the training. These weights are just a series of numbers, and saving these numbers to a file safeguards this knowledge and makes it available for later use.

{{< encrypt >}}

When we want to use a model to make predictions, we can simply set up a blank machine learning model, and then load knowledge into it by importing the weights back in to the model. This bypasses the entire training process, which is great because training a large model can sometimes take weeks or months!

Let's enhance our app with some simple code to save the weights of the fully trained model to a file.

#### Save The Model

Open Visual Studio Code and enter this prompt in the Copilot panel:

"Add code to save the fully trained model to a file."
{ .prompt }

Your agent should generate the following code:

```fsharp
// Save the trained model to a file
let modelPath = Path.Combine(Directory.GetCurrentDirectory(), "CaliforniaHousingModel.zip")
mlContext.Model.Save(regressionModel, transformedData.Schema, modelPath)
```

Saving a model is super easy. The `Save` method takes three arguments, a model instance, the dataset schema, and the path to save the weights to.

The generated ZIP file looks like this:

![Model Zip File Contents](../img/model-zip.jpg)
{ .img-fluid .mb-4 }

The archive contains a **Version.txt** file, a **Schema** file that describes the dataset schema, a list of subfolders that describe each data transformation in the pipeline, and a **Model.key** file with the trained model weights.

The model weights are stored in a Microsoft-specific format. This is fine if you are only using the ML.NET library and you're not transferring knowledge between models running on different machine learning libraries.

However, you can transfer knowledge if you want to. There's a special universal file format called ONNX that you can use to transfer knowledge between machine learning models running on different platforms.

#### Save The Model In ONNX Format

So let's modify our app to use the ONNX format instead. Enter the following prompt:

"Add code to save the fully trained model to a file in the ONNX format."
{ .prompt }

Your AI agent will discover that ONNX is not supported in ML.NET and requires a separate NuGet package. You'll see something like this in the chat:

_"It seems that the ConvertToOnnx method is not available in the current ML.NET version or setup. To save the model in ONNX format, you may need to use the Microsoft.ML.OnnxConverter package."_

And then your agent will ask to execute the following command:

```bash
dotnet add package Microsoft.ML.OnnxConverter
```

This is where agentic coding really shines. The AI agent analyzed the code, discovered that it needed an additional package, installed the package, and then added the code to save the model. It might even have built your code to check that there are no compile errors, or ran the code to check that it really does output an .onnx file.

Don't hesitate to ask the AI agent to build or run your code to check that it is working correctly. This is where unit tests come in really handy, you can tell the agent its code must pass all tests. 
{ .tip }

The code to save the model as an ONNX file looks like this:

```fsharp
// Save the trained model to a file in the ONNX format
let onnxPath = Path.Combine(Directory.GetCurrentDirectory(), "CaliforniaHousingModel.onnx")
let onnxStream = File.Create(onnxPath)
mlContext.Model.ConvertToOnnx(regressionModel, trainData, onnxStream)
```

Technically we should have used the `use` keyword to assign `onnxStream`, to make sure that the stream gets disposed as soon as possible. But this is not allowed in F# modules, so we have to use `let` instead. The stream will be disposed when the app terminates.

If you're interested. the Microsoft documentation for saving a model in the ONNX format is here: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/save-load-machine-learning-models-ml-net

#### Loading The Model

Let's add some code to load the model from a file. We can also have the app ask us if we want to train the model or simply load it from a file directly.

First, remove the ONNX code. This file format doesn't support all data transformations available in ML.NET, and we don't want distracting runtime errors while we expand the app. We'll only use the Microsoft format going forward.

Then, enter the following prompt:

"Add code to load the model from a file. When the app starts, ask the user if they want to train a model and save it to a file, or load the model from a file and use it to generate predictions."
{ .prompt }

This will add a query to your app, and based on your decision, it will either train the model and save it, or load the model and evaluate it.

The code to load a model from a file looks like this:

```fsharp
// Load the model
let modelPath = Path.Combine(Directory.GetCurrentDirectory(), "CaliforniaHousingModel.zip")
let schemaRef = ref Unchecked.defaultof<DataViewSchema>
let model = mlContext.Model.Load(modelPath, schemaRef)
let schema = !schemaRef
```

The `Load` method imports a model from a file and expects two arguments: the path of the file, and a variable reference to save the model schema. So we have to initialize a reference to a `DataViewSchema` first, pass it in as the second argument to `Load`, and then dereference the schema using the `!` operator.

Let's test the code. Here's what I get when I choose to train the model:

![Training And Saving The Model](../img/save-model.jpg)
{ .img-fluid .mb-4 }

But when loading the model, you're probably going to get a runtime error that looks like this:

![Error Message While Loading The Model](../img/load-model-error.jpg)
{ .img-fluid .mb-4 }

What's happening here is that ML.NET cannot run the custom mappings in the data transformation pipeline. If you look closely at the data loading code, you'll see that it is not using the pipeline at all. So the code has no idea how to run the custom mappings and aborts with an error message.

In my testing, I discovered that this bug is too complex for GPT 4.0, GPT 4.1 or Claude 3.7 to solve, and these agents very quickly destroyed my code while looking for a solution. So instead, I propose we debug and fix the code by hand.

It may be tempting to keep pushing your AI agent to fix the code automatically, but in cases like this when the agents are completely out of their depth, just debugging and fixing the code by hand is a lot faster. Plus, you'll learn something new too!
{ .tip }

The error message mentions an attribute `CustomMappingFactoryAttributeAttribute` that can be used to tag assemblies with custom mapping code. So, let's move the custom mappings to a new class and tag it with this attribute.

We have two mappings in our code: calculating rooms per person, and calculating the cross product of the latitude and longitude vectors.

Let me show you how to fix rooms per person, and then you can fix the other custom mappings in your code yourself:

```fsharp
[<CustomMappingFactoryAttribute("RoomsPerPersonMapping")>]
type RoomsPerPersonCustomAction() =
    inherit CustomMappingFactory<HousingData, TransformedHousingData>()
    
    override this.GetMapping() =
        Action<HousingData, TransformedHousingData>(fun input output ->
            output.longitude <- input.longitude
            output.latitude <- input.latitude  
            output.housing_median_age <- input.housing_median_age
            output.median_income <- input.median_income
            output.median_house_value <- input.median_house_value
            output.rooms_per_person <- if input.population > 0.0f then input.total_rooms / input.population else 0.0f
        )
```

This declares a new class `RoomsPerPersonCustomAction` with the code to perform the mapping.

Then in the pipeline, I can simply do this:

```fsharp
// Compute RoomsPerPerson
let pipeline = mlContext.Transforms.CustomMapping(
    RoomsPerPersonCustomAction().GetMapping(),
    "RoomsPerPersonMapping")
```

This is the same CustomMapping call as before, but now I provide the action by using the helper class I declared earlier.

There's one more step: I need to register the assembly that contains the custom mapping code. Here's how you do that:

```fsharp
// Register the assembly with custom conversions
mlContext.ComponentCatalog.RegisterAssembly(typeof<RoomsPerPersonCustomAction>.Assembly)
```

The `RegisterAssembly` method registers the assembly with the custom mapping code, so that the `Load` method can automatically find the mapping code when it is loading the model weights from a file.

It's all a bit convoluted, but with these fixes everything works.

Fix any custom mapping errors in your code with the technique I just showed you.
{ .homework }

By the way, Microsoft has helpful documentation about using custom mappings here: https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.custommappingcatalog.custommapping

With these fixes in place, the app now works flawlessly when I ask it to load the model from a file:

![Loading The Model](../img/load-model.jpg)
{ .img-fluid .mb-4 }

#### Summary

Saving the fully trained machine learning model is a great trick if you want to use the model to generate lots of predictions. Instead of having to train your model every time your app starts up, you can simply load all the knowledge into a blank model instantly.

We do this all the time in machine learning, because large complex models often require weeks (or even months) to train.

Do be careful with the ONNX format. It's great that we can transfer knowledge between models running on different platforms, but many data transformations (like custom mappings) are not supported. You would not be able to save and then reload the California Housing model in this format.

{{< /encrypt >}}