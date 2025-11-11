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

```csharp
// Save the trained model
var saver = tf.train.Saver();
var savePath = saver.save(sess, "./housing_model.ckpt");
Console.WriteLine($"Model saved to: {savePath}");

// Also save model metadata
using (var writer = new StreamWriter("model_info.txt"))
{
    writer.WriteLine($"Feature count: {numFeatures}");
    writer.WriteLine($"Training samples: {trainingData.Count}");
    writer.WriteLine($"Test samples: {testingData.Count}");
    writer.WriteLine($"Final training cost: {sess.run(cost, new FeedItem(X, trainX), new FeedItem(Y, trainY))}");
}
```

This code saves the trained TensorFlow model using tf.train.Saver, which creates checkpoint files containing the model's weights and graph structure. We also save metadata about the training process for future reference.

The model is saved in TensorFlow's checkpoint format (.ckpt), which includes the model weights, biases, and computational graph structure. This format allows for efficient loading and continued training if needed.

There's a special universal file format called ONNX that you can use to transfer knowledge between machine learning models running on different platforms. So let's modify our app to use the ONNX format instead. 

#### Save The Model In ONNX Format

Enter the following prompt:

"Add code to save the fully trained model to a file in the ONNX format."
{ .prompt }

The generated code should look like this:

```csharp
// Save model in ONNX format (requires tf2onnx converter)
// Note: This typically requires Python's tf2onnx package
// For demonstration, we show the conceptual approach:

// First save as TensorFlow SavedModel format
var builder = new SavedModelBuilder("./saved_model");
builder.add_meta_graph_and_variables(sess, new[] { "serve" });
builder.save();

Console.WriteLine("Model saved in SavedModel format");
Console.WriteLine("To convert to ONNX, use: python -m tf2onnx.convert --saved-model ./saved_model --output housing_model.onnx");
```

This code saves the model in TensorFlow's SavedModel format, which can then be converted to ONNX using external tools. The ONNX format enables cross-platform deployment and inference.

ONNX (Open Neural Network Exchange) is excellent for deploying models across different frameworks and platforms. However, the conversion process often requires additional tools and may not support all TensorFlow operations.
{ .tip }


#### Loading The Model

Let's add some code to load the model from a file. We can also have the app ask us if we want to train the model or simply load it from a file directly.

Enter the following prompt:

"Add code to load the model from a file. When the app starts, ask the user if they want to train a model and save it to a file, or load the model from a file and use it to generate predictions."
{ .prompt }

This will add a query to your app, and based on your decision, it will either train the model and save it, or load the model and evaluate it.

The code to load a model from a file looks like this:

```csharp
// Load previously saved model
public static Session LoadModel(string modelPath)
{
    var sess = tf.Session();
    var saver = tf.train.Saver();
    
    // Restore the model
    saver.restore(sess, modelPath);
    Console.WriteLine($"Model loaded from: {modelPath}");
    
    return sess;
}

// Usage:
if (File.Exists("./housing_model.ckpt.meta"))
{
    Console.WriteLine("Loading existing model...");
    var loadedSess = LoadModel("./housing_model.ckpt");
    // Use loadedSess for predictions
}
else
{
    Console.WriteLine("No existing model found. Please train a model first.");
}
```

This code demonstrates how to load a previously saved TensorFlow model using tf.train.Saver.restore(). The loaded session contains all the trained weights and can immediately be used for making predictions without retraining.

Let's test the code. Here's what I get when I choose to train the model:

![Training And Saving The Model](../img/save-model.jpg)
{ .img-fluid .mb-4 }

And this is what I get when I ask the app to load the model from a file:

![Loading The Model](../img/load-model.jpg)
{ .img-fluid .mb-4 }

#### Summary

Saving the fully trained machine learning model is a great trick if you want to use the model to generate lots of predictions. Instead of having to train your model every time your app starts up, you can simply load all the knowledge into a blank model instantly.

We do this all the time in machine learning, because large complex models often require weeks (or even months) to train.

Do be careful with the ONNX format. It's great that we can transfer knowledge between models running on different platforms, but many data transformations (like custom mappings) are not supported. You would not be able to save and then reload the California Housing model in this format.

{{< /encrypt >}}