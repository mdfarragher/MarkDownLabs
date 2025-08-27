---
title: "Train A Regression Model"
type: "lesson"
layout: "default"
sortkey: 10
---

We're going to continue with the code we wrote in the previous lab. That C# application set up a data transformation pipeline to load the California Housing dataset and clean up the data using several feature engineering techniques.

So all we need to do is append a few command to the end of the pipeline to train and evaluate a regression model on the data.

{{< encrypt >}}

#### Split The Dataset

But first, we need to split the dataset into two partitions: one for training and one for testing. The training partition is typically a randomly shuffled subset of around 80% of all data, with the remaining 20% reserved for testing.

We do this, because sometimes a machine learning model will memorize all the labels in a dataset, instead of learning the subtle patterns hidden in the data itself. When this happens, the model will produce excellent predictions for all the data it has been trained on, but struggle with data it has never seen before.

By keeping 20% of our data hidden from the model, we can check if this unwanted process of memorization (called **overfitting**) is actually happening.

So let's split our data into an 80% partition for training and a 20% partition for testing.

Open the code from the previous lesson in Visual Studio Code. Keep the data transformation pipeline intact, but remove any other code you don't need anymore.

Then open the Copilot panel and type the following prompt:

"Split the data into two partitions: 80% for training and 20% for testing."
{ .prompt }

You should get the following code:

```csharp
// Shuffle the data randomly
var random = new Random(42);
var shuffledData = filteredData.OrderBy(x => random.Next()).ToList();

// Split into 80% training and 20% testing
var splitIndex = (int)(shuffledData.Count * 0.8);
var trainingData = shuffledData.Take(splitIndex).ToList();
var testingData = shuffledData.Skip(splitIndex).ToList();

Console.WriteLine($"Training samples: {trainingData.Count}");
Console.WriteLine($"Testing samples: {testingData.Count}");
```

This code uses LINQ to randomly shuffle the data and then splits it into training (80%) and testing (20%) partitions. We set a random seed for reproducible results.


#### Add A Machine Learning Algorithm

Now let's add a machine learning algorithm to the pipeline.

"Add code to set up a linear regression algorithm using TensorFlow.NET."
{ .prompt }

That should produce the following code:

```csharp
// Train-test split using NumSharp (sklearn-style)
var n_samples = scaled_features.shape[0];
var train_size = (int)(n_samples * 0.8);

// Shuffle and split
var indices = np.arange(n_samples);
np.random.shuffle(indices);

var train_idx = indices[:train_size];
var test_idx = indices[train_size:];

// Create train/test sets
var X_train = scaled_features[train_idx];
var y_train = median_house_value[train_idx].reshape(-1, 1);
var X_test = scaled_features[test_idx];
var y_test = median_house_value[test_idx].reshape(-1, 1);

var n_features = X_train.shape[1];

// Create TensorFlow model (Python TensorFlow style)
var X = tf.placeholder(tf.float32, shape: (None, n_features), name: "X");
var y = tf.placeholder(tf.float32, shape: (None, 1), name: "y");

// Model parameters (Python TensorFlow style)
var W = tf.Variable(tf.random.normal((n_features, 1), stddev: 0.1), name: "weights");
var b = tf.Variable(tf.zeros((1,)), name: "bias");

// Linear model: predictions = X @ W + b (Python @ operator style)
var predictions = tf.add(tf.matmul(X, W), b, name: "predictions");

// Loss function: MSE (Python TensorFlow style)
var loss = tf.reduce_mean(tf.square(predictions - y), name: "mse_loss");

// Optimizer (Python style)
var learning_rate = 0.01;
var optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss);
```

This code sets up a linear regression model using TensorFlow.NET. We define placeholders for input features (X) and target values (Y), create weight and bias variables, and set up the linear model with mean squared error loss and gradient descent optimization.


Be careful when you run this prompt! My AI agent generated code that included all features, including **MedianHouseValue**, **Latitude**, **Longitude**, **TotalRooms**, **TotalBedrooms**, **Population**, **Househoulds** and the encoded latitude and longitude.

This is obviously wrong, as the location cross product replaces all other latitude and longitude columns, and **RoomsPerPerson** replaces all other room- and person-related columns.

Even worse, did you notice the **MedianHouseValue** column in that list? This is the label that we're trying to predict. If we train a model on the label itself, the model can simply ignore all other features and output the label directly. This is like asking the model to make a prediction, and then giving it the actual answer it is supposed to predict. 

Always be vigilant. AI agents can easily make mistakes like this, because they do not understand the meaning of each dataset column. Your job as a data scientist is to make sure that the generated code does not contain any bugs.
{ .tip }

Linear regression with gradient descent is a fundamental machine learning algorithm that finds the optimal weights by iteratively minimizing the cost function. TensorFlow.NET implements this efficiently using automatic differentiation. If you're interested in the mathematical foundations, you can read more about linear regression on Wikipedia:

https://en.wikipedia.org/wiki/Linear_regression


#### Train A Machine Learning Model

Now let's train a machine learning model using our data transformation pipeline and the regression algorithm:

"Train a machine learning model on the training set using the regression algorithm."
{ .prompt }

That will produce the following code:

```csharp
// Training session (Python TensorFlow style)
using var sess = tf.Session();
sess.run(tf.global_variables_initializer());

// Training hyperparameters
var epochs = 1000;
var print_every = 100;

// Training loop (Python style)
Console.WriteLine("Training started...");
for (int epoch in range(1, epochs + 1))  // Python range style
{
    // Forward pass and backpropagation
    var feed_dict = new FeedDict {
        [X] = X_train,
        [y] = y_train
    };
    
    var (_, loss_val) = sess.run((optimizer, loss), feed_dict);
    
    // Print progress (Python style)
    if (epoch % print_every == 0)
    {
        Console.WriteLine($"Epoch {epoch}/{epochs}, Loss: {loss_val:.4f}");
    }
}

Console.WriteLine("Training completed!");
```

This training loop uses Python-style syntax with FeedDict for cleaner feed management, Adam optimizer (more common in Python), and Python-style string formatting for progress reporting.


#### Summary

In this lesson, you completed the machine learning pipeline you built in the previous lesson, by adding a machine learning algorithm. Then you split the data into a training and testing set, and trained a model on the training set.

In the next lesson, we'll calculate the prediction evaluation metrics to find out how good the model is at predicting house prices.

{{< /encrypt >}}