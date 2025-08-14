---
title: "Load The Full Dataset"
type: "lesson"
layout: "default"
sortkey: 145
---

So far, we have been working with a subset of the MNIST dataset. This subset contains only 10,000 rows, but the official MNIST dataset actually contains 60,000 images for training and 10,000 images for testing. 

So let's download the full dataset and see how our app holds up. This [Github repository](https://github.com/phoebetronic/mnist) contains the full dataset files. Download the **mnist_train.zip** and **mnist_test.zip** files and unzip them to your project folder.

#### Load The Full Dataset

Now let's alter the app so that we load all 70,000 images. Enter the next prompt to refactor the application:

"Add code that alters the app so that it load the mnist_train.csv and mnist_test.csv files. Remove the RowID column (because it's not in these files) and adjust the LoadColumn attributes accordingly. Do not split the data, just use mnist_train to train the model and mnist_test to test the model. Fix all compiler errors due to this change."
{ .prompt }

This is a straightforward refactor, but the AI agent has to jiggle some variables around to make everything work. 

Homework: refactor your app to load the full dataset. Then run the app and examine the new evaluation metrics. What do you notice? Write down your observations.
{ .homework }

#### Evaluation Results

Here's what I got:

![Training a Model on all Images](../img/evaluate-full-1.png)
{.img-fluid .mb-4}

The macro accuracy is **91.0%**, slightly down from 91.13% for the partial dataset. This is still a great result. The model is performing well across all digits, with no single class dragging the average down too much.

The micro accuracy is **91.09%**, slightly down from 91.34% for the partial dataset. This is also great. Over 9 out of 10 predictions are correct, which is great for a model trained on 60,000 images. Also note that the micro- and macro accuracies are still close together, which means that the full dataset does not have a class-imbalance, just like the partial dataset.

The log loss is **0.3007**, slightly lower than 0.3538 for the partial dataset. Lower is better, so the probability outputs are now slightly better calibrated and confident when correct.

The log loss reduction is **0.8693**, slightly higher than 0.8460 for the partial dataset. Higher is better, so the model now shows an 86.93% improvement over baseline.

Remember the New York TLC dataset, where the results got worse when we loaded the full dataset? Well, here the opposite happened: we got slightly better results training on the full 60,000 images. It means that even with only 10,000 images, the model picked up enough patterns to be able to recognize every digit. The additional 50,000 training images were just more of the same, so the model hardly improved (or regressed).

Here's the next part of the output:

![Training a Model on all Images](../img/evaluate-full-2.png)
{.img-fluid .mb-4}

The most common incorrect prediction is still a '9' where the actual image turned out to be a '4'. This now happens 58 times. The sampled image has almost equal scores for 4 and 9: respectively **29.0%** and **29.23%**. The '9' prediction wins out by online a tiny margin. 

The next two scores, in descending order, are '8' (15.56%) and '6' (14.10%). 

And these scores make perfect sense when you look at the actual image:

![Training a Model on all Images](../img/evaluate-full-3.png)
{.img-fluid .mb-4}

That could definitely be a '4', or a '9'.

#### The Confusion Matrix

Here is the confusion matrix I got when training on the full MNIST dataset:

![The Confusion Matrix](../img/confusion-full.png)
{.img-fluid .mb-4}

The main diagonal contains all correct predictions, and each cell is deep black as expected. The matrix cells off the main diagonal describe incorrect predictions, and from the plot we can quickly see that the most popular mistakes are:

- The model predicts a '9' but the digit is a '4' (58 times)
- The model model predicts a '5' but the digit is an '8' (46 times)
- The model predicts an '8' but the digit is actually a '2' (46 times)
- The model predicts a '5' but the digit is actually a '3' (45 times)
- The model predicts a '9' but the digit is actually a '7' (44 times)

We've seen these mistakes before. There are a bit more of them now, but the total number of incorrect predictions is still quite small compared to the number of correct predictions (visible as the color contrast between the main diagonal and the other matrix cells).

All in all this is a pretty good result. We trained on the full dataset and everything is fine!
