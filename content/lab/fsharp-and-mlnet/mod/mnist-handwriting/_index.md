---
title: "6. Recognize Handwriting"
description: "Recognize handwritten digits on checks and scanned forms"
type: "mod"
layout: "single"
image: mod-mnist.jpg
sortKey: 50
hideModule: true
---

In this lab, you're going to write an app in C# that can recognize handwritten digits on bank checks, printed documents, scanned forms and so on. You'll use the well-known MNIST dataset, which is often used to benchmark computer vision models. The dataset contains 60,000 images, each 28 by 28 pixels, of a handwritten digit from zero to nine. 

Computer vision datasets are always a difficult challenge. There's no point generating histograms or correlation matrices, because each feature represents one pixel in an image. It makes no sense to remove 'outlier pixels' or to check if individual pixels have a linear relationship with the label to predict.

We want the machine learning model to recognize complex visual patterns across groups of pixels in the image. The go-to learning algorithm for this type of task is the neural network. So in this lab, you'll train a neural network on the MNIST dataset. Next, you'll evaluate the performance of the model by calculating the classification metrics, and use the fully trained model to generate predictions for a sample of handwritten digits. 
