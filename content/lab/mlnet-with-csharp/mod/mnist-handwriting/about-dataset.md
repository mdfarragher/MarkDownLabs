---
title: "The MNIST Handwriting Dataset"
type: "lesson"
layout: "default"
sortkey: 10
---
Digit recognition systems are machine learning models that are trained to recognize digits from many different sources like emails, bank checks, papers, PDF documents, images, etc. These systems have many real-world applications, for example in processing bank checks, scanning PDF documents, recognizing handwritten notes on tablets, read number plates of vehicles, process tax forms and so on.

Handwriting recognition is a hard challenge to solve because handwritten digits are not always the same size and orientation, and handwriting tends to differ from person to person. Many people write a single digit with a variety of different handwriting styles.

MNIST is the de facto "Hello World" dataset of computer vision. Since its release in 1999, this classic dataset of handwritten digits has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and students alike.

![Cleveland CAD Dataset](../img/data.jpg)
{ .img-fluid .pb-4 }

The dataset was created in 1999 by mixing handwriting samples from American Census Bureau employees and American high school students. The black and white images of handwritten digits were normalized to fit into a 28x28 pixel bounding box and anti-aliased to introduce grayscale levels.

In this assignment, you are going to build a C# app that trains a machine learning model to recognize the MNIST digits.