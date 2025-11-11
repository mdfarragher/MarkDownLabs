---
title: "The MNIST Handwriting Dataset"
type: "lesson"
layout: "default"
sortkey: 10
---

# The MNIST Handwriting Dataset

Optical Character Recognition (OCR) systems are machine learning models that are trained to recognize written text. These systems have many real-world applications, for example in scanning books, printed documents and receipts, processing bank checks and forms, reading car license plates and much more. 

Processing handwriting is an expecially hard challenge to solve, because the letters and numbers are not always the same size and the writing style tends to differ from person to person. In this field, the MNIST dataset is famous. Since its release in 1999, this classic dataset of handwritten digits has served as the basis for benchmarking OCR systems. 

![MNIST Dataset](../img/data.jpg)
{ .img-fluid .pb-4 }

The dataset was created in 1999 by mixing handwriting samples from American Census Bureau employees and American high school students. The black and white images of handwritten digits were normalized to fit into a 28x28 pixel bounding box and anti-aliased to introduce grayscale levels.

In this assignment, you are going to build an F# app that trains a machine learning model to recognize the handwritten digits in the MNIST dataset.