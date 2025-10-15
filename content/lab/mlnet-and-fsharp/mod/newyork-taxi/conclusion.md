---
title: "Conclusion"
type: "lesson"
layout: "default"
sortkey: 170
---

This concludes the two regression modules in this lab. I hope you enjoyed training models on the California Housing and New York Taxi datasets. You now have hands-on experience building two F# apps that train models on a dataset. 

The New York TLC dataset is a nice example of a large training dataset. We have 8.1 million taxi trips in the month of December alone. If you wanted to train your model on all of 2018, you would have to deal with roughly 100 million rows of data. Very large datasets are common in machine learning. Computer vision models are routinely trained on 10 million images, and contemporary large language models are trained on pretty much the entire Internet!

If you need to produce other kinds of numerical predictions in the future, feel free to just copy and paste the code from the labs. The steps to build a regression pipeline are the same every time, all you need to tweak are the data processing steps, the learning algorithm and the hyperparameters.

I hope that you're starting to realize that machine learning applications are actually very simple. With just a few hundred lines of F# code, you can process a dataset, train a model, evaluate the metrics, and then start generating predictions. And F#'s functional programming features make the code much more concise and expressive than similar C# implementations.