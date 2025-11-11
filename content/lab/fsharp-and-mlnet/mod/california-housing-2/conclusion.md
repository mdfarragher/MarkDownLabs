---
title: "Conclusion"
type: "lesson"
layout: "default"
sortkey: 80
---

You now have hands-on experience building an F# app that trains a regression model on a dataset, and then using the fully trained model to generate predictions. F#'s functional programming features make the code more concise and expressive than equivalent C# implementations. This specific dataset, California Housing, required a lot of preprocessing and has features that only weakly correlate with the label. This makes predicting accurate house prices quite challenging. Nevertheless, the best mean absolute error you can achieve is around $28,000.

I hope you also noticed that you cannot simply keep increasing the number of longitude and latitude bins. There's an optimum, and once you go beyond that point, the predictions start degrading in quality again. Knowing where a house is with centimeter-level accuracy is not helpful at all, so we always want to work with a specific level of uncertainty to help our models make better predictions.

This was a fun lab, right?