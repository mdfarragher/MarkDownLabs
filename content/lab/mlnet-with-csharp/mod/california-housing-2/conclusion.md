---
title: "Conclusion"
type: "lesson"
layout: "exit"
sortkey: 910

pathColor: "blue"
courseName: "Supervised Machine Learning"
exitLink: "https://www.mdft.academy/view/courses/lectures-supervised-machine-learning/3119189-data-processing-labs/10149827-up-next"
---

You now have hands-on experience building a C# app that trains a regression model on a dataset, and then using the fully trained model to generate predictions.

California Housing is a difficult dataset, and predicting house prices is quite challenging for machine learning to get right. The best mean absolute error you can achieve is around $30,000.

I hope you also noticed that you cannot simply keep increasing the number of longitude and latitude bins. There's an optimum, and once you go beyond that point, the predictions start degrading in quality again. Knowing where a house is with centimeter-level accuracy is not helpful at all, so we always want to work with a specific level of uncertainty to help our models make better predictions.

This was a fun lab, right?
