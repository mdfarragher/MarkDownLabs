---
title: "California Housing - Part 1"
description: "Process a dataset with house prices across California neighborhoods"
type: "mod"
layout: "single"
image: mod-calhousing-1.jpg
sortKey: 20
---
In this lab, you're going to write an app in C# that use feature engineering to process the data in a machine learning dataset.

You'll use the California Housing Dataset, which is famous for its use in many machine learning courses. But the dataset has several issues that you'll need to discover and mitigate before you can use it for machine learning training.

You will have to detect and deal with outliers, scale and normalize columns to a sane numeric range, bin- and one-hot encode categorical data columns, and cross latitude and longitude columns if present.

The California Housing dataset is perfect for practicing your feature engineering skills. It's virtually unprocessed and requires lots of transformation steps before it is suitable for machine learning training.