---
title: "The California Housing Dataset"
type: "lesson"
layout: "default"
sortkey: 10
---

In machine learning circles, the **California Housing** dataset is a bit of a classic. It's the dataset used in the second chapter of Aurélien Géron's excellent machine learning book [Hands-On Machine learning with Scikit-Learn and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646).

The dataset serves as an excellent introduction to building machine learning apps because it requires rudimentary data cleaning, has an easily understandable list of variables and has the perfect size for fast training and experimentation. it was compiled by Pace, R. Kelley and Ronald Barry for their 1997 paper titled [Sparse Spatial Autoregressions](https://www.sciencedirect.com/science/article/abs/pii/S016771529600140X). They built it using the 1990 California census data.

![The California Housing Dataset](../img/data.jpg)
{ .img-fluid .pb-4 }

The dataset contains one record per census block group, with a census block group being the smallest geographical unit for which the U.S. Census Bureau publishes sample data. A census block group typically has a population of around 600 to 3,000 people.

In this lab, you're going to use the California Housing dataset to build a model that can predict the price of any house in the state of California.
