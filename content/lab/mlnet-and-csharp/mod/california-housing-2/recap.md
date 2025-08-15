---
title: "Recap"
type: "lesson"
layout: "default"
sortkey: 900
---

Congratulations on finishing the lab. Here's what you have learned.

You learned to **split the dataset** into separate parts for training and testing to prevent the machine learning model from memorizing the median house values for every single housing block in the dataset. You then used a pipeline to process the California housing data. Extending the pipeline you built in the previous lab, you added the `Concatenate` step to combine all dataset columns into a single feature column to train on.

You used a **training algorithm** (like SDCA) to train a regression model on the transformed data, and called the `Transform` and `Evaluate` methods to calculate the regression metrics and evaluate the quality of your predictions. You saved the fully trained model with the `Save` method, and reloaded it with the `Load` method. You also learned about the **ONNX** file format, which can be used to exchange weights between models running on different software platforms. You learned that custom data transformations cannot be reloaded, unless they are implemented in a separate class tagged with `CustomMappingFactoryAttributeAttribute`.

And finally, you learned how you can call the `CreatePredictionEngine` method to **generate predictions** with the fully-trained regression model.

You completed the lab by experimenting with different data processing steps and regression algorithms to find the best-performing model. 