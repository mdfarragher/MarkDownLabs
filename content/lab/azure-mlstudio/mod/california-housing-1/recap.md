---
title: "Recap"
type: "lesson"
layout: "default"
sortkey: 900
---

Congratulations on finishing the lab. Here's what you have learned.

{{< encrypt >}}

You learned how to analyze the **California Housing dataset**, both automatically by having an AI agent scan the data for you, and manually by inspecting the data by hand. 

You created an **Azure Storage Account** to hold uploaded machine learning datasets, and referenced it from an **Azure Machine Learning Datastore**. Then you created a **Dataset** for the California Housing data and uploaded the datafile to the cloud. 

You created a **Profile** of the data and discovered that the dataset needs a lot of preprocessing before it can be used for machine learning training. The data needs to be **normalized** and there are **outlier** blocks with a very high population and total number of rooms. 

Finally, you created an **Azure Machine Learning Pipeline** to load the dataset, normalize the data, filter outliers, and create a new column called rooms per person. 

{{< /encrypt >}}
