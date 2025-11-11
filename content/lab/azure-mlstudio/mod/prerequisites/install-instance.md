---
title: "Install The Compute Instance"
type: "lesson"
layout: "default"
sortkey: 20
---

Azure Machine Learning workspaces provide two types of compute resources for training machine learning models: compute instances and compute clusters. 

{{< encrypt >}}

- **Compute instances**: these are single virtual machines that host a Jupyter Labs installation that provides a cloud-based Python notebook environment for machine learning practicioners. We can  train small machine learning models on a compute instance.

- **Compute clusters**: these are scalable clusters of multiple virtual machines that can be used to train complex machine learning models on large datasets. Clusters expand and shrink automatically to handle variable workloads. 

In this lesson, you're going to set up a compute instance for model training. 

In the Azure Machine Learning Studio you'll notice a menu bar on the left. In this menu click on the **Compute** link (it's in the Manage section near the bottom). 

![Compute setup step 1](../img/image5.png)
{ .img-fluid .mb-4 }

You're now on the compute page. You'll notice that there are no compute instances configured yet and the list is empty. 

Make sure the Compute Instance tab is selected, and then click on the blue **+New** button:

![Compute setup step 1b](../img/image6.png)
{ .img-fluid .mb-4 }

To create a new compute instance, specify the following:

- The name of the new compute instance. Type a nice name like `compute-instance`.
- The virtual machine type, this can be either CPU or GPU. Select `CPU` for this training to keep costs low. 
- The virtual machine size. Click **Select from all options**  and then type `DS11` in the search field. The search result list will then show the `Standard_DS11_v2` size. Select it.

Note that we're selecting the smallest possible virtual machine size for this training in order to keep daily costs as low as possible. 

Now click the blue **Create** button.

![Compute setup step 2](../img/image8.png)
{ .img-fluid .mb-4 }

The compute instance will now be created, this can take a couple of minutes. When the instance is ready, you will see it appear in the compute instance list with the status 'Running':

![Compute setup step 2](../img/image10.png)
{ .img-fluid .mb-4 }

Compute instances are great for quick and dirty machine learning training. The instance is always running and immediately available for training. But for large datasets and complex machine learning algorithms like neural networks, we'll need a lot more compute capacity.

So let's also set up a compute cluster that can smoothly scale to any required cloud workload. 

{{< /encrypt >}}
