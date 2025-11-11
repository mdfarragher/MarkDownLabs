---
title: "Install The Azure Machine Learning Workspace"
type: "lesson"
layout: "default"
sortkey: 10
---

In this lesson you will set up the Azure Machine Learning workspace that you will use throughout the rest of this training. The workspace is a centralized place to manage all of the Azure machine learning resources you need to work on an AI project.

{{< encrypt >}}

Go to your Azure portal page. It should look something like this:

![Azure Portal](../img/portal.png)
{ .img-fluid .mb-4 }

This is a screenshot of my own Azure portal, yours will have a different list of Azure services and recent resources. 

Now click the Create A Resource button (with the blue plus sign icon) and search for "machine learning". Select the Machine Learning result from the dropdown list:

![Setup step 1](../img/step1.png)
{ .img-fluid .mb-4 }

You'll see an overview page with information about Azure Machine Learning. 

Click the Create button to confirm your choice:

![Setup step 2](../img/step2.png)
{ .img-fluid .mb-4 }

Now create a new machine learning workspace and provide the following details:

* The Azure subscription in which you will place the workspace. Select your subscription from the list.
* The resource group in which you will place the workspace. It's a good idea to create a new resource group for this training and give it a descriptive name, for example: 'rg-machinelearning'. 
* The name of the machine learning workspace. Give your workspace a nice unique name, perhaps by adding your initials too.
* The region in which to place the workspace. This corresponds to the Azure datacenter that will host your machine learning files. You should choose the location that's closest to where you are working to minimize latency.

The Storage account, Key vault and Application insights will be automatically filled in as you type the workspace name. You do not have to change these fields and can leave them at their default values.

You also don't need to change the Container registry field. We do not need a specific container registry in this training.

![Setup step 3](../img/image1.png)
{ .img-fluid .mb-4 }

Now click the blue Review+Create button, and then click Create. Your machine learning workspace will now be deployed. This can take a couple of minutes.

After deployment is complete, you'll be notified like in the image below. Click the Go To Resource button to access your new workspace.

![Setup step 4](../img/image2.png)
{ .img-fluid .mb-4 }

## Access the Azure Machine Learning Studio interface

You can now manage your workspace in the Azure portal, but it contains lots of information that's specifically intended for cloud administrators.

For us machine learning practicioners Microsoft has provided a much better web interface that's specially designed for managing machine learning resources. This interface is called the Azure Machine Learning Studio. 

You can access the studio interface by clicking the blue 'Launch studio' button:

![Setup step 5](../img/image3.png)
{ .img-fluid .mb-4 }

This will launch the Azure Machine Learning Studio. 

You're now looking at the Azure Machine Learning Studio interface. We will be spending most of our time here while we work through the assignments in this course. 

![Azure Machine Learning Studio](../img/image4.png)
{ .img-fluid .mb-4 }

Next, let's set up a compute instance for training machine learning models. 

{{< /encrypt >}}
