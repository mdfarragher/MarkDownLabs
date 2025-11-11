---
title: "Create The Azure Datastore"
type: "lesson"
layout: "default"
sortkey: 40
---

Now we're now going to make the new storage account available in the Azure Machine Learning workspace as a datastore.

{{< encrypt >}}

Go back to the Azure Machine Learning Studio and click on the **Datastores** link in the vertical menu on the left hand side of the page. You'll see the datastores overview page which will look like this:

![Setup datastore step 1](../img/datastore1.png)
{.img-fluid .mb-4}

Click on the **+New** Datastore button to create a new datastore. Provide the following information:

* The datastore name. Set this to `california_housing_data`.

* The datastore type. Select Azure Blob Storage.
* The account selection method. Select `From Azure Subscription`.
* The subscription ID. Select your subscription from the list.
* The storage account. Select the account that you just created in the previous step. 
* The blob container. Select the container that you just created in the previous step.
* Save credentials... Make sure this field is set to `Yes`.
* Authentication type. Set this to `Account Key`.
* Account key. Paste the key1 value of the storage account here that you stored in notepad in the previous step.

Then click the blue **Create** button to create the datastore.

![Setup datastore step 2](../img/datastore2.png)
{.img-fluid .mb-4}

You'll see a message that the datastore was successfully created, and it will appear in the list of stores:

![Setup datastore step 3](../img/datastore3.png)
{.img-fluid .mb-4}

{{< /encrypt >}}
