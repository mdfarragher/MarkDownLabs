---
title: "Create The Azure Storage Account"
type: "lesson"
layout: "default"
sortkey: 30
---

In this assignment you're going to load a dataset with the prices of houses in California into Azure.  

Your first task is to set up an Azure Storage Account to hold all the data we're going to be using in this course. 

{{< encrypt >}}

Go to your Azure portal page at https://portal.azure.com/. Click on the search field in the top blue bar and search for `storage accounts`. Click on the Storage Accounts link in the search dropdown.

You'll be taken to the Azure Storage Accounts page. Click on the **+Create** button in the top left of the page:

![Setup storage account step 1](../img/storage1.png)
{.img-fluid .mb-4}

We're going to create a simple storage account to hold all our machine learning data. Provide the following information:

* The Azure subscription in which you will place the storage account. Select your subscription from the list.
* The resource group in which to place the storage account. Select the same resource group that is currently holding your Azure Machine Learning workspace.
* The storage account name. Type a nice name, for example `mldatastorage`. Note that you cannot use dashes, underscores or spaces in the name.
* The location in which to place the storage account. Use the same location as where your Azure Machine Learning workspace is located.
* The performance level. Set this to `Standard` to save money.
* Redundancy. Set this to `Locally-Redundant Storage (LRS)`.

![Setup storage account step 2](../img/storage2.png)
{.img-fluid .mb-4}

Click the blue **Review+Create** button to confirm your choices and then click the blue **Create** button to create the storage account. This will take up to a minute. 

When the deployment confirmation appears, click on the **Go To Resource** button to navigate to the overview page of your new storage account. 

![Setup storage account step 3](../img/storage3.png)
{.img-fluid .mb-4}

The page should look like this:

![Setup storage account step 4](../img/storage4.png)
{.img-fluid .mb-4}

Now click on the **Containers** menu option in the vertical menu on the left hand side of the page (you may have to scroll down a little). Click the link and you'll be taken to the containers overview page.

![Setup storage account step 5](../img/storage5.png)
{.img-fluid .mb-4}

Click the **+Container** button and provide the following information:

* The name of the container to create. Fill in `california-housing-data` here.
* The access level of the container. Set this to `Private`.

Click the blue **Create** button to confirm your choices and create the new container.

![Setup storage account step 6](../img/storage6.png)
{.img-fluid .mb-4}

When the container appears in the list, click on it. Then click the **Upload** button to upload the California Housing datafile into the container. In the panel on the right hand side of the page, click the small blue browse button to select the datafile from your local computer. Then click the blue **Upload** button to start the upload. 

![Setup storage account step 7](../img/storage7.png)
{.img-fluid .mb-4}

You now have a new storage account with a data container that holds the California Housing data. We are almost ready to bring this data into the Azure Machine Learning Workspace, there's one more thing to do.

## Write down the storage account key

In the breadcrumb at the top of the page, click on the last entry in the breadcrumb. This is the link with the name of your storage account. 

Clicking the link will take you back to the overview page of your storage account:

![Get storage key step 1](../img/storage4.png)
{.img-fluid .mb-4}

In the vertical menu on the left hand side of the page, locate the menu entry named **Access keys** and click on it.

![Get storage key step 2](../img/key2.png)
{.img-fluid .mb-4}

Now click on the blue **Show keys** button to reveal the two access keys: key1 and key2. Copy the value of key1 and store it somewhere safe. For example, you can open an empty notepad window and paste the key in there.

We are going to need this key in a couple of minutes. 

{{< /encrypt >}}


