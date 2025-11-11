---
title: "Create The Azure Dataset"
type: "lesson"
layout: "default"
sortkey: 50
---

Our final step is to set up the California Housing dataset in our Azure Machine Learning Workspace. 

{{< encrypt >}}

Go to the vertical menu on the left hand side of the page and click the **Datasets** menu option. Then click on the **+Create Dataset** button, and then select `From Datastore` from the dropdown menu.

![Setup dataset step 1](../img/dataset1.png)
{.img-fluid .mb-4}

In the next page, provide the following information:

* The name of the dataset to create. Fill in `california-housing-dataset` here.
* The type of the dataset. Since we're working with a comma-separated datafile, the type should be set to `Tabular`.
* A description of the dataset. You can leave this field blank.

Note that the dataset also has a version number which is set to `1`. If we decide to load more data later on, or change the existing data, the version number will automatically increment. 

![Setup dataset step 2](../img/dataset2.png)
{.img-fluid .mb-4}

Click the blue **Next** button at the bottom of the page to continue to the next step. 

Now you need to select the datastore which contains the data to load. Select the datastore you created previously: `california_housing_data`.

You also need to provide a path specification that determines which files to load into this dataset. You can specify subfolders and wildcard characters here.  

Note that there's also a special "/**" wildcard that will load all files in all subfolders into the dataset.  

Our datastore only contains one csv file in the root folder, so you can just enter a single `*` wildcard here.

![Setup dataset step 3](../img/dataset3.png)
{.img-fluid .mb-4}

Leave everything else at their default values and click the blue **Next** button to continue. 

Azure Machine Learning will now scan the files in the datastore and attempt to determine the file format. You'll see the following information appear:

* File format: Delimited. This is correct.
* Delimiter: Comma. This is correct.
* Encoding: UTF-8. This is correct.
* Column headers: None. This is not correct, our file has the column headers in the first row. So change this field to: `All files have same headers`.
* Skip rows: None. This is correct.

You can see a preview of the first few lines of the file at the bottom of the page. Note how all the columns appear with their data and headers. This indicates that the file is being parsed correctly.  

![Setup dataset step 4](../img/dataset3.png)
{.img-fluid .mb-4}

Click the blue **Next** button to continue. 

You're now seeing a list of data types for each column in the file. Azure Machine Learning has determined that all columns hold Decimal data. This is correct, so we don't need to change anything.  

Also note the Include switches for each column. By toggling these switches, we can decide which columns get included in the dataset. 

Note that there is an initial column called **Path** (which will not be included). This column contains the full path of the file that contains the record.

As we're working with only a single datafile here, this is not very useful right now. But you can use this column in scenarios where you are loading data from many different csv files and you need to keep track from which source file each record originated.

![Setup dataset step 5](../img/dataset5.png)
{.img-fluid .mb-4}

Click the blue **Next** button at the bottom of the page.

The dataset is now fully configured and Azure Machine Learning shows a summary of all settings. You can double-check that you entered everything correctly and then click the blue **Create** button to create the dataset.

![Setup dataset step 6](../img/dataset6.png)
{.img-fluid .mb-4}

The dataset will now be created. After a few seconds the overview page appears, with a green notification that the dataset has been created successfully. The California housing data should now also appear in the list of datasets.

Your overview page will look like this:

![Setup dataset step 7](../img/dataset7.png)
{.img-fluid .mb-4}

Congratulations! You have successfully set up the California Housing dataset, and we are now ready to start training machine learning models with this data.

We will use this dataset in later assignments.

{{< /encrypt >}}
