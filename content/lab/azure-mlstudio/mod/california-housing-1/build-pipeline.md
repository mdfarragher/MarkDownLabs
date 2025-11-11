---
title: "Build The Pipeline"
type: "lesson"
layout: "default"
sortkey: 70
---

In this assignment you are going to build an Azure Machine Learning pipeline that loads the California Housing dataset and transforms any columns that require extra processing. Our objective is to get the data ready for machine learning training. 

Your first task is to start the Azure Machine Learning designer and use it to create the pipeline.

{{< encrypt >}}

Go to your Azure Machine Learning workspace and click the **Designer** link in the menu on the left. You'll be taken to the designer overview page:

![Setup pipeline step 1](../img/pipeline1.png)
{ .img-fluid .mb-4 }

Click the big **+** button in the top left of the page to create a new pipeline. 

The graphical pipeline designer will open. You'll see a red prompt to specify the compute target. This is the virtual machine cluster that the pipeline will run on. 

Click the **Please select the default compute target** banner and specify the virtual machine cluster you set up earlier. 

Then change the default name and description of the pipeline:

- Name: `california-housing-pipeline`
- Description: `Train a model on the California housing dataset`.

![Setup pipeline step 2](../img/pipeline2.png)
{ .img-fluid .mb-4 }

Now let's start building the pipeline. We want to start by loading the California Housing dataset.

Click on the **Datasets** link in the menu on the left, and drag the **california-housing-dataset** component onto the pipeline canvas like this:

![Setup pipeline step 3](../img/pipeline3.png)
{ .img-fluid .mb-4 }

An information panel will open on the right of the page with details about the dataset component. 

You can quickly inspect the data by clicking on the Output tab and then clicking the graph icon:

![Setup pipeline step 4](../img/pipeline4.png)
{ .img-fluid .mb-4 }

This will open the dataset explorer and display interesting statistics about the data columns, just like when we generated the dataset profile. 

Click on the **total_rooms** column. Notice the max value (37937) and the histogram with the long tail of outliers. Clearly there's something strange going on with this column. Some of the housing blocks have almost 38,000 rooms! 

![Setup pipeline step 5](../img/pipeline5.png)
{ .img-fluid .mb-4 }

These outliers are probably hotels. Unfortunately they are going to mess up our training if we want to use this data to predict residential housing block prices. 

The population column has the same issue. The housing block with the largest population has 35,682 people living in it and the histogram also shows a massive long tail.  

We also have the **median_house_value** column which is on a numeric range from 15,000 to 500,000. This is significantly larger than any of the other columns, and we run the risk of having our machine learning algorithm prioritising the median house value over all other data columns.

To fix these issues, we're going to do the following:

* Divide the total_rooms column by the population column to generate a new data column called **rooms_per_person**. This is the average number of rooms available to each person in a housing block.
* We will keep all records with a rooms_per_person value of 4 or less, and discard all records with more than four rooms per person.
* We will divide the **median_house_value** column by 1,000 to bring its numeric range in line with the other columns.

We can achieve all of these transformations with a single component. 

Click on the Data Transformation group and drag the **Apply SQL Transformation** component onto the pipeline canvas:

![Setup pipeline step 6](../img/pipeline6.png)
{ .img-fluid .mb-4 }

Connect the dataset and the SQL transformation components together. Drag the bottom circle of the **california-housing-dataset** component on to the top-left circle of the **Apply SQL Transformation** component, like this:

![Setup pipeline step 7](../img/pipeline7.png)
{ .img-fluid .mb-4 }

Now enter the following SQL query in the information panel on the right: 

```sql
select 
    longitude,
    latitude,
    housing_median_age,
    total_rooms/population as rooms_per_person,
    total_bedrooms,
    households,
    median_income,
    median_house_value/1000.0 as median_house_value
from t1
where
    total_rooms/population <= 4
```

This SQL statement sets up a new column called **rooms_per_person**, and it divides the median house value by 1,000. The WHERE statement keeps only housing blocks with 4 or less rooms per person and discards everything else.

Now we're ready to test this pipeline and check if everything works. 

Click on the blue **Submit** button in the top right. This will start a new experiment and run the pipeline. 

Provide the following information:

* Experiment: `Create new`
* New experiment name: `california-housing-transformation`
* Run description: `Test the SQL data transformation`

And click the blue **Submit** button to start the run.

![Setup pipeline step 8](../img/pipeline8.png)
{ .img-fluid .mb-4 }

While the pipeline is running, you'll see the run status in the top right of the pipeline designer page. 

Wait for the status to be completed. This can take a couple of minutes. 

When the run has completed, you'll see the status 'Run Finished' in the top right of the page. The **Apply SQL Transformation** component will also have a vertical green bar on the left of the box to indicate that it has run successfully:

![Setup pipeline step 9](../img/pipeline9.png)
{ .img-fluid .mb-4 }

You can now click the **Apply SQL Transformation** component again. You'll notice in the information panel on the right that there is an **Output + Logs** tab. 

Click the tab and then click the visualize icon in the Result_dataset box. 

![Setup pipeline step 10A](../img/pipeline10a.png)
{ .img-fluid .mb-4 }

You'll see the same data explorer pop up that we saw previously when testing the dataset component. 

But now notice the columns. Do you see the new **rooms_per_person** column? Note the nicely balanced distribution in the histogram, this is proof that our scrubbing tactic has worked.

Also check out the **median_house_value** column and notice that all values have been divided by 1,000. 

![Setup pipeline step 10B](../img/pipeline10b.png)
{ .img-fluid .mb-4 }

Congratulations! You have successfully prepared the California Housing dataset for machine learning training. 

In the next lab module, we will expand on this pipeline and add components that train a machine learning model to predict house prices. 

{{< /encrypt >}}
