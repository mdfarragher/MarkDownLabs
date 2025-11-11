---
title: "Install The Compute Cluster"
type: "lesson"
layout: "default"
sortkey: 30
---

You now have a compute instance up and running for training your machine learning models. But let's also set up a compute cluster to complete the picture. 

For this lab you will need only a tiny cluster. We will limit the total number of virtual machines to 2 to keep costs down. 

{{< encrypt >}}

Make sure you select the **Compute Cluster** tab at the top of the page, and then click the blue **+New** button. 

![Compute setup step 2](../img/image12.png)
{ .img-fluid .mb-4 }

We're going to set up a new cluster by providing the following information:

* The virtual machine priority. Set this to `Low priority` to save money. 
* The virtual machine type. Just like with the compute instance, select the `CPU` type here.
* The virtual machine size. Click **Select from all options**  and then type `D1` in the search field. The search result list will then show the `Standard_D1` size. Select it.

Note that we're selecting the smallest possible virtual machine size for this training in order to keep daily costs as low as possible. 

If you want, you can also select the `Dedicated` priority, `GPU` virtual machine type and then enter `NC6` in the search field. This will display the `Standard_NC6` size. This virtual machine will give you better performance but at a higher cost.

Now click the blue **Next** button.

![Compute setup step 2](../img/image13.png)
{ .img-fluid .mb-4 }

Now we need to provide the following:

* The name of the new compute cluster. Give it a nice name like `compute-cluster`. 
* Minimum number of nodes. Set this to `0` to ensure that all virtual machines in the cluster shut down when we are not training models.
* Maximum number of nodes. Set this to `2` so that during peak load we have two virtual machines running.
* Idle seconds before shut down. Set this to `120` seconds. If we are not using our cluster for more than two minutes, it will automatically shut down. 

Click the blue **Create** button to set up the cluster. 

![Compute setup step 2](../img/image15.png)
{ .img-fluid .mb-4 }

After a couple of minutes the compute cluster will be running. You should see a green icon and the text 'Succeeded' in the Compute Cluster tab:

![Azure ML Compute overview](../img/image17.png)
{ .img-fluid .mb-4 }

Congratulations! Your Azure Machine Learning Studio is now fully operational. We will use these compute resources in the upcoming course assignments to build, train, and run several cool machine learning models. 
 
 {{< /encrypt >}}
