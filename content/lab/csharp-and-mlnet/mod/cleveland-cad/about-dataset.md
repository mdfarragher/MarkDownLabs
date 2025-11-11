---
title: "The Cleveland CAD Dataset"
type: "lesson"
layout: "default"
sortkey: 10
---
Coronary artery disease (CAD) involves the reduction of blood flow to the heart muscle due to build-up of plaque in the arteries of the heart. It is the most common form of cardiovascular disease, and the only reliable way to detect it right now is through a costly and invasive procedure called coronary angiography. This procedure represents the gold standard for detecting CAD, but unfortunately comes with increased risk to CAD patients.

So it would be really nice if we had some kind of reliable and non-invasive alternative to replace the current gold standard.

Other less invasive diagnostic tools do exist. Doctors have proposed using electrocardiograms, thallium scintigraphy and fluoroscopy of coronary calcification. However the diagnostic accuracy of these tests only ranges between 35%-75%.

We can try to build a machine learning model that trains on the results of these non-invasive tests and combines them with other patient attributes to generate a CAD diagnosis. If the model predictions are accurate enough, we can use the model to replace the current invasive procedure and save patient lives.

To train our model, we're going to use the the Cleveland CAD dataset from the University of California UCI. The data contains real-life diagnostic information of 303 anonymized patients and was compiled by Robert Detrano, M.D., Ph.D of the Cleveland Clinic Foundation back in 1988.

![Cleveland CAD Dataset](../img/data.jpg)
{ .img-fluid .pb-4 }

The dataset contains 13 features which include the results of the aforementioned non-invasive diagnostic tests along with other relevant patient information. The label represents the result of the invasive coronary angiogram and indicates the presence or absence of CAD in the patient. A label value of 0 indicates absence of CAD and label values 1-4 indicate the presence of CAD.

Do you think you can develop a medical-grade diagnostic tool for heart disease?

Let's find out!