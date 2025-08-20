---
title: "Recap"
type: "lesson"
layout: "default"
sortkey: 900
---

Congratulations on finishing the lab. Here's what you have learned.

{{< encrypt >}}

You learned how to analyze the **California Housing dataset**, both automatically by having an AI agent scan the data for you, and manually by inspecting the data by hand. You also learned how to prompt an AI agent to quickly generate **dataset visualizations** with ScottPlot, like feature histograms and correlation heatmaps.

You learned that the California Housing dataset needs a lot of preprocessing before we can use it for machine learning training. The median house values have been **clipped** at $500,000 and there are extreme **outliers** with a very high population and total number of rooms. You also discovered that many columns are strongly **correlated** with each other.

You learned how to prompt an AI agent to generate all the code to set up a **data transformation pipeline**. You learned what reference implementations of common data transformations look like in Microsoft.ML, and are now able to supervise the output of your AI agent for future work.

The dataset contains latitude and longitude columns. You learned how to **bin-, one-hot encode- and cross** them to create a 10x10 grid overlaying the state of California.

{{< /encrypt >}}
