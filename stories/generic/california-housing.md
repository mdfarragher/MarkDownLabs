 # The California Housing Dataset

 In machine learning circles, the **California Housing** dataset is a bit of a classic. It's the dataset used in the second chapter of Aurélien Géron's excellent machine learning book [Hands-On Machine learning with Scikit-Learn and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646).
 
 The dataset serves as an excellent introduction to building machine learning apps because it requires rudimentary data cleaning, has an easily understandable list of variables and has the perfect size for fast training and experimentation. it was compiled by Pace, R. Kelley and Ronald Barry for their 1997 paper titled [Sparse Spatial Autoregressions](https://www.sciencedirect.com/science/article/abs/pii/S016771529600140X). They built it using the 1990 California census data.
 
 ![The California Housing Dataset](../img/data.jpg)
 { .img-fluid .pb-4 }
 
 The dataset contains one record per census block group, with a census block group being the smallest geographical unit for which the U.S. Census Bureau publishes sample data. A census block group typically has a population of around 600 to 3,000 people.
 
 In this lab, you're going to use the California Housing dataset to build a model that can predict the price of any house in the state of California.
 
 # Get The Data
 
 Let's start by downloading the California Housing dataset. 
 
 {{< encrypt >}}
 
 You can grab the file from here:
 [California 1990 housing census](https://csvbase.com/mdfarragher/California-Housing).
 
 Download the file and save it as **California-Housing.csv**.
 
 The dataset is a CSV file with 17,000 records that looks like this:
 
 ![The California Housing Dataset](../img/data.jpg)
 { .img-fluid .mb-4 }
 
 The file contains information on 17,000 housing blocks all over the state of California. Here's a description of each column:
 
 -    Column 1: The unique row identifier (added by CsvBase)
 -    Column 2: The longitude of the housing block
 -    Column 3: The latitude of the housing block
 -    Column 4: The median age of all the houses in the block
 -    Column 5: The total number of rooms in all houses in the block
 -    Column 6: The total number of bedrooms in all houses in the block
 -    Column 7: The total number of people living in all houses in the block
 -    Column 8: The total number of households in all houses in the block
 -    Column 9: The median income of all people living in all houses in the block
 -    Column 10: The median house value for all houses in the block
 
 This dataset cannot be used directly for machine learning. It must be cleaned, scaled, and preprocessed—this is what we’ll focus on in the next steps.
 
 #### Set Up Your Project
 
 Now open your terminal (called Windows Terminal on Windows, or Console on Mac/Linux). Navigate to the folder where you want to create the project (e.g., **~/Documents**), and run:
 
 [comment]: <> (AI-TODO: provide command-line installation for specified platform, and add a 'cd' command to change to the created project folder)
 ```bash
 # installation commands ...
 ```

 [comment]: <> (AI-TODO: describe files created by installation)
 This creates a new application with:
 
 - **File** – description of file
 - ...
 
 Then move the California-Housing.csv file into this folder.
 
 [comment]: <> (AI-TODO: describe how to install support packages)
 Now run the following command to install additional packages:
 
 ```bash
# installation commands ...
 ```

 [comment]: <> (AI-TODO: describe each installed package)

- **Package** is a ...
- **Package 2** is a ...
 
 Next, we're going to analyze the dataset and come up with a feature engineering plan.
 
 {{< /encrypt >}}
 
 # Analyze The Data
 
 We’ll begin by analyzing the California Housing dataset and come up with a plan for feature engineering. You won’t write any code yet, our goal is to first map out all required data transformation steps to make later machine learning training possible.
 
 {{< encrypt >}}
 
 #### Manually Explore the Data
 
 Let’s start by exploring the dataset manually.
 
 Open **California-Housing.csv** in Visual Studio Code, and start looking for patterns, issues, and feature characteristics.
 
 What to look out for:
 
 -    Are there any missing values, zeros, or inconsistent rows?
 -    Are the values in each column within a reasonable range?
 -    Can you spot any extremely large or very small values?
 -    What’s the distribution of values in columns like median_income, total_rooms, or households?
 -    Are longitude and latitude useful as-is, or will they need transformation?
 
 Write down 3 insights from your analysis.
 {.homework}
 
 #### Ask Copilot To Analyze The Dataset
 
 You can also ask Copilot to analyze the CSV data and determine feature engineering steps. You should never blindly trust AI advice, but it can be insightful to run an AI scan after you've done your own analysis of the data, and compare Copilot's feedback to your own conclusions. 
 
 Make sure the CSV file is still open in Visual Studio Code. Then expand the Copilot panel on the right-hand side of the screen, and enter the following prompt:
 
 "You are a machine learning expert. Analyze this CSV file for use in a regression model that predicts median_house_value. What problems might the dataset have? What preprocessing steps would you suggest?"
 {.prompt}
 
 You can either paste in the column names and 5–10 sample rows, or upload the CSV file directly (if your agent supports file uploads).
 
 ![Analyze a dataset with an AI agent](../img/analyze.jpg)
 {.img-fluid .mb-4}
 
 #### What Might The Agent Suggest?
 
 The agent may recommend steps like:
 
 -    Normalizing income or house value columns
 -    Handling extreme outliers
 -    One-hot encoding categorical features (like location bins)
 -    Engineering new features (e.g., rooms_per_person)
 -    Scaling values for better model convergence
 
 Write down 3 insights from the agent’s analysis.
 {.homework}
 
 #### Compare and Reflect
 
 Now let's compare your findings with Copilot's analysis.
 
 -    Did Copilot suggest anything surprising?
 -    Did you find anything it missed?
 -    Are you confident in which preprocessing steps are necessary?
 
 Try this follow-up prompt based on your observations:
 
 "I inspected the data and found [ .... ]. What kind of preprocessing steps should I use to process this data? And are there any challenges I should take into account?"
 {.prompt}
 
 This back-and-forth helps you learn how to collaborate with AI agents as intelligent assistants, not just code generators.
 
 #### Key Takeaway
 
 Agents are powerful, but they’re not magic. Use them to speed up analysis, but always verify their suggestions through manual inspection and common sense.
 
 {{< /encrypt >}}
 
 # Plot A Histogram Of Total Rooms
 
 Before you build a machine learning model, it’s important to understand your data visually. Just looking at the numbers, like you did in the previous lesson, may not be enough. A good chart can clearly reveal patterns in the dataset.
 
 In this section, you’ll going to build a visualization to detect any outliers in the **total_rooms** feature.
 
 Let's get started.
 
 {{< encrypt >}}
 
 #### Install ScottPlot
 
 ScottPlot is a very nice plotting and visualization library for C# and NET that can go toe-to-toe with Python libraries like matplotlib and seaborn. We will use it in these labs whenever we want to visualize a dataset.
 
 First, let's install the ScottPlot NuGet package. In your terminal (inside the CaliforniaHousing folder), install ScottPlot like this:
 
 [comment]: <> (AI-TODO: describe how to install plotting library) 
 ```bash
 # installation commands ...
 ```
 
 Then open Visual Studio Code in the current folder, like this:
 
 ```bash
 code .
 ```
 
 Open the Program.cs file and remove all existing content, because we don't want the agent to get confused. Replace the content with this:

 [comment]: <> (AI-TODO: describe default import statements) 
 ```csharp
 // default import commands
 ```
 
 #### Create a Histogram of total_rooms
 
 Now let's ask Copilot to write the code for us.
 
 At the bottom of the Copilot panel in Visual Studio Code, make sure the AI mode is set to 'Agent'. Then select your favorite model. I like to use GPT 4.1 or Claude 3.7 for coding work.
 
 ![Enable Agent Mode](../img/agent-mode.jpg)
 { .img-fluid .mb-4 .border }
 
 Now enter the following prompt:
 
 [comment]: <> (AI-TODO: modify prompt as needed) 
 "Write C# code using ScottPlot to generate a histogram of the total_rooms column from the CSV file."
 { .prompt }
 
 And let Copilot write the code for you.

 [comment]: <> (AI-TODO: describe ideal generated data loading code) 
 You should see the following data loading code in your project:
 ```csharp
 // ideal generated data loading code ...
 ```
 
 [comment]: <> (AI-TODO: explain generated data loading code) 
 This code uses ...
 
 [comment]: <> (AI-TODO: add tip for students) 
 We want to ...
 { .tip }
  
 [comment]: <> (AI-TODO: describe ideal data plotting code)  
 The code to plot the histogram should look like this:
 ```csharp
 // ideal generated data plotting code ...
 ```
 
 [comment]: <> (AI-TODO: explain generated data plotting code) 
 Note that the code ...
 
 Now let's look at the histogram. It should look like this:
 
 ![Histogram Of TotalRooms](../img/totalrooms-histogram.png)
 { .img-fluid .mb-4 }
 
 Think about the following:
 
 -    Do you notice the long tail (outliers)?
 -    How should you deal with this?
 -    Could other columns in the dataset have the same issue?
 
 Write down which data transformation steps you are going to apply to deal with the outliers in the total_rooms column.
 { .homework }
 
 #### Create a Histogram of Every Feature
 
 Now let's modify the code to generate histograms for all the columns in the dataset. Enter the following prompt:
 
 "Modify the code so that it creates histograms for every column in the dataset."
 { .prompt }

 [comment]: <> (AI-TODO: describe ideal generated histogram code) 
 You should see the following code in your project:
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain generated histogram code) 
 This code uses ...
 
 Your histograms should look like this:
 
 ![Histogram Of All Columns](../img/all-histograms.png)
 { .img-fluid .mb-4 }
 
 You can see that the **total_rooms**, **total_bedrooms**, **population** and **household** columns have outliers. These are apartment blocks with a very large number of occupants and rooms, and we'll have to deal with them.
 
 #### Summary
 
 Visualization is one of the most important sanity checks in machine learning.
 It helps you spot issues, guide preprocessing, and understand how features behave—even before you build a model.
 
 By using ScottPlot and agents together, you’re learning how to both automate and supervise the exploration process.
 
 {{< /encrypt >}}
 
 # Plot Median House Value By Median Income
 
 In machine learning training, we're looking for features with a balanced distributions of values. Basically, you want the feature histogram to look like a symmetric hump, with flanks on the left and right tapering to zero. 
 
 But do you remember the histogram of the **median_house_value** column from the previous lab lesson? It looks like this:
 
 {{< encrypt >}}
 
 ![Histogram of Median House Value](../img/medianhousevalue-histogram.png)
 { .img-fluid .mb-4 }
 
 So why does this histogram have that weird peak on the right?
 
 The peak represents a statistically significant number of housing blocks in the dataset with a house value around half a million dollars. This is an example of **bias**, and it will make it harder for a machine learning model to generate good predictions.
 
 Let's explore this a little further by creating a scatterplot of median house value by median income.
 
 #### Create a Scatterplot
 
 Before we start, let's remove all the old code we don't need anymore. The data loading code can stay but we'll delete all the plotting code because we don't want the agent to get confused.
 
 Always remove dead code, incorrect code, or workarounds you don't agree with from the code base before giving your agent the next assignment. This keeps your code clean and prevents the agent from going off the rails.
 { .tip }
 
 Delete any code you don't need anymore, and don't worry about compile errors. The agent can fix those in the next pass.
 
 Another trick you can use is to add the following comment in your Program.cs file at the point where you want the agent to add new code:
 
 ```csharp
 // the rest of the code goes here
 ```
 
 This will guide the agent to the correct location in your code base where you want to add new code.
 
 Now enter the following prompt:
 
 "Write C# code using ScottPlot to generate a scatterplot of the median house value column versus the median income column."
 { .prompt }

  [comment]: <> (AI-TODO: describe ideal generated histogram code) 
 You should see the following code in your project:
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain generated histogram code) 
 This code uses ...
 
 Inspect your new code. Make sure the agent uses `Plot.Add.ScatterPoints` instead of `Plot.Add.Scatter`, because the latter draws lines between the data points, and we don't want that here.
 
 For reference, [here is the ScottPlot documentation on scatterplots](https://www.scottplot.net/cookbook/5.0/Scatter/).
 
 You should get a plot that looks like this:
 
 ![Scatterplot of Median House Value by Median Income](../img/income-vs-housevalue.png)
 { .img-fluid .mb-4 }
 
 As the median income level increases, the median house value also increases. There's still a big spread in the house values, but a vague 'cigar' shape is visible which suggests a roughly linear relationship between these two variables.
 
 But look at the horizontal line at 500,000. What's that all about?
 
 This is what **clipping** looks like. The creator of this dataset has clipped all housing blocks with a median house value above $500,000 back down to $500,000. We see this appear in the graph as a horizontal line that disrupts the linear 'cigar' shape.
 
 We will have to deal with this in the upcoming lessons, because we don't want a machine learning model to learn that house prices can somehow never be higher than half a million dollars.
 
 #### Summary
 
 Did I mention that visualization is one of the most important sanity checks in machine learning? It not only helped us identify outliers in the total_rooms column, but it also clearly shows clipping in the median_house_value column.
 
 In upcoming lessons, I'll show you how to deal with outliers and clipping.
 
 {{< /encrypt >}}
 
 # Plot The Pearson Correlation Matrix
 
 When you train a machine learning model on a dataset, you always face the challenge of deciding which columns to include in the training.
 
 You only want to include columns that are uncorrelated with each other. This means that the columns behave independently from one another, and do not increase or decrease together.
 
 In the California Housing dataset, we have two sets of columns that at first glance appear to be highly correlated: 
 
 - **total_rooms** and **total_bedrooms**
 - **population** and **households**
 
 Think about it: a housing block with many rooms will probably also have a lot of bedrooms. And a block with a large population will also host many households.
 
 If we find strongly correlated pairs of columns, we can consider training a machine learning model on a single column. Including the other correlated columns is not useful, because they are not independent variables and we would be wasting valuable algorithmic memory space.
 
 So let's write some code (= have Copilot write it for us) to calculate the correlation matrix.
 
 #### Set Up a Code Structure
 
 In Visual Studio Code, remove all unwanted code from the Program.cs file, and then create the following code structure right after the data loading code:
 
 ```csharp
 // get column names
 var columnNames = (from p in typeof(HousingData).GetProperties()
                    where p.Name != "RowID"
                    select p.Name).ToArray();
 
 // Calculate the correlation matrix
 var matrix = CalculateCorrelationMatrix<HousingData>(housingData, columnNames);
 
 // print correlaton matrix
 PrintCorrelationMatrix(matrix, columnNames);
 
 // plot correlation matrix
 PlotCorrelationMatrix(matrix, columnNames);
 ```
 
 This creates a nice scaffold for the agent to work with. We set up an array of column names (using reflection), and then call `CalculateCorrelationMatrix` to calculate the Pearson correlation matrix. The method is generic so we can reuse it in other projects.
 
 You may be wondering why we calculate the `columnNames` array at all? It's to ensure that `PrintCorrelationMatrix` and `PlotCorrelationMatrix` use the exact same column list and present the columns in the exact same order.
 
  A code scaffold is a great trick to run your agent incrementally, asking it to implement each method in turn. It also forces the agent to use the same data every time, ensuring that the final code works as expected.
  { .tip }
 
 Now we can ask the agent to implement the `CalculateCorrelationMatrix`, `PrintCorrelationMatrix` and `PlotCorrelationMatrix` methods for us.
 
 #### Calculate the Correlation Matrix
 
 Enter the following prompt in Copilot:
 
 "Implement the CalculateCorrelationMatrix method with C# that calculates the Pearson correlation matrix for all columns in the dataset. Use MathNet.Numerics to calculate the matrix."
 { .prompt }
 
 Note how we're guiding the agent by explicitly mentioning **MathNet.Numerics**? We do that, because the Numerics library is the easiest way to calculate a correlation matrix. We can use other libraries (like Deedle or NumSharp), but with Numerics we only need a single call to `Correlation.PearsonMatrix` to calculate the matrix!
 
 If you have a preference for a specific library, mention this in your prompt. This is much better than having the agent pick a library at random and possibly generate convoluted code to make everything work. 
 { .tip }
 
 [comment]: <> (AI-TODO: describe ideal generated correlation matrix code) 
 You should see the following code in your project:
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain generated correlation matrix code) 
 This code uses ...

 [comment]: <> (AI-TODO: provide a tip for students) 
 Don't hesitate to ... 
 { .tip }
 
 #### Print the Correlation Matrix
 
 Now ask the agent to implement the next method:
 
 "Implement the PrintCorrelationMatrix method to print a nice matrix on the console using unicode lines. Use the BetterConsoleTables package."
 { .prompt }

 [comment]: <> (AI-TODO: describe ideal generated printing code) 
 You should see the following code in your project:
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain generated printing code) 
 This code uses ...

 You should get output that looks like this:
 
 ![Correlation Matrix](../img/correlation-console.png)
 { .img-fluid .mb-4 }
 
 You can clearly see that the **TotalRooms**, **TotalBedrooms**, **Population** and **Household** columns are strongly correlated. So we could consider condensing them into a single feature for machine learning training.
 
 It's also interesting to look at the final column in the matrix. Notice how only **MedianIncome** is correlated to the median house value? This means that the median income level at the location of the apartment block most strongly affects the median house value in that block. And that makes perfect sense when you think about it. House prices are indeed strongly correlated to neighborhood income level.
 
 #### Plot the Correlation Heatmap
 
 Now let's see if Copilot can generate a heatmap for us with ScottPlot:
 
 "Implement the PlotCorrelationMatrix method to plot a heatmap of the correlation matrix, using ScottPlot."
 { .prompt }
 
 [comment]: <> (AI-TODO: describe ideal generated heatmap code) 
 You should see the following code in your project:
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain generated heatmap code) 
 This code uses ...
 
 [comment]: <> (AI-TODO: provide a tip for students) 
  Always ...
  { .tip }
 
 Here is the final heatmap in all its glory:
 
 ![Correlation Heatmap](../img/correlation-heatmap.png)
 { .img-fluid .mb-4 }
 
 The correlation blocks are clearly visible now. **TotalRooms**, **TotalBedrooms**, **Population** and **Households** are highly correlated (all numbers go up when housing blocks get bigger).
 
 **Longitude** and **Latitude** are negatively correlated. This makes sense if you think about the shape of the state of California: as you travel south through the state, you also move east because the entire state is angled in the south-east direction.
 
 For machine learning, we are most interested in what correlates with **MedianHouseValue**, our training label. The heatmap clearly shows that there's only a single column with a strong correlation: **MedianIncome**.
 
 #### Create a Utility Class
 
 We'll return to the correlation matrix in later lessons, so this is a good moment to preserve your work.
 
 In Visual Studio Code, select the implementation of the `CalculateCorrelationMatrix`, `PrintCorrelationMatrix` and `PlotCorrelationMatrix` methods (plus any associated helper methods).
 
 Then press CTRL+I to launch the in-line AI prompt window, and type the following prompt:
 
 "Move all of this code to a separate utility class called CorrelationUtils."
 { .prompt }
 
 This will produce a new class file called `CorrelationUtils`, with all of the code for creating, printing and plotting the correlation matrix. You can now use this class in other projects.
 
 When you're happy with generated code and you want to keep it, move it aside into separate class files. That keeps your main code file (Program.cs) clean and ready for the next agent experiment.
 { .tip }
 
 And if you want to clean up your code and make it as side-effect-free as possible, you can edit `PlotCorrelationMatrix` and have it return the `Plot` instance. You can then save the grid in the main program class instead. Your main calling code will then look like this:
 
 ```csharp
 // plot correlation heatmap
 var plot = PlotCorrelationMatrix(matrix, columnNames);
 
 // Save the plot to a file
 plot.SavePng("correlation_heatmap.png", 900, 800);
 ```
 
 If you get stuck or want to save some time, feel free to download my completed CorrelationUtils class from Codeberg and use it in your own project:
 
 https://codeberg.org/mdft/ml-mlnet-csharp/src/branch/main/CaliforniaHousing/CorrelationUtils.cs
 
 #### Summary
 
 The correlation matrix is a great tool to figure out which dataset columns are strongly correlated to the label column we're trying to predict. We must include these columns in machine learning training, because they will strongly contribute to good predictions.
 
 We can also find feature columns that are strongly correlated to each other. These columns can be combined into one. For example, we could combine **TotalRooms**, **TotalBedrooms**, **Population** and **Households** into a new composite column named **RoomsPerPerson**, by dividing the total number of rooms by the population.
 
 In the next lesson, we're going to do exactly that.

# Design And Build The Transformation Pipeline
 
 Now let's start designing the ML.NET data transformation pipeline. This is the sequence of feature engineering steps that will transform the dataset into something suitable for a machine learning algorithm to train on.
 
 {{< encrypt >}}
 
 #### Decide Feature Engineering Steps
 
 After completing the previous lessons, you should have a pretty good idea which feature engineering steps are needed to get this dataset ready for machine learning training.
 
 Write down all feature engineering steps you want to perform on the California Housing dataset, in order.
 { .homework }
 
 Here are some steps you could consider:
 
 -   Normalize every feature in the dataset
 -   Remove outliers with very high **TotalRooms** or **Population** values
 -   Remove the 'clipped' houses with a median house value > $499,999
 -   Condense the **TotalRooms**, **TotalBedrooms**, **Population** and **Households** columns into one or two computed columns.
 -   Bin and one-hot encode the **Latitude** and **Longitude**.
 
 Which steps will you choose?
 
 #### Implement The Transformation Pipeline
 
 Now let's ask Copilot to implement our chosen data transformation steps with an ML.NET machine learning pipeline.
 
 First, remove any code you don't need anymore (for example, the calls to the `CalculateCorrelationMatrix`, `PrintCorrelationMatrix` and `PlotCorrelationMatrix` methods).
 
 Then, enter the following prompt in the Copilot panel:
 
 "Implement the following data transformations by building a machine learning pipeline:<br>- [your first transformation step]<br>- [your second transformation step]<br>- ..."
 { .prompt }
 
 You should now have a nice data transformation pipeline that prepares your dataset for machine learning training. Let's take a look at the code.
 
 #### Removing outliers
 
 If you decided to remove outliers, for example by removing any rows that have **Population** > 5000 and **MedianHouseValue** > $499,999, your code should look like this:

  [comment]: <> (AI-TODO: describe ideal code to filter data) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain filter data code) 
 This code uses ...
 
 #### Adding computed columns
 
 If you decided to create a new computed column like **RoomsPerPerson**, you'll notice the following code in your project:

   [comment]: <> (AI-TODO: describe ideal code to add computed columns) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain code to add computed columns) 
 This code uses ...
 
 #### Bin- and one-hot encode latitude and longitude
 
 If you decided to bin- and one-hot encode **Latitude** and **Longitude**, you'll see the following code:

 [comment]: <> (AI-TODO: describe ideal code to bin and one-hot encode latitude and longitude) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain code to bin and one-hot encode latitude and longitude) 
 This code uses ...

  
 ### Normalization
 
 If you decided to normalize any columns in the dataset, it will look like this:

 [comment]: <> (AI-TODO: describe ideal code to normalize columns) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain code to normalize columns) 
 This code uses ...

 
 These code examples are reference implementations of common data transformations in ML.NET. Compare the output of your AI agent with this code, and correct your agent if needed.
 { .tip }
 
 To perform the transformations and get access to the transformed data, you'll need code like this:

  [comment]: <> (AI-TODO: describe ideal code to transform data) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain code to transform data) 
 This code uses ...

 
 #### Test The Code
 
 My Claude 3.7 agent added a bit of extra code after the pipeline to output a sample row from the transformed data. My run looked like this:
 
 ![Pipeline Run Output](../img/pipeline-run.png)
 { .img-fluid .mb-4 }
 
 You can see that I decided to remove outliers by getting rid of all rows with a population larger than 5000. There were 265 housing blocks matching that condition in the dataset. The new computed column **RoomsPerPerson** has a numeric range from 0.0019 to 1.0, this is because I normalized all columns, including this one. And in the sample row, you can clearly see that the latitude and longitude values have been one-hot encoded into 10-element numerical vectors.
 
 Everything seems to be working.
 
 #### Summary
 
 In this lesson, you put on your data scientist hat and decided which data transformation steps to apply to the dataset. The AI agent then generated the corresponding MLNET pipeline code for you.
 
 Coming up with the correct data transformations for any given dataset requires deep domain knowledge and a fair bit of intuition, and this is not something an agent can reliably do for you. Don't fall into the trap of asking the agent to come up with the transformations, because this is your job!
 
 So always make a plan first, based on your analysis of the dataset. Then prompt the agent and ask it to follow your plan.
 
 To wrap this lab up, let's see if we can create a cross product of the latitude and longitude vectors.
 
 {{< /encrypt >}}
 
 # Cross Latitude And Longitude
 
 Now let's perform one final data transformation: we're going to calculate the cross product of the encoded latitude and longitude, to create a new 100-element vector of zeroes and ones. We're layering a 10x10 grid over the state of California and placing a single '1' value in the grid to indicate the location of the housing block.
 
 Let's get started.
 
 {{< encrypt >}}
 
 #### Cross The Latitude and Longitude
 
 Open the Copilot panel in Visual Studio Code and enter the following prompt:
 
 "Add a step to the transformation pipeline to calculate a vector cross product of LatitudeEncoded and LongitudeEncoded, creating a new vector with 100 elements."
 { .prompt }
 
My AI agent implemented the cross product like this:

  [comment]: <> (AI-TODO: describe ideal crossing code) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain crossing code) 
 This code uses ...
 

 You should get something like the following output:
 
 ![Cross Of Latitude And Longitude](../img/cross-console.png)
 { .img-fluid .mb-4 }
 
 You can see from the three sampled test rows that the cross product is a 100-element one-hot encoded vector, and that we have only a single '1' value in each vector.
 
 #### Summary
 
 Vector crossing is a handy trick for dealing with latitude and longitude features. Instead of training a machine learning model on two separate 10-element vectors, we now train on a single 100-element vector.
 
 This gives a machine learning model the freedom to treat each grid cell independently from all others. For example, a model could learn that housing blocks in San Francisco are very expensive, but if you travel a couple of miles east, the price drops rapidly. The model will be able to optimize predictions for these two regions independently.
  
 {{< /encrypt >}}
 
 # Recap
 
Congratulations on finishing the lab. Here's what you have learned.

{{< encrypt >}}

You learned how to analyze the **California Housing dataset**, both automatically by having an AI agent scan the data for you, and manually by inspecting the data by hand. You also learned how to prompt an AI agent to quickly generate **dataset visualizations** with ScottPlot, like feature histograms and correlation heatmaps.

You learned that the California Housing dataset needs a lot of preprocessing before we can use it for machine learning training. The median house values have been **clipped** at $500,000 and there are extreme **outliers** with a very high population and total number of rooms. You also discovered that many columns are strongly **correlated** with each other.

You learned how to prompt an AI agent to generate all the code to set up a **data transformation pipeline**. You learned what reference implementations of common data transformations look like in Microsoft.ML, and are now able to supervise the output of your AI agent for future work.

The dataset contains latitude and longitude columns. You learned how to **bin-, one-hot encode- and cross** them to create a 10x10 grid overlaying the state of California.

{{< /encrypt >}}

# Conclusion

This concludes the section on feature engineering. 

You now have some hands-on experience scrubbing, scaling, transforming, binning, one-hot encoding and crossing feature data. You'll use these skills in later labs to optimize your machine learning predictions.

The exact sequence of data transformation steps has a huge impact on the accuracy of machine learning predictions. This is why feature engineering is such an important step in data science.

As you practice with more and more datasets, you will slowly build an intuition for choosing the right transformation step for each feature column in your data.

But until then, just remember to always normalize your data and one-hot encode anything that looks like a category!

# Train A Regression Model

We're going to continue with the code we wrote in the previous lab. That C# application set up a data transformation pipeline to load the California Housing dataset and clean up the data using several feature engineering techniques.

So all we need to do is append a few command to the end of the pipeline to train and evaluate a regression model on the data.

{{< encrypt >}}

#### Split The Dataset

But first, we need to split the dataset into two partitions: one for training and one for testing. The training partition is typically a randomly shuffled subset of around 80% of all data, with the remaining 20% reserved for testing.

We do this, because sometimes a machine learning model will memorize all the labels in a dataset, instead of learning the subtle patterns hidden in the data itself. When this happens, the model will produce excellent predictions for all the data it has been trained on, but struggle with data it has never seen before.

By keeping 20% of our data hidden from the model, we can check if this unwanted process of memorization (called **overfitting**) is actually happening.

So let's split our data into an 80% partition for training and a 20% partition for testing.

Open the code from the previous lesson in Visual Studio Code. Keep the data transformation pipeline intact, but remove any other code you don't need anymore.

Then open the Copilot panel and type the following prompt:

"Split the data into two partitions: 80% for training and 20% for testing."
{ .prompt }

You should get the following code:

 [comment]: <> (AI-TODO: describe ideal code to split dataset) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain code to split dataset) 
 This code uses ...


#### Add A Machine Learning Algorithm

Now let's add a machine learning algorithm to the pipeline.

"Add code to set up a linear regression algorithm."
{ .prompt }

That should produce the following code:

 [comment]: <> (AI-TODO: describe ideal code to perform linear regression) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain code to perform linear regression) 
 This code uses ...


Be careful when you run this prompt! My AI agent generated code that included all features, including **MedianHouseValue**, **Latitude**, **Longitude**, **TotalRooms**, **TotalBedrooms**, **Population**, **Househoulds** and the encoded latitude and longitude.

This is obviously wrong, as the location cross product replaces all other latitude and longitude columns, and **RoomsPerPerson** replaces all other room- and person-related columns.

Even worse, did you notice the **MedianHouseValue** column in that list? This is the label that we're trying to predict. If we train a model on the label itself, the model can simply ignore all other features and output the label directly. This is like asking the model to make a prediction, and then giving it the actual answer it is supposed to predict. 

Always be vigilant. AI agents can easily make mistakes like this, because they do not understand the meaning of each dataset column. Your job as a data scientist is to make sure that the generated code does not contain any bugs.
{ .tip }

 [comment]: <> (AI-TODO: describe the linear regression algorithm, add Wikipedia link for reference) 
By the way, SDCA is an optimized stochastic variance reduction algorithm that converges very quickly on an optimal solution. If you're interested, you can read more about the algorithm on Wikipedia:

https://en.wikipedia.org/wiki/Stochastic_variance_reduction#SDCA


#### Train A Machine Learning Model

Now let's train a machine learning model using our data transformation pipeline and the regression algorithm:

"Train a machine learning model on the training set using the regression algorithm."
{ .prompt }

That will produce the following code:

 [comment]: <> (AI-TODO: describe ideal code to train model) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain code to train model) 
 This code uses ...


#### Summary

In this lesson, you completed the machine learning pipeline you built in the previous lesson, by adding a machine learning algorithm. Then you split the data into a training and testing set, and trained a model on the training set.

In the next lesson, we'll calculate the prediction evaluation metrics to find out how good the model is at predicting house prices.

{{< /encrypt >}}

# Evaluate The Results

Now let's evaluate the quality of the model by having it generate predictions (called **scoring**) for the remaining 20% of data in the test partition. Then we'll compare those predictions to the actual median house values, and calculate the regression evaluation metrics.

So imagine you are a realtor in California selling houses. What kind of prediction accuracy would you consider acceptable?

{{< encrypt >}}

Determine the minimum mean absolute error or root mean square error values you deem acceptable. This will be the target your model needs to beat.
{ .homework }

#### Calculate Evaluation Metrics

Enter the following prompt:

"Use the trained model to create predictions for the test set, and then calculate evaluation metrics for these predictions and print them."
{ .prompt }

That should create the following code:

 [comment]: <> (AI-TODO: describe ideal code to create predictions and evaluate them) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain code to create predictions and evaluate them) 
 This code uses ...


This code calculates the following metrics:

- **RSquared**: this is the coefficient of determination, a common evaluation metric for regression models. It tells you how well your model explains the variance in the data, or how good the predictions are compared to simply predicting the mean.
- **RootMeanSquaredError**: this is the root mean squared error or RMSE value. It’s the go-to metric in the field of machine learning to evaluate models and rate their accuracy. RMSE represents the length of a vector in n-dimensional space, made up of the error in each individual prediction.
- **MeanSquaredError**: this is the mean squared error, or MSE value. Note that RMSE and MSE are related: RMSE is the square root of MSE.
- **MeanAbsoluteError**: this is the mean absolute prediction error.

Note that both RMSE and MAE are expressed in dollars. They can both be interpreted as a kind of 'average error' value, but the RMSE will respond much more strongly to large prediction errors. Therefore, if RMSE > MAE, it means the model struggles with some predictions and generates relatively large errors. 

You should get the following output:

![Regression Model Evaluation](../img/evaluate.jpg)
{ .img-fluid .mb-4 }

Let's analyze my results:

The R-squared value is **0.6013**. This means the model is able to explain ~60% of the variance in housing prices. This is not bad, it's capturing a majority of the pattern in the data, but still leaves ~40% unexplained. This indicates a moderate-to-strong fit, and for real-world housing data, we could consider this a reasonable result.

The mean absolute error (MAE) is **$42,760**. So on average, the model's predictions are off by about $42k. That's not bad at all, given that the most expensive house in our dataset is $500k.

The mean squared error (MSE) is **~3.65 billion**. Large errors are penalized much more heavily in this metric because of the squaring, so this large number indicates the presence of some high-error predictions.

The root mean squared error (RMSE) is **~$60,439**. This metric is similar to MAE, but gives more weight to large errors. The fact that the RMSE > MAE suggests that the model occasionally makes big errors in its predictions.

So how did your model do?

Compare your model with the target you set earlier. Did it make predictions that beat the target? Are you happy with the predictive quality of your model? Feel free to experiment with other learning algorithms to try and get a better result.
{ .homework }

#### Summary

Evaluation is an essential step in machine learning, because this is where we check if the predictions our model is making are any good. If our model cannot beat the target we set, we'll have to go back to the drawing board and tweak the data transformations or pick a different learning algorithm.

A target of $50k is perfectly reasonable for housing, it means you'll accept a 10% prediction error for houses worth $500k. And the SDCA learning algorithm combined with all the data transformations we discussed was able to beat that target.

Let's continue working on our machine learning app and add a couple more features.

{{< /encrypt >}}

# Save And Load The Model

When you have a machine learning model with good prediction quality, you may want to save the model to a file so that you can easily use it later.

Saving a model will export all of the internal model weights, which represent the knowledge the model has gathered during the training. These weights are just a series of numbers, and saving these numbers to a file safeguards this knowledge and makes it available for later use.

{{< encrypt >}}

When we want to use a model to make predictions, we can simply set up a blank machine learning model, and then load knowledge into it by importing the weights back in to the model. This bypasses the entire training process, which is great because training a large model can sometimes take weeks or months!

Let's enhance our app with some simple code to save the weights of the fully trained model to a file.

#### Save The Model

Open Visual Studio Code and enter this prompt in the Copilot panel:

"Add code to save the fully trained model to a file."
{ .prompt }

Your agent should generate the following code:

 [comment]: <> (AI-TODO: describe ideal code to save model to file) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain code to save model to file) 
 This code uses ...

 
 [comment]: <> (AI-TODO: describe the saved file) 
 The file is saved in the .... format which is ...

There's a special universal file format called ONNX that you can use to transfer knowledge between machine learning models running on different platforms. So let's modify our app to use the ONNX format instead. 

#### Save The Model In ONNX Format

Enter the following prompt:

"Add code to save the fully trained model to a file in the ONNX format."
{ .prompt }

 The generated code should look like this:

 [comment]: <> (AI-TODO: describe ideal code to save model as ONNX file) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain code to save model as ONNX file) 
 This code uses ...


 [comment]: <> (AI-TODO: add a tip for students) 
Don't hesitate to ... 
{ .tip }


#### Loading The Model

Let's add some code to load the model from a file. We can also have the app ask us if we want to train the model or simply load it from a file directly.

Enter the following prompt:

"Add code to load the model from a file. When the app starts, ask the user if they want to train a model and save it to a file, or load the model from a file and use it to generate predictions."
{ .prompt }

This will add a query to your app, and based on your decision, it will either train the model and save it, or load the model and evaluate it.

The code to load a model from a file looks like this:

 [comment]: <> (AI-TODO: describe ideal code to load model from file) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain code to load model from file) 
 This code uses ...

Let's test the code. Here's what I get when I choose to train the model:

![Training And Saving The Model](../img/save-model.jpg)
{ .img-fluid .mb-4 }

And this is what I get when I ask the app to load the model from a file:

![Loading The Model](../img/load-model.jpg)
{ .img-fluid .mb-4 }

#### Summary

Saving the fully trained machine learning model is a great trick if you want to use the model to generate lots of predictions. Instead of having to train your model every time your app starts up, you can simply load all the knowledge into a blank model instantly.

We do this all the time in machine learning, because large complex models often require weeks (or even months) to train.

Do be careful with the ONNX format. It's great that we can transfer knowledge between models running on different platforms, but many data transformations (like custom mappings) are not supported. You would not be able to save and then reload the California Housing model in this format.

{{< /encrypt >}}

# Make A Prediction

To wrap up, let’s use the model to make a prediction.

We're going to invent a fake housing block in San Francisco, in the middle of the Mission district. The block has 2500 rooms, 1000 bedrooms, houses 500 people and 150 households. The apartments are 10 years old on average, and the normalized median income in that neighborhood is 2.0.

For how much could you sell an apartment in that housing block?

We will ask our AI agent to write code that prompts us for all the properties of a single housing block, and then we'll use the machine learning model to predict what the median house value will be for any apartment in the block.

{{< encrypt >}}

#### Make A Prediction

Enter the following prompt:

"Add code to prompt the user for all the properties of a single housing block, and then use the model to generate a prediction of the median house value."
{ .prompt }

The agent will generate the following code:

 [comment]: <> (AI-TODO: describe ideal code to prompt user for data and then make a prediction) 
 ```csharp
 // ideal generated code ...
 ```
 
 [comment]: <> (AI-TODO: explain code to prompt user for data and then make a prediction) 
 This code uses ...


Let's try this for the fake apartment I introduced earlier. Here is the data you need to enter:

- Longitude = 37.760653
- Latitude = -122.418725
- HousingMedianAge = 10
- TotalRooms = 2500
- TotalBedrooms = 1000
- Population = 500
- Households = 150
- MedianIncome = 2.0

And this is the output I get:

![Using The Model To Make A Prediction](../img/prediction.jpg)
{ .img-fluid .mb-4 }

I get a predicted median house value of **$143,265**.

What prediction did you get? Try changing the input data to see how this affects the predicted house value. Do the changes in prediction value make sense to you?
{ .homework }

#### Summary

Making predictions with the fully trained machine learning model is super easy. All you need to do is set up a prediction engine and then feed data into it. The engine will perform the exact same data transformations you used when training the model, and this ensures that the predictions make sense.

Using a machine learning model to generate predictions is called inference, and it requires a lot less compute capacity than training the model did. The inference compute load is often thousands of times smaller than the training load.

So it makes perfect sense to run a model in production, where you can quickly initialize it by importing the weights from a file. If over time the data changes significantly and the prediction quality starts dropping (this is called **data drift**), you can re-train the model on a separate compute platform and copy the new weight file back to production. 

{{< /encrypt >}}

# Improve Your Results

There are many factors that influence the quality of your model predictions, including how you process the dataset, which regression algorithm you pick, and how you configure the training hyperparameters.

Here are a couple of things you could do to improve your model:

{{< encrypt >}}

- Bin the latitude and longitude into more than 10 bins, thus creating a finer grid over the state of California.
- Use a different learning algorithm.
- Use different hyperparameter values for your learning algorithm.
- Eliminate outliers with a large number of rooms.
- Eliminate outliers with very small populations.
- Try to bin other columns, for example replacing **HousingMedianAge** with a three-element one-hot encoding vector with columns "Young", "Median" and "Old".

Experiment with different data processing steps and regression algorithms. Document your best-performing machine learning pipeline for this dataset, and write down the corresponding regression evaluation metrics.
{ .homework }

How close can you make your predictions to the actual house prices? Feel free to try out different approaches. This is how you build valuable machine learning skills!

{{< /encrypt >}}

# Hall Of Fame

Would you like to be famous? You can [submit your best-performing model](mailto:mark@mdfteurope.com) for inclusion in this hall of fame, which lists the best regression evaluation scores for the California Housing dataset. I've added my own results as a baseline, using the transformations I mentioned in the lab. 

Can you beat my score?

| Rank | Name | Platform | Algorithm | Transformations |   MAE   |  RMSE   |
|------|------|----------|-----------|-----------------|---------|---------|
|  1   | Mark | ML.NET   | SDCA      | As mentioned in lab | $42,760 | $60,439 |

I will periodically collect new submissions and merge them into the hall of fame. I'll share the list in my courses and on social media. If you make the list, you'll be famous!

# Recap-2

Congratulations on finishing the lab. Here's what you have learned.

{{< encrypt >}}

 [comment]: <> (AI-TODO: update recap text to refer to correct functions and classes) 
 
You learned to **split the dataset** into separate parts for training and testing to prevent the machine learning model from memorizing the median house values for every single housing block in the dataset. You then used a pipeline to process the California housing data. Extending the pipeline you built in the previous lab, you added the `Concatenate` step to combine all dataset columns into a single feature column to train on.

You used a **training algorithm** (like SDCA) to train a regression model on the transformed data, and called the `Transform` and `Evaluate` methods to calculate the regression metrics and evaluate the quality of your predictions. You saved the fully trained model with the `Save` method, and reloaded it with the `Load` method. You also learned about the **ONNX** file format, which can be used to exchange weights between models running on different software platforms. You learned that custom data transformations cannot be reloaded, unless they are implemented in a separate class tagged with `CustomMappingFactoryAttributeAttribute`.

And finally, you learned how you can call the `CreatePredictionEngine` method to **generate predictions** with the fully-trained regression model.

You completed the lab by experimenting with different data processing steps and regression algorithms to find the best-performing model. 

{{< /encrypt >}}

# Conclusion-2

You now have hands-on experience building a C# app that trains a regression model on a dataset, and then using the fully trained model to generate predictions. This specific dataset, California Housing, required a lot of preprocessing and has features that only weakly correlate with the label. This makes predicting accurate house prices quite challenging. Nevertheless, the best mean absolute error you can achieve is around $28,000.

I hope you also noticed that you cannot simply keep increasing the number of longitude and latitude bins. There's an optimum, and once you go beyond that point, the predictions start degrading in quality again. Knowing where a house is with centimeter-level accuracy is not helpful at all, so we always want to work with a specific level of uncertainty to help our models make better predictions.

This was a fun lab, right?
