# Clustering Implementation

# Basic Theory

The program implementation is done in python file. The structure of the program consists of initialization phase, where the data are set to make it easier to be processed, by giving a label to the data. The main program will implement the k-means, with 4 parameters : the number of k, maximum iteration, the method of distance counting (Euclidean, Manhattan, or Cosine), and Data Normalization (True or False).

The result of the program will be used to predict the class of the data without the label. This label of prediction will be joined with the label of the data, giving each cluster the labels and the number of occurrences in the cluster. For example, for data like this :

This data will be converted into a table that filled with the number of labels and the number of occurrences (the leftmost cluster is 1, followed by the next):

From this table of clusters and labels, there are values to be counted, there are Precision (P), Recall (R), and F-score (F). The formula of each will be shown below :

# Results

The task in total creates 6 possible graphs, for each method of distance counting (Euclidean, Manhattan, and Cosine) as well as normalization method (True or False). Each graph will be shown below :

Figure 3 and Figure 2 are the answers for question 2 and 3, figure 5 and 4 are the answers for questions 4 and 5, and figure 7 are the answer for question 6. Figure 6 are shown to complete the graphs.

To determine the best setting for the clustering, there must be some important considerations: number of clusters, best method, and the normalization of the data. The number of clusters is the most important, since more cluster means less precision/recall, and less cluster could mean the data cannot be represented.

From Figures 2 until 7, notice that the values of precision, recall, and f-score is met on a single cluster point. This single point will be considered the optimum value because after this point, the precision becomes too low and before this point, the recall becomes too high. After these values met, precision will tend to increase, recall tend to decrease, and f-score follows the trend of both recall and precision. The results of the meeting points for each configuration and normalization of the data will be shown in the table below, with the f-score shown:

As a conclusion, the best configuration is the Manhattan Distance to count the length to the clusters, and the data is not normalized. This configuration is chosen since it shows the highest score of f-score.
