# ### Data Preprocessing

# Import Libraries
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import Files and Labelling : Animals, Countries, Fruits, Veggies
animals = pd.read_csv("animals", sep=" ", header=None)
animals['class'] = 'a'
countries = pd.read_csv("countries", sep=" ", header=None)
countries['class'] = 'c'
fruits = pd.read_csv("fruits", sep=" ", header=None)
fruits['class'] = 'f'
veggies = pd.read_csv("veggies", sep=" ", header=None)
veggies['class'] = 'v'

# Declare cluster names, methods of finding distance, and l2norm status
clusters = ["a", "c", "f", "v"]
methods = ["euclidean", "cityblock", "cosine"]
isl2norm = [True, False]

# Declaring new data to be processed
new_data = pd.concat([animals, countries, fruits, veggies], ignore_index=True)
data_test = []
data_test = pd.DataFrame(data_test)
data_test['class'] = new_data['class']
del new_data[0], new_data['class']


# ### Main Function

# The main class to implement k-means
class K_Means:
    # The constructor of the function
    # Parameters : k (clusters), tolerance, maximum iterations, method: euclidean, cityblock, cosine, l2norm: True, False     
    def __init__(self, k=2, tol=0, max_iter=300, method="euclidean", l2norm=False):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.l2norm = l2norm

    # The main function to do fit function
    def fit(self,data):
        
        # Will do the normalization, if necessary
        if (self.l2norm == True):
            transformer = Normalizer().fit(data)
            data = transformer.transform(data)
        
        # Initialize the Centroids
        self.ctrds = {}
        
        # Set the default centroids as the first k points of the data
        for i in range(self.k):
            self.ctrds[i] = data[i]
        
        # Set the default running iteration
        for i in range(self.max_iter):
            self.classes = {}
            
            # Set the default classification for k classes
            for i in range(self.k):
                self.classes[i] = []
            
            # Count the distances between the data and the centroids. Methods : Euclidean, Cityblock, Cosine
            for data_rows in data:
                if (self.method == "euclidean"):
                    dst = [distance.euclidean(data_rows, self.ctrds[ctrd]) for ctrd in self.ctrds]
                elif (self.method == "cityblock"):
                    dst = [distance.cityblock(data_rows, self.ctrds[ctrd]) for ctrd in self.ctrds]
                elif (self.method == "cosine"):
                    dst = [distance.cosine(data_rows, self.ctrds[ctrd]) for ctrd in self.ctrds]
                clf = dst.index(min(dst))
                self.classes[clf].append(data_rows)
            
            # Enter the centroids to a dictionary
            prev_ctrds = dict(self.ctrds)
            
            # Determine the average of the distances to determine a new centroid
            for clf in self.classes:
                self.ctrds[clf] = np.average(self.classes[clf],axis=0)

            optimized = True
            
            # Determine the loop if it has passed the minimum tolerance or not
            for c in self.ctrds:
                default_ctrd = prev_ctrds[c]
                current_ctrd = self.ctrds[c]
                if np.sum((current_ctrd - default_ctrd)/default_ctrd * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break
    
    # Predict the data using the model that is made on fit. Methods : euclidean, cityblock, cosine. Returns the classification of the data
    def predict(self,data):
        if (self.method == "euclidean"):
            dst = [distance.euclidean(data, self.ctrds[ctrd]) for ctrd in self.ctrds]
            clf = dst.index(min(dst))
        elif (self.method == "cityblock"):
            dst = [distance.cityblock(data, self.ctrds[ctrd]) for ctrd in self.ctrds]
            clf = dst.index(min(dst))
        elif (self.method == "cosine"):
            dst = [distance.cosine(data, self.ctrds[ctrd]) for ctrd in self.ctrds]
            clf = dst.index(min(dst))
        return clf


# ### Main Program

# Creates an array of items based on total number of clusters. See report on table 1 for example
def array_maker(k):
    array_temp = []
    for i in range(0, k):
        if(i + 1) < 10:
            array_temp.append(str(0) + str(i+1))
        else:
            array_temp.append(str(10))
    return array_temp

# Copy the data to be processed on the main program
data_test_temp = data_test

# Initialize a new array of results, to be used in the program
array_of_results = []

# This loop runs on 3 variables of distance search as declared above : Euclidean, Cityblock, Cosine 
for methods_names in methods:
    # This loop runs on 2 variables of l2norm condition as declared above : True, False
    for l2normstatus in isl2norm :
        # This loop runs on 10 possible values of k, 1 until 10
        for clusters_k in tqdm(range (1, 11)):
            # The model is made and initialized
            model = K_Means(method = methods_names, l2norm = l2normstatus, k = clusters_k)
            model.fit(new_data.values)
            
            # Declare a new array of data to be used on prediction
            array_of_data = new_data.values
            
            # Checking possible normalization of a data
            if (l2normstatus == True):
                transformer = Normalizer().fit(array_of_data)
                array_of_data = transformer.transform(array_of_data)
            
            # Declare a new array to put all the prediction
            data_prediction = []
            
            # This loop does the prediction and put it on table "data_prediction"
            for i in range(len(new_data.values)):
                if ((model.predict(array_of_data[i]) + 1) < 10) :
                    data_prediction.append("0" + str(model.predict(array_of_data[i]) + 1))
                else:
                    data_prediction.append(str("10"))
            
            # Declare a new array to put all the distortion values (SSE), please view the report
            distortion = []
            
            # Determine the distance and the SSE for the whole data
            for i in range(len(new_data.values)):
                distances = [distance.euclidean(array_of_data[i], model.ctrds[ctrd]) for ctrd in model.ctrds]
                distances = min(distances)
                distortion.append(distances)
            
            # Count the SSE values over the number of data to get the distortion value
            dist = sum(distortion)/new_data.shape[0] 
            
            # Create a new table that consists of the label and the predicted table
            data_prediction_temp = pd.DataFrame(data_prediction, columns=["prediction"])
            data_test_temp['prediction'] = data_prediction_temp['prediction']
            data_test_temp['analyzed'] = data_test_temp['class'] + data_test_temp['prediction']
            data_test_temp = data_test_temp['analyzed']
            data_test_analyzed = data_test_temp.values.ravel()
            data_test_temp = data_test
            
            # Create a new array of k elements, please refer to table 1 of the report
            index_clustermaker = array_maker(clusters_k)
            
            # Initialize all the value of the array to 0
            dfObj = pd.DataFrame(index=clusters)
            for i in index_clustermaker:
                dfObj[i] = 0

            # Count the number of elements with a specified label and specified prediction
            for i in data_test_analyzed:
                for j in index_clustermaker:
                    for k in clusters:
                        if((i[0] == k) & (i[1] == j[0]) & (i[2] == j[1])):
                            dfObj[j][k] += 1
            
            # Count the number of Possible pairs
            values = (len(data_test_analyzed) * (len(data_test_analyzed)-1))/2
            TV = 0
            TP = 0

            x = dfObj.sum(axis = 0).to_frame()
            
            # Count the number of possible pairs that can be created in a single k
            # This value will be the number of true values
            x['Combined'] = (x[0] * (x[0]-1))/2
            del x[0]
            TV = x.sum(axis = 0)
            
            # Count the number of true positive by counting the possible pairs created based on the prediction
            for i in index_clustermaker:
                for j in clusters:
                    TP += (dfObj[i][j] * (dfObj[i][j] - 1))/2
                    
            # Count the number of true negative, the difference of true value pairs and true positive value pairs
            FP = TV - TP
            
            # Count the number of negative values, the difference of all possible pairs and positive pairs
            FV = values - TV
            
            # Initialize the number of False Negative
            FN = 0
            
            # Count the number of False Negative, the total number of possible pairs on the same colour created in a cluster
            for i in clusters:
                for j in index_clustermaker:
                    for k in index_clustermaker:
                        if(j < k):
                            FN += dfObj[j][i] * dfObj[k][i]
            
            # Count the number of False Positive, the difference of negative pairs and false negative pairs
            TN = FV - FN
            
            # Count the important metrics of k-means : Accuracy, Precision, Recall, and F-Score
            accuracy = ((TP + FN)/(TP + TN + FP + FN)) * 100
            precision = (TP/(TP + FP)) * 100
            recall = (TP/(TP + FN)) * 100
            fscore = (2 * precision * recall)/(precision + recall)
            
            # Insert all result to a single table to be processed
            array_of_values = [TV, TP, TN, FV, FP, FN, accuracy, precision, recall, fscore, dist]
            array_of_values = np.asarray(array_of_values, dtype='int')
            array_of_results = np.append(array_of_results, array_of_values, axis=0)
            array_of_results = array_of_results.astype(int)
            
        # Parse the result to a single value that can be passed
        array_of_results = array_of_results.reshape(int(len(array_of_results)/11), 11)
        
        # Use dataframes to plot te results
        to_plot_results = pd.DataFrame(array_of_results, columns=["TV", "TP", "TN", "FV", "FP", "FN", 
                                                                  "accuracy", "precision", "recall", "fscore", "dist"],
                                                                  index=[1,2,3,4,5,6,7,8,9,10])
        to_plot_prf = to_plot_results[["precision", "recall", "fscore"]]
        to_plot_dist = to_plot_results[["dist"]]
        to_plot_dist.astype('float')
        
        # Draw the diagrams
        lines = to_plot_prf.plot.line()
        lines2 = to_plot_dist.plot.line()
        print(lines)
        print(lines2)
        
        # Save the results to an external file and re-initialize the array_of_results
        np.savetxt(methods_names + "_" + str(l2normstatus) + ".csv", array_of_results, delimiter=",")
        array_of_results = []
