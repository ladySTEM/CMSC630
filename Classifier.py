"""

CMSC 630 - Part 3
Gabriella Graziani

Class utilized to perform KNN classification and 10-fold Cross Validation

"""


import numpy as np
import math
import statistics 


class Classifier():
  
  
        
    def knn(self,test_data,train_data,k):
        """
        This function implements k'th nearest neighbors on an input of test and training data.
        
        Inputs: 
            test_data - unclassified data which is sorted based on its distance from training data
            train_data - data whose classification is known
            
        Outputs: 
            accuracy - accuracy of test_data classification
        """
        
        success = []
        
        for index1, row1 in test_data.iterrows():
            
            dist = []
            
            point1 = test_data.loc[index1].values
            point1 = point1[:-1]
            #print("Point1: {} Row: {}".format(point1,row))
            
            
            for index2, row2 in train_data.iterrows():
                point2 = train_data.loc[index2].values
                point2 = point2[:-1]
                
                
                euc_dist = [np.linalg.norm(point1-point2) , row2['name']]
                
                dist.append(euc_dist)
            
            dist = sorted(dist,key=lambda l:l[0])
            near_neigh= [row[1] for row in dist[:k]]
            data_label = statistics.mode(near_neigh)
            
            if(data_label == row1['name']):
                success.append(1)
            else: 
                success.append(0)
       # near_neigh = dist[0:k]
        accuracy = sum(success)/len(success)*100
               # print("perim: {}, area: {}, index: {}".format(perim_k,area_k,index))
       # print("Distance array: {} \n \n Near Neighbors Labels: {} \nTest data Label:{} \n".format(dist,near_neigh,data_label))
       # print("Success Vector: {}".format(success))
        print("Accuracy: {}".format(accuracy))
        return accuracy
    
    def cross_validation(self, data, k):
        """
        This function perform 10-fold cross validation where data is separated into 
        10 separate sub-arrays and undergos 10 iterations. For each iterations, a different 
        subarray is a designated "testing" set and the remaining are considered "training" sets.
        The accuracy of each iteration is recorded and averaged for a new, total average accuracy.
        
        Inputs: 
            data - total dataset, python pandas dataframe
            k - int value which determines how many neighbors each data point will be compared to
            
        Outputs: 
            average_acc - average accuracy of validation
        """
        
        # split data into 10 different datasets
        data = data.sample(frac=1)
        
        data_frame_array = []
        
   
        
        #print("Mydata: \n",data)
        
        accuracy = []
        
        for i in range(10):
            
            beg_index = math.ceil(len(data)/10)*i
            end_index = beg_index+(math.ceil(len(data)/10) -1)
            
            if (end_index>len(data)-1):
                end_index = len(data)-1
            
            test_data = data.iloc[beg_index:end_index]
            train_data = data.drop(data.index[beg_index:end_index])
            prelim_acc = self.knn(test_data,train_data,k)
            accuracy.append(prelim_acc)
           # print("Beg Index: {} End Index: {}".format(beg_index,end_index))
           # print("Train_data: \n",train_data)
           # print("Test_data: \n",test_data)
        average_acc = np.mean(accuracy)
        print("Train data length: ",len(train_data))
        
        return average_acc
        
        
        
                
                
    