import numpy as np
import random
  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
class RTLearner(object):
    def __init__(self, leaf_size = 1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        pass
  		  	   		  		 			  		 			     			  	 
    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        #followed the JQ Quinlan method
        if data_x.shape[0] <= self.leaf_size:
            return np.array([(-1, np.mean(data_y), None, None)])
        if len(np.unique(np.array(data_y))) == 1:
            return np.array([(-1, np.mean(data_y), None, None)])
        else:
            #i = self.find_best_feature(data_x, data_y)
            i = random.randint(0, data_x.shape[1]-1)
            SplitVal = np.median(data_x[:,i])
            sth_else = data_x[:,i] <= SplitVal
            if len(np.unique(np.array(sth_else))) == 1:
                return np.array([(-1, np.mean(data_y), None, None)])
            else:
                lefttree = self.build_tree(data_x[data_x[:, i] <= SplitVal], data_y[data_x[:, i] <= SplitVal])
                righttree = self.build_tree(data_x[data_x[:, i] > SplitVal], data_y[data_x[:, i] > SplitVal])

            root = np.array([(i, SplitVal, 1, lefttree.shape[0]+1)])
            root_tree = np.vstack((root, lefttree, righttree))
        return root_tree

    def query(self, points):
        predictions = np.array([])
        for point in points:
            current_node = 0
            while self.tree[current_node][0] != -1:
                i = int(self.tree[current_node][0])
                Split_val = self.tree[current_node][1]
                if point[i] <= Split_val:
                    current_node += int(self.tree[current_node][2])
                else:
                    current_node += int(self.tree[current_node][3])
            current_predict = self.tree[current_node][1]
            predictions = np.append(predictions,current_predict)
        return predictions


