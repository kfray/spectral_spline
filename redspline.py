#fits a  spectral spline to data

#TO DO: send email when done
#TO DO: write files with datetime appended
#TO DO: save seed
#TO DO: paralleleized eigen value deocomp

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy import interpolate
from apgl.graph import SparseGraph
from scipy.stats import bernoulli, pearsonr
import datetime
import os
import random

class Spectral:
    """
        Provides a fitting, plotting, predicting, and scoring
        object for graph spectrum learning for link
        prediction.
    """

    def __init__(self, file_name, n):
        """ Reads in a specially formatted graph file"""
        self.fileName = file_name
        self.name = os.path.splitext(os.path.basename(file_name))[0]
        self.vertices = range(n)
        self.adjList = adj_from_listfile(file_name, n)
        print "adjacency matrix loaded"

    # def setRep(self, *repList):
    #     if not set(repList).issubset(set('A', 'nA', 'L', 'nL')):
    #         print "INPUT ERROR"
    #     elif 'A' in repList:
    #         self.A = self.adjList
    #     elif 'nA' in repList:
    #         pass
    #     elif 'L' in repList:
    #         pass
    #     elif 'nL' in repList:
    #         pass

#laplacianMatrix()
#normalisedLaplacianSym()
#row_sums = A.sum(axis=1)
#new_matrix = A / row_sums[:, numpy.newaxis]
    
    def fit(self, function, source=False, target=False, split=False, k=False):
        """
            Sets up the training and test set and fits the spectral
            learning function to it.
        """

        self.train, self.test = splitGraph(self.adjList, split)
        print "train test split created"
        cc = self.train.findConnectedComponents()[0]
        print "connected component found"
        self.cc = self.train.subgraph(cc)
        print "connected component assigned"
        self.source , self.target =  splitGraph(self.cc)
        print "source and target created"
        self.rrVal, self.rrVec = np.linalg.eigh(self.source.adjacencyMatrix())
        print "eigen decomposition completed"
        if k:
            self.rrVal = np.take(self.rrVal, range(k))
            self.rrVec = np.take(self.rrVec, range(k),1)
            print "rank reduced approximation found"
        B = np.dot( self.target.adjacencyMatrix(), self.rrVec)
        self.targetTrans = np.dot( np.transpose(self.rrVec), B)
        print "transformed target completed"
        self.fitted = function(self.rrVal, self.targetTrans.diagonal())
        print "learning function fitted"

    def plot(self):
        """
            Produces a plot of eigenvalues against the Frobenius norm fitting
        """
        self.plotFile = plot_against(self.rrVal, self.targetTrans.diagonal() , self.fitted, self.name)
        print "plot produced"
        
    def predict(self, graph=False):
        """
            If test is false, then use the test set
        """
        if not graph:
            graph = self.train
        Val, Vec = np.linalg.eigh(graph.adjacencyMatrix())
        print "eigen decomposition of training set completed"
        return projRot(Vec, np.diag(self.fitted(Val)))
        
    def score(self, ):
        """
        """
        testEdges = self.test.getAllEdges()
        print "test edges retrieved"
        predMat = self.predict()
        print "prediction for scoring made"
        predicted = []
        true = []
        for pair in testEdges:
            i, j = pair
            predicted.append(predMat[i,j])
            true.append(1)
        print "scoring tallied"
        return pearsonr( predicted, true)

    def document(self):
        with open(self.name + ".md", 'a') as md:
            md.write(mdText(self))


class Bootstral(Spectral):

    def __init__(self, spectral_obj, i, n):
        self.fileName = spectral_obj.fileName
        self.name = spectral_obj.name + "_" + str(i)
        self.vertices = random.sample(spectral_obj.vertices, n)
        print "sample chosen"
        self.adjList = spectral_obj.adjList.subgraph(self.vertices)
        print "graph constructed"

def projRot(omatrix, cmatrix):
    return np.dot(np.transpose(omatrix), np.dot(cmatrix, omatrix))
#interpolate.UnivariateSpline(self.rrVal, self.targetTrans.diagonal())
    
def mdText(obj):
    return "#" + obj.name + ": " + "source" +" -> " + "target" + ":" + "fit" + " \n" + \
            "Time split: " + "timeSplit" + "\n" + \
            "##Newtowrk Statistics \n " + "networkStats" + "\n" + \
            "Pickled model" + "pickled" + "\n" + \
            "Score" + "score" + "\n" + \
            "![](" + obj.plotFile + ")" + "\n"

def getList(data, skip=4):    
    with open(data, "r") as big_file:
        i = 0
        myData = big_file.readlines()
    adj1 = []
    adj2 = []
    for each in myData[skip:]:
        i,j = each.strip('\n').split('\t')
        adj1.append(int(i))
        adj2.append(int(j))
    print "data retrieved"
    return adj1, adj2

def getAdjDict(data, skip=4):    
    with open(data, "r") as big_file:
        i = 0
        myData = big_file.readlines()
    adjDict = {}
    for each in myData[skip:]:
        i,j = each.strip('\n').split('\t')
        if i in adjDict:
            adjDict[i].append[j]
        else:
            adjDict[i] = [j]
        if j in adjDict:
            adjDict[j].append[i]
        else:
            adjDict[j] = [i]
    print "data retrieved"
    return adjDict

def adj_from_listfile(adj_list, n, vertices=False, skip=4):
    if not vertices:
        vertices = range(n)
    adj = SparseGraph(len(vertices))
    print "sparse graph created"
    adj1, adj2 = getList(adj_list, skip)
    for i,j in zip(adj1,adj2):
        if i >= n:
            break
        if j >= n:
            pass
        elif i in vertices and j in vertices:
            adj[i, j] = 1
    print "adj list constructed"
    return adj

def arrange(adj1, adj2):
    index = 0
    for i,j in zip(adj1,adj2):
        if i > j:
            adj1[index] = j
            adj2[index] = i
        index+=1
    return adj1, adj2

def breadthFirstSearch(adjDict):
    stack = []
    def explore(i):
        stack = set(stack + adjDict[i])




def splitGraph(graph, random=True ,p=1/float(3)):
    all_edges = graph.getAllEdges()
    vertices =  graph.vlist
    inc = bernoulli.rvs(p, size = len(all_edges))
    train = SparseGraph(vertices)
    test =  SparseGraph(vertices)
    train.addEdges(all_edges[inc==0])
    test.addEdges(all_edges[inc==1])
    return train, test

def plot_against(D, B, fit, name="fit"):
    fig_name = name + "_" + str(datetime.date.today()) + '.png'
    plt.plot(D, B, 'ro', ms=5)
    plt.plot(D, fit(D), 'g', lw=3)
    fit.set_smoothing_factor(0.5)
    plt.plot(D, fit(D), 'b', lw=3)
    plt.savefig(fig_name)
    return fig_name
