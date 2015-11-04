from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy import interpolate
from apgl.graph import SparseGraph
from scipy.stats import bernoulli

def spectralSpline(A_0, A_1, k):
	A_0a = normalize(A_0.adjacencyMatrix())
	A_1a = A_1.adjacencyMatrix()
	Da, Ua = np.linalg.eigh(A_0a)
	U = np.take(Ua, range(k), 1)
	D = np.take(Da, range(k))
	B = np.dot( A_1a, U)
	B = np.dot( np.transpose(U), B)
	spl = interpolate.UnivariateSpline(D, B.diagonal())
	plot_against(D, B.diagonal() , spl)
	return spl

def normalize(A):
	D = A.degreeSequence()
	D = [degree**(-1/2) for degree in D]
	D = np.daigflat(D)
	return np.dot(D, np.dot(A,D))

def adj_from_listfile(adj_list, n, vertices=False, skip=4):
	if not vertices:
		vertices = range(n)
	adj = SparseGraph(len(vertices))
	with open(adj_list, 'r') as a:
		k = 0
		for line in a:
			if k < skip:
				k+=1
			else:
				i, j = line.split()
				i = int(i)
				j = int(j)
				if i in vertices and j in vertices:
					adj[i, j] = 1
					adj[j, i] = 1
	return adj

def train_test(graph, p=1/float(3)):
	all_edges = graph.getAllEdges()
	vertices =  graph.vlist
	inc = bernoulli.rvs(p, size = len(all_edges))
	train = SparseGraph(vertices)
	test =  SparseGraph(vertices)
	train.addEdges(all_edges[inc==0])
	test.addEdges(all_edges[inc==1])
	return train, test

def plot_against(D, B, fit):
	plt.plot(D, B, 'ro', ms=5)
	plt.plot(D, fit(D), 'g', lw=3)
	fit.set_smoothing_factor(0.5)
	plt.plot(D, fit(D), 'b', lw=3)
#	plt.show()
	plt.savefig('spline_fit.png')

def main():
    parser = argparse.ArgumentParser(description="Do something.")
    parser.add_argument('-A_0','--A_0_file', type=str, required=True)
    parser.add_argument('-N','--num_vertices', type=int, required=True)
    parser.add_argument('-E','--num_edges', type=int, required=True)
#    A_0 = np.loadtxt(A_0_file)
#    A_1 = np.loadtxt(A_1_file)
    args = parser.parse_args().__dict__
    n = args['num_vertices']
 #   vertices = list(np.random.choice(n, round(100), replace=False ))
 	###GET biggest connected component from graph
 	###only use those vertices
    full_graph = adj_from_listfile(args['A_0_file'], n)
    connected_vertices = full_graph.findConnectedComponents()[-1]
    subgraph = full_graph.subgraph(connected_vertices)
    train, test = train_test(subgraph)
#    A_0 = adj_from_listfile(args['A_0_file'], n, vertices, p=1/3)
    return spectralSpline(train, test, k= 126)

if __name__ == '__main__':
    main()
