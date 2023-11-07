import networkx as nx
import numpy
import random
import sys
import itertools

# Create a directed graph based on a .txt file. This code is from the assignment description.
def read_graph():
    fh = open(sys.argv[1], 'rb')
    G = nx.read_adjlist(fh, create_using=nx.DiGraph())
    fh.close()
    return G

# Use 'random' module to choose a random node (type 'str') among list of all nodes in graph. 
def random_node_char(G: nx.classes.digraph.DiGraph):
    return random.choice(list(G.nodes))

# Create a dictionary with a key for each node in graph, set all values to 0.
def init_dict(G: nx.classes.digraph.DiGraph):
    dict = {}
    for i in range(0, G.number_of_nodes()):
        dict[str(i)]=0
    return dict

# My implementation of the random surfer simulation.
def random_surf(G: nx.classes.digraph.DiGraph, E:int, m:float):
    dict = init_dict(G)
    node = random_node_char(G) # The current node
    out_edges = []
    
    for _ in range(E): # Go through E times
        dict[node] += 1
        out_edges = list(G.out_edges(node))
        rand_prob = random.random()
        
        if (rand_prob <= m) or (len(out_edges) == 0) :  # JUMP if rand_prob with value 0-1 is <= m (m=0.15) OR if node has no outgoing edges.
            node = str(random_node_char(G))  
        else:                                           # Otherwise WALK to a random neighbor from the list of neighbouring nodes.
            rand_num = random.randint(0, len(out_edges)-1)
            node = str(out_edges[rand_num][1])
            
    return dict

# My implementation of PageRank, which computes an approximation of the eigenvector
# Here I iteratively compute x_k+1 = (1-m)*Ax_k + (1-m)*Dx_k + m*Sx_k
def page_rank(G: nx.classes.digraph.DiGraph, E: int, m: float):
    size = len(G)
    backlinks = lambda x : G.in_edges(x)
    branch_degree = lambda x : G.out_degree(x)
    
    # SETUP, and find dangling nodes in the process.
    danglings = list()
    x_k = numpy.empty(size, dtype=float)
    for i in G.nodes: 
        if branch_degree(i) == 0 : 
            danglings.append(i)
        x_k[int(i)] = (1/size)
    
    # S TERM. 
    # Note that mSx_k = m/n * a n*1 matrix of 1's, if we assume that x_k is column stochastic (so sum of its values is 1). So we treat it as a constant
    mS = m/size

    for _ in range(E): # Each iteration is a step        
        
        #S TERM. 
        # I initialize (x_k+1)_i to be m(Sx_k)_i
        x_k_plus_1 = mS

        # D TERM. 
        # Sum the values of dangling nodes for term "+(1-m)*Dx_k". Note that matrix D is 0's except for columns j where node is dangling.
        dangling_vals = 0
        for x_k_node in danglings: 
            dangling_vals += x_k[int(x_k_node)]
        Dx_k = dangling_vals/size         
        term_with_D = (1-m)*Dx_k
                
        # A TERM. 
        # A keeps track of OUTGOING links from column to row. Term "(1-m)*Ax_k"
        ax_k = numpy.zeros(size, float)
        for to_node in range(size):
            sum = 0.0
            for (from_node, _) in backlinks(str(to_node)):
                src_node_out_nr = branch_degree(from_node)
                if src_node_out_nr != 0:
                    sum += (x_k[int(from_node)]/src_node_out_nr) 
            ax_k[int(to_node)] = sum
        
        term_with_A = (1-m)*ax_k
        
        # UPDATE x_k
        x_k_plus_1 += term_with_A + term_with_D        
        x_k = x_k_plus_1
                      
    return x_k

# Helper method, runs the random surfer simulation and processes the result.
def init_random_surf(G: nx.classes.digraph.DiGraph, E: int, m: float): 
    rand_surf_dict = random_surf(G, E, m)
    sorted_dict = dict(sorted(rand_surf_dict.items(), key=lambda item:item[1], reverse=True)) # Sort the items of the dictionary based on its values (aka "item[1]")
    normalized_dict = {k:v/E for (k, v) in sorted_dict.items()} # Normalize values of dict by dividing by E
    sliced_dict = dict(itertools.islice(normalized_dict.items(), 10))  # Only get 10 first values
    
    value_sum = round(sum(normalized_dict.values()), 6) # Check if column stochastic, sum of values should be 1 after normalising by E.
    return sliced_dict, value_sum

# Helper method, runs the page_rank method and processes the result.
def init_page_rank(G: nx.classes.digraph.DiGraph, E: int, m: float):
    res_arr = page_rank(G, E, m)
    
    indexes = numpy.empty(G.number_of_nodes(), dtype='U25')
    for i in range(G.number_of_nodes()): indexes[int(i)] = str(i)
    zipped_res = list(zip(indexes, res_arr))
    sorted_res = sorted(zipped_res, key=lambda a: a[1], reverse=True)
    sliced_res = sorted_res[:10]
    rounded_res = list(map(lambda t: (t[0], round(t[1], 4)), sliced_res))
    
    sum = round((numpy.sum(res_arr)), 6)    
    return rounded_res, sum

def main():
    G = read_graph()
    E = 10000 # Number of iterations
    m = 0.15 # Damping factor
    
    print(str(E) + " iterations.\n")
    
    surf_10_res, surf_val_sum = init_random_surf(G, E, m)
    print("Top 10 most visited nodes by the random surfer simulation: \n" + str(surf_10_res))
    print ("Sum of all values: " + str(surf_val_sum) + "\n")

    page_rank_res, page_rank_val_sum = init_page_rank(G, E, m)
    print("Top 10 nodes with highest importance score according to PageRank algorithm: \n" + str(page_rank_res))
    print ("Sum of all values: " + str(page_rank_val_sum) + "\n")

if __name__ == "__main__":
    main()