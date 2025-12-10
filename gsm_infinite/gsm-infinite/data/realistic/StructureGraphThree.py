import random 
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt 
import pickle
import os 

class Node: 
    def __init__(self, id, leveli = None): 
        self.id = id 
        self.name = None 
        
        # list in set 
        # from others to node 
        self.adjacent = set() 
        self.leveli = leveli 
        self.layercategory = None 
    
    def __str__(self): 
        return self.name 
    
    def add_adjacent(self, node): 
        self.adjacent.add(node) 
        node.adjacent.add(self) 

class GraphStructure: 
    def __init__(self, d, w0, w1): 
        self.d = d 
        self.w0 = w0 
        self.w1 = w1 
        self.e = 0 
        self.layers = [[] for _ in range(d)] 
        self.l = [self.w0 for _ in range(d)] 
        self.layers_names = None 
        self.nodes = set() 
    
    def fill_ite(self): 
        # Construct Gs with exactly li items on layer i ∈ [d]. 
        for i in range(self.d): 
            for j in range(self.l[i]): 
                newnod = Node(i * self.w1 + j, i) 
                self.nodes.add(newnod) 
                self.layers[i].append(newnod) 
    
    def construct_first(self): 
        # foreachitemaineachlayeri≥2do
        #     randomly select an item b in layer i − 1 and connect (a, b) in Gs. ⋄ this creates e− edges 
        for i in range(1, self.d): 
            for j in range(self.l[i]): 
                nod = self.layers[i][j] 
                idx22 = np.random.randint(0, len(self.layers[i - 1])) 
                node_past_le = self.layers[i - 1][idx22] 
                nod.add_adjacent(node_past_le) # add edge goes in nod and node_past_le 
                self.e += 1 
    
    def checklayerconnectivity(self, levelidx): 
        """
        output nodes that are not connected 
        """ 
        listofnodes = [] 
        for i, node in enumerate(self.layers[levelidx]): 
            if len(node.adjacent) == 0: 
                listofnodes.append(i) 
        return len(listofnodes), listofnodes 
    
    def construct_second(self, e): 
        # while number of edges < e do
        #     randomly select two items a,b from adjacent layers to create an edge in Gs. 
        countattempt = 0 
        while self.e < e: 
            countattempt += 1 
            if countattempt > e * 10: 
                raise ValueError("countattempt > needed") # This is needed to avoid infinite loop 
            levelidx1 = random.randint(0, (self.d - 1) - 1) 
            levelidx2 = levelidx1 + 1 
            
            numberofnodes, listofnodes = self.checklayerconnectivity(levelidx1) 
            if numberofnodes == 0: 
                idx1 = random.randint(0, self.l[levelidx1] - 1) 
            else: 
                idx1 = listofnodes[random.randint(0, numberofnodes - 1)] 
            idx2 = random.randint(0, self.l[levelidx2] - 1) 
            
            node1 = self.layers[levelidx1][idx1] 
            node2 = self.layers[levelidx2][idx2] 
            
            appov = True 
            for node in node2.adjacent: 
                if node1.id == node.id: 
                    appov = False 
                    break 
            if not appov: 
                continue 
            else: 
                node2.add_adjacent(node1) 
                self.e += 1 
    
    def construct_extra(self, e): 
        if self.d == 3: 
            picknode = self.layers[2][0] 
            for node in self.layers[1]: 
                picknode.add_adjacent(node) 
    
    def attachEnglish(self, hierarchical_categorizations = None, subcategories = None): 
        # pick from d consecutive categories 
        if self.d == 2 or self.d == 3: 
            categorynames = hierarchical_categorizations[0][: self.d] 
        else: 
            raise ValueError("d must be 2 or 3") 
        
        assert len(categorynames) == self.d 
        self.layer_names = categorynames 
        for node in self.nodes: 
            node.layercategory = self.layer_names[node.leveli] 
        
        for i in range(self.d): 
            layer_category = categorynames[i] 
            subcategoriestwo = subcategories[layer_category] 
            choices = list(subcategoriestwo.keys()) 
            subcategory = choices[random.randint(0, len(choices) - 1)] 
            # pick li items from the subcategory 20 
            if self.d == 2 or (self.d == 3 and i != 2): 
                nodenames = random.sample(subcategories[layer_category][subcategory], self.l[i]) 
                for node in self.layers[i]: 
                    node.name = nodenames.pop() 
            else: 
                # nodenames = ["Average Number of Newborn Children"] 
                nodenames = subcategories[layer_category][subcategory] # it seems that sample doesn't want list of one element 
                for node in self.layers[i]: 
                    node.name = nodenames[0] 
    
    def draw(self): 
        structure_graph = nx.Graph() 
        pos = {} 
        for i in range(self.d): 
            for j, node in enumerate(self.layers[i]): 
                structure_graph.add_node(node.name) 
                pos[node.name] = (j, -i) 
        
        added_edges = [] 
        for node in self.nodes: 
            for adj in node.adjacent: 
                if (adj.id, node.id) in added_edges: 
                    continue 
                else: 
                    added_edges.append((node.id, adj.id)) 
                    structure_graph.add_edge(node.name, adj.name, directed = False) 
        
        plt.figure(figsize = (20, 10)) 
        nx.draw(structure_graph, pos, with_labels = True, node_color = "lightblue", node_size = 5000, font_size = 10) 
        plt.title("Structure Graph") 
        plt.axis("off") 
        
        for i in range(self.d): 
            plt.text(-0.1, -(i + 0.1), self.layer_names[i], fontsize = 8, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)) 
        
        filename = "structure_graph.png" 
        plt.savefig(filename) 
    
    def save_graph_structure(graph_structure, filename):
        """
        Save the GraphStructure object to a file using pickle.
        
        :param graph_structure: The GraphStructure object to save
        :param filename: The name of the file to save to
        """
        with open(filename, 'wb') as file:
            pickle.dump(graph_structure, file)
        print(f"GraphStructure saved to {filename}") 
    
    def load_graph_structure(filename):
        """
        Load a GraphStructure object from a file.
        
        :param filename: The name of the file to load from
        :return: The loaded GraphStructure object
        """
        if not os.path.exists(filename):
            print(f"File {filename} does not exist.")
            return None
        
        with open(filename, 'rb') as file:
            graph_structure = pickle.load(file)
        print(f"GraphStructure loaded from {filename}")
        return graph_structure
                