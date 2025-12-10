from typing import Dict, Set, List, Tuple 
import pydot 
import os 

import math 

class ParameterNode: 
    def __init__(self, name): 
        self.name = name 
        self.edgetolist = [] 
        self.edgefromlist = [] 
        self.type = None 
    
    def __str__(self): 
        return self.name 
    
    def __eq__(self, other): 
        if other == None: 
            return False 
        return (self.name == other.name) and (self.type == other.type) 
    
    def __hash__(self): 
        return hash(self.name) 

class AbstractParameter(ParameterNode): 
    def __init__(self, name): 
        super().__init__(name) 
        self.type = "abstract" 
        self.c_op = None 
        self.m_op = None 
        
        self.variable = None 
        self.value = None 
        self.notation = None # only used for two factors 
        self.reversibletext = None 
    
    def __repr__(self): 
        return self.name 
    
    def alldependentparameters(self): 
        listab = [] 
        listin = [] 
        for item in self.edgefromlist: 
            if isinstance(item, InstanceParameter): 
                if item not in listin: 
                    listin.append(item) 
            else: 
                # remove unreasonable abstract 
                if len(item.edgefromlist) == 0: # abstract parameter that is unreasonable 
                    continue 
                if item not in listab: 
                    listab.append(item) 
                    listabt, listint = item.alldependentparameters() 
                    for item2 in listabt: 
                        if item2 not in listab: 
                            listab.append(item2) 
                    for item2 in listint: 
                        if item2 not in listin: 
                            listin.append(item2) 
        return listab, listin 

class AbstractParameterSpecial(AbstractParameter): 
    def __init__(self, name, instanceparameter): 
        super().__init__(name) 
        self.isonamicinstanceparameter = instanceparameter 
    
class InstanceParameter(ParameterNode): 
    def __init__(self, name): 
        super().__init__(name) 
        self.type = "instance" 
        self.c_op = None 
        self.m_op = None 
        
        self.variable = None 
        self.value = None 
        self.notation = None # only used for two factors 
        
        self.rngv = None # only needed when self.rng in dependencies 
        self.rngnot = None # only needed when self.rng in 
        
        self.coretoflag = 0 
        self.peripheralabflat = 0 
        self.includedflag = 0 
        
        self.reversibletext = None 
    
    def __repr__(self): 
        return self.name 

class RNG(ParameterNode): 
    def __repr__(self): 
        return self.name 

class DependencyGraph: 
    def __init__(self): 
        self.abstractparameters = [] 
        self.instanceparameters = [] 
        self.rng = RNG("RNG") 
    
    def add_abstractparameter(self, abstractparameter: AbstractParameter): 
        if abstractparameter not in self.abstractparameters and len(abstractparameter.edgefromlist) != 0: 
            self.abstractparameters.append(abstractparameter) 
            # assert False, "found the same abstract para 
        listab, listin = abstractparameter.alldependentparameters() 
        for item in listab: 
            if item not in self.abstractparameters: 
                if len(item.edgefromlist) != 0: 
                    self.abstractparameters.append(item) 
                else: 
                    print("item {} edgefromlist {} edgetolist {}".format(item, item.edgefromlist, item.edgetolist)) 
                    exit(0) 
                    continue 
            else: 
                # the following lines 
                # adjust for the case from recusrion 
                # it is needed for abstract 
                item2 = self.abstractparameters[self.abstractparameters.index(item)] 
                item2.edgefromlist = item.edgefromlist + [k for k in item2.edgefromlist if k not in item.edgefromlist] 
                item2.edgetolist = item.edgetolist + [k for k in item2.edgetolist if k not in item.edgetolist] 
                # item2 = item 
                item2.edgetolist = item.edgetolist + [k for k in item2.edgetolist if k not in item.edgetolist] 
                item2.edgefromlist = item.edgefromlist + [k for k in item2.edgefromlist if k not in item.edgefromlist] 
                # item2 = item 
        for item in listin: 
            if item not in self.instanceparameters: 
                self.instanceparameters.append(item) 
        return True 
    
    def op(self): 
        sum = 0 
        for item in self.abstractparameters: 
            sum += len(item.edgefromlist) 
        for item in self.instanceparameters: 
            sum += len(item.edgefromlist) 
        return sum 
    
    def copy(self): 
        newgraph = DependencyGraph() 
        for item in self.abstractparameters: 
            newgraph.abstractparameters.append(item) 
        for item in self.instanceparameters: 
            newgraph.instanceparameters.append(item) 
        return newgraph 
    
    def next_one_two(self, topo: List[ParameterNode]): 
        # first step, find parameters that are not in topo 
        output: List[ParameterNode] = [] 
        for item in self.abstractparameters: 
            if item not in topo: 
                output.append(item) 
        for item in self.instanceparameters: 
            if item not in topo: 
                output.append(item) 
        
        # second step, find parameters in output that has one or more dependencies in topo 
        nextone: List[ParameterNode] = [] 
        for candidate in output: 
            for item in topo: 
                if item in candidate.edgetolist: # candidate -> item 
                    nextone.append(candidate) 
                    break 
        
        # third step, find parameters in output that doesn't have dependencies in output 
        nexttwo: List[ParameterNode] = [] 
        for candidate in output: 
            foundoneexception = False 
            for item in candidate.edgetolist: # candidate -> item 
                if item in output: 
                    foundoneexception = True 
                    break 
            if not foundoneexception: 
                nexttwo.append(candidate) 
        
        return nextone, nexttwo, output 
    
    def add_edge(self, from_node: ParameterNode, to_node: ParameterNode): 
        # from_node -> to_node 
        if isinstance(from_node, AbstractParameter): 
            if from_node not in self.abstractparameters and len(from_node.edgefromlist) != 0: 
                self.abstractparameters.append(from_node) 
        if isinstance(from_node, InstanceParameter): 
            if from_node not in self.instanceparameters: 
                self.instanceparameters.append(from_node) 
        if isinstance(to_node, AbstractParameter) and len(to_node.edgefromlist) != 0: 
            if to_node not in self.abstractparameters: 
                self.abstractparameters.append(to_node) 
        if isinstance(to_node, InstanceParameter): 
            if to_node not in self.instanceparameters: 
                self.instanceparameters.append(to_node) 
        
        if to_node not in from_node.edgetolist: 
            from_node.edgetolist.append(to_node) 
        if from_node not in to_node.edgefromlist: 
            to_node.edgefromlist.append(from_node) 
    
    def has_cycle(self):
        # State for each node: 0 = unvisited, 1 = visiting (in current path), 2 = visited 
        nodelist = self.abstractparameters + self.instanceparameters 
        node_state = {node: 0 for node in nodelist} 

        def dfs(node):
            if node_state[node] == 1:  # If visiting a node already in the current path -> cycle found
                return True
            if node_state[node] == 2:  # If already visited, no need to check this node again
                return False

            # Mark the node as visiting
            node_state[node] = 1

            # Visit all nodes in the edgetolist (nodes this node points to)
            for neighbor in node.edgetolist:
                if dfs(neighbor):
                    return True

            # Mark the node as fully visited
            node_state[node] = 2
            return False

        # Check for a cycle in the graph starting from each unvisited node 
        for node in nodelist: 
            if node_state[node] == 0:  # Unvisited node
                if dfs(node):
                    return True 

        return False 
    
    def drawmess(self, core_graph): 
        coreparams = core_graph.abstractparameters + core_graph.instanceparameters + [core_graph.rng] 
        path = os.pathsep + os.getcwd() 
        
        graph = pydot.Dot(graph_type = "digraph", rankdir = "LR", splines = "true") 
        
        # print("op {} abstractparameters {} instanceparameters {}".format(self.op(), len(self.abstractparameters), len(self.instanceparameters))) 
        
        allnodelist = [] 
        for item in self.abstractparameters: 
            if item in coreparams: 
                continue 
            node = pydot.Node(item.name, shape = "ellipse", fillcolor = "lightblue", color = "none", style = "filled") 
            graph.add_node(node) 
            allnodelist.append(node) 
        
        for item in self.instanceparameters: 
            if item in coreparams: 
                continue 
            name = "{}({})".format(item.name, len(item.edgefromlist)) 
            if item.coretoflag == 0: 
                node = pydot.Node(name, shape = "rectangle", fillcolor = "lightblue", color = "none", style = "filled") 
            else: 
                node = pydot.Node(name, shape = "rectangle", fillcolor = "lightgreen", color = "none", style = "filled") 
            graph.add_node(node) 
            allnodelist.append(node) 
        
        for item in self.abstractparameters: 
            # print("item name {} length edgefromlist {} length edgetolist {}".format(item.name, len(item.edgefromlist), len(item.edgetolist))) 
            for item2 in item.edgefromlist: 
                if item2 in coreparams: 
                    continue 
                name = "{}({})".format(item2.name, len(item2.edgefromlist)) 
                graph.add_edge(pydot.Edge(name, item.name, color = "red")) 
        
        for item in self.instanceparameters: 
            for item2 in item.edgefromlist: 
                if item2 in coreparams: 
                    continue 
                name = "{}({})".format(item2.name, len(item2.edgefromlist)) if not isinstance(item2, AbstractParameter) else item2.name 
                graph.add_edge(pydot.Edge(name, "{}({})".format(item.name, len(item.edgefromlist)), color = "black")) 
        
        graph.set_size("200,200") 
        
        # Graph attributes for circular layout
        graph.set_graph_defaults(overlap='false', splines='true', layout='neato') 
        
        graph.write_png("noise.png") 

    def draw1(self): 
        path = os.pathsep + os.getcwd() 
        
        graph = pydot.Dot(graph_type = "digraph", rankdir = "LR", splines = "true") 
        
        # print("op {} abstractparameters {} instanceparameters {}".format(self.op(), len(self.abstractparameters), len(self.instanceparameters))) 
        
        allnodelist = [] 
        for item in self.abstractparameters: 
            node = pydot.Node(item.name + " (a)", shape = "ellipse", fillcolor = "lightblue", color = "none", style = "filled") 
            graph.add_node(node) 
            allnodelist.append(node) 
        
        for item in self.instanceparameters: 
            node = pydot.Node(item.name + " (i)", shape = "rectangle", fillcolor = "lightblue", color = "none", style = "filled") 
            graph.add_node(node) 
            allnodelist.append(node) 
        if len(allnodelist) >= 15: 
            radius = 100 
        else: 
            radius = 50 
        
        # for i, node_name in enumerate(nodes): 
        for i, node in enumerate(allnodelist): 
            n = len(allnodelist) 
            angle = 2 * math.pi * i / n
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            # Get the node and set its position
            # node = graph.get_node(node.get_name())[0] 
            node.set('pos', f"{x},{y}!") 
        
        for item in self.abstractparameters: 
            # print("item name {} length edgefromlist {} length edgetolist {}".format(item.name, len(item.edgefromlist), len(item.edgetolist))) 
            for item2 in item.edgefromlist: 
                childrenname = item2.name + " (a)" if isinstance(item2, AbstractParameter) else item2.name + " (i)" 
                graph.add_edge(pydot.Edge(childrenname, item.name + " (a)", color = "red")) 
        
        for item in self.instanceparameters: 
            # print("item name {} length edgefromlist {} length edgetolist {}".format(item.name, len(item.edgefromlist), len(item.edgetolist))) 
            for item2 in item.edgefromlist: 
                childrenname = item2.name + " (a)" if isinstance(item2, AbstractParameter) else item2.name + " (i)" 
                graph.add_edge(pydot.Edge(childrenname, item.name + " (i)", color = "black")) 
        
        graph.set_size("200,200") 
        
        # Graph attributes for circular layout
        graph.set_graph_defaults(overlap='false', splines='true', layout='neato') 
        
        graph.write_png("dependency_graph.png") 
    
    def draw2(self): 
        path = os.pathsep + os.getcwd() 
        
        graph = pydot.Dot(graph_type = "digraph", rankdir = "LR", splines = "true") 
        
        # print("op {} abstractparameters {} instanceparameters {}".format(self.op(), len(self.abstractparameters), len(self.instanceparameters))) 
        
        allnodelist = [] 
        for item in self.abstractparameters: 
            node = pydot.Node(item.name + " (a)", shape = "ellipse", fillcolor = "lightblue", color = "none", style = "filled") 
            graph.add_node(node) 
            allnodelist.append(node) 
        
        for item in self.instanceparameters: 
            node = pydot.Node(item.name + " (i)", shape = "rectangle", fillcolor = "lightblue", color = "none", style = "filled") 
            graph.add_node(node) 
            allnodelist.append(node) 
        if len(allnodelist) >= 15: 
            radius = 100 
        else: 
            radius = 50 
        
        # for i, node_name in enumerate(nodes): 
        for i, node in enumerate(allnodelist): 
            n = len(allnodelist) 
            angle = 2 * math.pi * i / n
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            # Get the node and set its position
            # node = graph.get_node(node.get_name())[0] 
            node.set('pos', f"{x},{y}!") 
        
        for item in self.abstractparameters: 
            # print("item name {} length edgefromlist {}".format(item.name, len(item.edgefromlist))) 
            for item2 in item.edgefromlist: 
                childrenname = item2.name + " (a)" if isinstance(item2, AbstractParameter) else item2.name + " (i)" 
                graph.add_edge(pydot.Edge(childrenname, item.name + " (a)", color = "red")) 
        
        for item in self.instanceparameters: 
            # print("item name {} length edgefromlist {}".format(item.name, len(item.edgefromlist))) 
            for item2 in item.edgefromlist: 
                childrenname = item2.name + " (a)" if isinstance(item2, AbstractParameter) else item2.name + " (i)" 
                graph.add_edge(pydot.Edge(childrenname, item.name + " (i)", color = "black")) 
        
        graph.set_size("200,200") 
        
        # Graph attributes for circular layout
        graph.set_graph_defaults(overlap='false', splines='true', layout='neato') 
        
        graph.write_png("dependency_graph_2.png") 
    
    def visualization_topo(self, topo: List[ParameterNode]): 
        path = os.pathsep + os.getcwd() 
        
        graph = pydot.Dot(graph_type = "digraph", rankdir = "LR", splines = "curved") 
        
        # print("op {} abstractparameters {} instanceparameters {}".format(self.op(), len(self.abstractparameters), len(self.instanceparameters))) 
        
        allnodelist = [] 
        for item in topo: 
            if isinstance(item, AbstractParameter): 
                node = pydot.Node(item.name + " (a)", shape = "ellipse", fillcolor = "lightblue", color = "none", style = "filled") 
            else: 
                node = pydot.Node(item.name + " (i)", shape = "rectangle", fillcolor = "lightblue", color = "none", style = "filled") 
            graph.add_node(node) 
            allnodelist.append(node) 
        
        # for i, node_name in enumerate(nodes)
        for i, node in enumerate(allnodelist): 
            interval = 5 
            x = i * interval 
            y = 0  # All nodes on the same y-coordinate
            node.set('pos', f"{x},{y}!") 
        
        for item in topo: 
            for item2 in item.edgefromlist: 
                if isinstance(item, AbstractParameter): 
                    childrenname = item2.name + " (a)" if isinstance(item2, AbstractParameter) else item2.name + " (i)" 
                    graph.add_edge(pydot.Edge(childrenname, item.name + " (a)", color = "red")) 
                else: 
                    childrenname = item2.name + " (a)" if isinstance(item2, AbstractParameter) else item2.name + " (i)" 
                    graph.add_edge(pydot.Edge(childrenname, item.name + " (i)", color = "black")) 
        
        graph.set_size("{},{}".format(len(allnodelist) * 5, len(allnodelist) * 5)) 
        
        # Graph attributes for circular layout
        graph.set_graph_defaults(overlap='false', splines='true', layout='neato') 
        
        graph.write_png("topo.png") 
    
    def draw3(self): 
        path = os.pathsep + os.getcwd() 
        
        graph = pydot.Dot(graph_type = "digraph", rankdir = "LR", splines = "true") 
        
        # print("op {} abstractparameters {} instanceparameters {}".format(self.op(), len(self.abstractparameters), len(self.instanceparameters))) 
        
        allnodelist = [] 
        for item in self.abstractparameters: 
            node = pydot.Node(item.name + " (a)", shape = "ellipse", fillcolor = "lightblue", color = "none", style = "filled") 
            graph.add_node(node) 
            allnodelist.append(node) 
        
        for item in self.instanceparameters: 
            node = pydot.Node(item.name + " (i)", shape = "rectangle", fillcolor = "lightblue", color = "none", style = "filled") 
            graph.add_node(node) 
            allnodelist.append(node) 
        
        node = pydot.Node("RNG", shape = "ellipse") 
        allnodelist.append(node) 
        
        if len(allnodelist) >= 15: 
            radius = 100 
        else: 
            radius = 50 
        
        # for i, node_name in enumerate(nodes): 
        for i, node in enumerate(allnodelist): 
            n = len(allnodelist) - 1 
            angle = 2 * math.pi * i / n
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            # Get the node and set its position
            # node = graph.get_node(node.get_name())[0] 
            node.set('pos', f"{x},{y}!") 
            
            if node.get_name() == "RNG": 
                node.set('pos', f"{0},{0}!") 
        
        for item in self.abstractparameters: 
            # print("item name {} length edgefromlist {}".format(item.name, len(item.edgefromlist))) 
            for item2 in item.edgefromlist: 
                childrenname = item2.name + " (a)" if isinstance(item2, AbstractParameter) else item2.name + " (i)" 
                graph.add_edge(pydot.Edge(childrenname, item.name + " (a)", color = "red")) 
        
        for item in self.instanceparameters: 
            # print("item name {} length edgefromlist {}".format(item.name, len(item.edgefromlist))) 
            for item2 in item.edgefromlist: 
                childrenname = item2.name + " (a)" if isinstance(item2, AbstractParameter) else item2.name + " (i)" 
                graph.add_edge(pydot.Edge(childrenname, item.name + " (i)", color = "black")) 
        
        graph.set_size("200,200") 
        
        # Graph attributes for circular layout
        graph.set_graph_defaults(overlap='false', splines='true', layout='neato') 
        
        graph.write_png("dependency_graph_3.png") 
    
    def draw4(self, topo: List[ParameterNode]): 
        path = os.pathsep + os.getcwd() 
        
        graph = pydot.Dot(graph_type = "digraph", rankdir = "LR", splines = "true") 
        
        # print("op {} abstractparameters {} instanceparameters {}".format(self.op(), len(self.abstractparameters), len(self.instanceparameters))) 
        
        allnodelist = [] 
        for item in self.abstractparameters: 
            if item in topo: 
                if topo.index(item) == len(topo) - 1: 
                    node = pydot.Node(item.name + " (a)", shape = "ellipse", fillcolor = "cyan", color = "none", style = "filled") 
                else: 
                    node = pydot.Node(item.name + " (a)", shape = "ellipse", fillcolor = "lightblue", color = "none", style = "filled") 
            else: 
                node = pydot.Node(item.name + " (a)", shape = "ellipse", fillcolor = "purple", color = "none", style = "filled") 
            graph.add_node(node) 
            allnodelist.append(node) 
        
        for item in self.instanceparameters: 
            if item in topo: 
                if topo.index(item) == len(topo) - 1: 
                    node = pydot.Node(item.name + " (i)", shape = "rectangle", fillcolor = "cyan", color = "none", style = "filled") 
                else: 
                    node = pydot.Node(item.name + " (i)", shape = "rectangle", fillcolor = "lightblue", color = "none", style = "filled") 
            else: 
                node = pydot.Node(item.name + " (i)", shape = "rectangle", fillcolor = "purple", color = "none", style = "filled") 
            graph.add_node(node) 
            allnodelist.append(node) 
        
        node = pydot.Node("RNG", shape = "ellipse") 
        allnodelist.append(node) 
        
        if len(allnodelist) >= 15: 
            radius = 100 
        else: 
            radius = 50 
        
        # for i, node_name in enumerate(nodes): 
        for i, node in enumerate(allnodelist): 
            n = len(allnodelist) - 1 
            angle = 2 * math.pi * i / n
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            # Get the node and set its position
            # node = graph.get_node(node.get_name())[0] 
            node.set('pos', f"{x},{y}!") 
            
            if node.get_name() == "RNG": 
                node.set('pos', f"{0},{0}!") 
        
        for item in self.abstractparameters: 
            # print("item name {} length edgefromlist {}".format(item.name, len(item.edgefromlist))) 
            for item2 in item.edgefromlist: 
                childrenname = item2.name + " (a)" if isinstance(item2, AbstractParameter) else item2.name + " (i)" 
                graph.add_edge(pydot.Edge(childrenname, item.name + " (a)", color = "red")) 
        
        for item in self.instanceparameters: 
            # print("item name {} length edgefromlist {}".format(item.name, len(item.edgefromlist))) 
            for item2 in item.edgefromlist: 
                childrenname = item2.name + " (a)" if isinstance(item2, AbstractParameter) else item2.name + " (i)" 
                graph.add_edge(pydot.Edge(childrenname, item.name + " (i)", color = "black")) 
        
        graph.set_size("200,200") 
        
        # Graph attributes for circular layout
        graph.set_graph_defaults(overlap='false', splines='true', layout='neato') 
        
        graph.write_png("dependency_graph_4.png") 
