# Below are the code used for generating the forward problem 
# It contains our faithful replication of algorithms written in https://arxiv.org/abs/2407.20311 

import random 
import numpy as np 
from StructureGraphThree import Node 
from StructureGraphThree import GraphStructure 

from simple_names_three import hierarchical_categorizations 
from simple_names_three import subcategories 
from simple_names_three import hierarchical_categorizationstwo 
from simple_names_three import subcategoriestwo 
from simple_names_three import hierarchical_categorizationsthree 
from simple_names_three import subcategoriesthree 
from simple_names_three import message 
from simple_names_three import messagetwo 
from simple_names_three import messagethree 

from simple_dummy_namesone import dummyhierarchical_categorizations 
from simple_dummy_namesone import dummysubcategories 
from simple_dummy_namestwo import dummyhierarchical_categorizationstwo 
from simple_dummy_namestwo import dummysubcategoriestwo 
from simple_dummy_namesthree import dummyhierarchical_categorizationsthree 
from simple_dummy_namesthree import dummysubcategoriesthree 

from DependencyGraph import AbstractParameter 
from DependencyGraph import InstanceParameter 
from DependencyGraph import ParameterNode 
from DependencyGraph import AbstractParameterSpecial 

from DependencyGraph import DependencyGraph 
from math import exp 
from typing import Dict, Set, List, Tuple 

from termcolor import colored 
from DependencyGraph import RNG 

# from simple_dummy_names import hierarchical_categorizations, subcategories 
from transformers import AutoTokenizer 
from simple_names_three import lengthmessage, lengthmessagetwo, lengthmessagethree 

class StrangeError(Exception):
    """A custom exception for strange errors."""
    pass 

def whilecondition(G_structure, w1): 
    output = True 
    for item in G_structure.l: 
        output = output and (item == w1) 
    return output 

def is_topo(topo: List[ParameterNode]): 
    """ Check if the list is topological 
    
    Args: 
        topo (List[ParameterNode]) 
    
    Returns: 
        bool 
    """ 
    for i in range(len(topo)): 
        for j in range(i + 1, len(topo)): 
            if topo[i] in topo[j].edgetolist: 
                # print("topo[j] in topo[i] edgefromlist? {}".format(topo[j] in topo[i].edgefromlist)) 
                print(colored("{} --> {}".format(topo[j], topo[i]), "red")) 
                return False 
        for j in range(i): 
            if topo[i] in topo[j].edgefromlist: 
                # print("topo[j] in topo[i] edgetolist? {}".format(topo[j] in topo[i].edgetolist)) 
                # print("topo[i] edgetolist {}".format(topo[i].edgetolist)) 
                print(colored("{} --> {}".format(topo[i], topo[j]), "red")) 
                return False 
    return True 
    
# Step 1. Generate Structure Graph 
def drawStructure(e, d, w0, w1, template = None): 
    """ Structure Graph 

    Args:
        e: number of edges 
        d: number of layers 
        w0: minimum number of nodes per layer 
        w1: maximum number of nodes per layer 
    
    Returns: 
        G_structure: a structure graph 
    """ 
    goingthree = d == 3 
    d = 2 # number of layers is fixed first to 2 
    G_structure = GraphStructure(d, w0, w1) 
    # l ← (w0,w0,...,w0) ∈ Zd 
    assert len(G_structure.l) == d 
    for i in range(d): 
        assert G_structure.l[i] == w0 
    # p ← uniform random from (0, 1) 
    p = random.random() 
    # while l ̸= (w1,w1,...,w1) do 
    while not whilecondition(G_structure, w1): 
        e_minus = np.sum(G_structure.l[1 : ]) 
        e_plus = 0 
        for i in range(d - 1): 
            e_plus += G_structure.l[i] * G_structure.l[i + 1] 
        # if e+ <ethen
        #     randomly select i ∈ [d] such that li < w1, and increase it li ← li + 1. 
        if e_plus < e: 
            i_set = [] 
            for i in range(d): 
                if G_structure.l[i] < w1: 
                    i_set.append(i) 
            if len(i_set) != 0: 
                j = np.random.choice(i_set, size = 1, p = [1/len(i_set) for _ in range(len(i_set))])[0] 
                G_structure.l[j] += 1 
            else: 
                raise ValueError("i_set is empty.") 
        # else if e− = e then 
        elif e_minus == e: 
            # break 
            break 
        # else if randomly choose a number in (0, 1) and it is less than p then
        #      randomly select i ∈ [d] such that li < w1, and increase it li ← li + 1. 
        elif random.random() < p: 
            i_set = [] 
            for i in range(d): 
                if G_structure.l[i] < w1: 
                    i_set.append(i) 
            if len(i_set) != 0: 
                j = np.random.choice(i_set, size = 1, p = [1/len(i_set) for _ in range(len(i_set))])[0] 
                G_structure.l[j] += 1 
            else: 
                raise ValueError("i_set is empty.") 
        # 11: else 
        # 12: break 
        else: 
            break 
    # Construct Gs with exactly li items on layer i ∈ [d]. 
    G_structure.fill_ite() 
    # foreachitemaineachlayeri≥2do
    #     randomly select an item b in layer i − 1 and connect (a, b) in Gs. ⋄ this creates e− edges 
    G_structure.construct_first() 
    # while number of edges < e do
    #     randomly select two items a,b from adjacent layers to create an edge in Gs. 
    G_structure.construct_second(e) 
    
    if goingthree: # three out of four we add the attribute to the structure 
        print(colored("three", "cyan")) 
        G_structure.d = d + 1 
        d = d + 1 
        G_structure.layers.append([]) 
        G_structure.l.append(1) 
        # G_structure.layers_names.append("Total Newborn Children") 
        
        newnode = Node(d * G_structure.w1, d - 1) 
        G_structure.nodes.add(newnode) 
        G_structure.layers[d - 1].append(newnode) 
        G_structure.construct_extra(e) 
    else: 
        print(colored("two", "yellow")) 
    
    if template == "crazy_zootopia": 
        G_structure.attachEnglish(hierarchical_categorizations = hierarchical_categorizations, subcategories = subcategories) 
    elif template == "teachers_in_school": 
        G_structure.attachEnglish(hierarchical_categorizations = hierarchical_categorizationstwo, subcategories = subcategoriestwo) 
    elif template == "movie_festival_awards": 
        G_structure.attachEnglish(hierarchical_categorizations = hierarchical_categorizationsthree, subcategories = subcategoriesthree) 
    
    # return Gs and attach English to it. 
    return G_structure 
def combine_unique(list1, list2): 
    output = [] 
    return list1 + [item for item in list2 if item not in list1] 

def recursivelyaddpara(G_structure, parentnode, parentparameter, end_layer, parentlevelinstanceparameter = None): 
    # helper 
    outlist = {} 
    outtwo = [] 
    abvalid = False 
    if end_layer - parentnode.leveli == 0: 
        return False, {}, [] 
    elif end_layer - parentnode.leveli == 1: 
        # print(colored("parentparameter {} parentlevelinstanceparameter {}".format(parentparameter.name, parentlevelinstanceparameter), "green")) 
        for adjacentnode in parentnode.adjacent: 
            if adjacentnode.leveli == end_layer: 
                abvalid = abvalid or True 
                if parentlevelinstanceparameter == None: 
                    # name = "{}'s {}".format(parentnode.name, adjacentnode.name) 
                    name = "{} in {}".format(adjacentnode.name, parentnode.name) 
                else: 
                    # name = "{} of {}".format(adjacentnode.name, parentlevelinstanceparameter.name) 
                    name = "{} per {}".format(adjacentnode.name, parentlevelinstanceparameter.name) 
                newinstanceparameter = InstanceParameter(name) 
                if newinstanceparameter not in parentparameter.edgefromlist: # in the parent parameter case, we might have already added 
                    parentparameter.edgefromlist.append(newinstanceparameter) 
                # print("parent parameter {} length edgefromlist {}".format(parentparameter.name, len(parentparameter.edgefromlist))) 
                if parentparameter not in newinstanceparameter.edgetolist: 
                    newinstanceparameter.edgetolist.append(parentparameter) 
                outtwo.append(newinstanceparameter) 
        return abvalid, outlist, outtwo 
    else: 
        for adjacentnode in parentnode.adjacent: 
            if adjacentnode.leveli > parentnode.leveli: 
                
                # add the abstract parameter in 
                # the following line is a hack for naming the abstract parameter spanning the middle layer and end concept 
                # name = "{}'s {}".format(parentnode.name, adjacentnode.name) 
                name = "{} in {}".format(adjacentnode.name, parentnode.name) 
                newinstanceparameter = InstanceParameter(name) 
                if G_structure.d == 2: 
                    # name = "{}'s {}".format(adjacentnode.name, G_structure.layer_names[end_layer]) 
                    name = "{} in {}".format(G_structure.layer_names[end_layer], adjacentnode.name) 
                    newabstractparameter = AbstractParameter(name) 
                else: 
                    # name = "{} of {}".format(G_structure.layers[end_layer][0].name, newinstanceparameter.name) 
                    name = "{} per {}".format(G_structure.layers[end_layer][0].name, newinstanceparameter.name) 
                    newabstractparameter = AbstractParameterSpecial(name, newinstanceparameter) 
                abvalidtwo, outlistt, outtwoot = recursivelyaddpara(G_structure, adjacentnode, newabstractparameter, end_layer, newinstanceparameter) 
                abvalid = abvalid or abvalidtwo 
                if abvalidtwo: 
                    # add the instance parameter in 
                    if newinstanceparameter not in parentparameter.edgefromlist: 
                        parentparameter.edgefromlist.append(newinstanceparameter) 
                    if parentparameter not in newinstanceparameter.edgetolist: 
                        newinstanceparameter.edgetolist.append(parentparameter) 
                    outtwo.append(newinstanceparameter) 
                    
                    if newabstractparameter not in parentparameter.edgefromlist: 
                        parentparameter.edgefromlist.append(newabstractparameter) 
                    if parentparameter not in newabstractparameter.edgetolist: 
                        newabstractparameter.edgetolist.append(parentparameter) 
                    
                    if (end_layer - adjacentnode.leveli) not in outlist.keys(): 
                        outlist[end_layer - adjacentnode.leveli] = [newabstractparameter] 
                    else: 
                        outlist[end_layer - adjacentnode.leveli] += [newabstractparameter] 
                    if len(newabstractparameter.edgefromlist) == 0: 
                        outlist[end_layer - adjacentnode.leveli].remove(newabstractparameter) 
                        continue 
                    for key in outlistt.keys(): 
                        if key in outlist.keys(): 
                            # checking repetition and merge same parameter edges 
                            for parameter in outlistt[key]: # merging 
                                if parameter in outlist[key]: 
                                    parameter2 = outlist[key][outlist[key].index(parameter)] 
                                    # parameter2.edgetolist += parameter.edgetolist 
                                    parameter2.edgetolist = combine_unique(parameter2.edgetolist, parameter.edgetolist) 
                                    # parameter2.edgefromlist += parameter.edgefromlist 
                                    parameter2.edgefromlist = combine_unique(parameter2.edgefromlist, parameter.edgefromlist) 
                                    found = True 
                                else: 
                                    outlist[key].append(parameter) 
                        else: 
                            outlist[key] = outlistt[key] # inserting 
                    for parameter in outtwoot: 
                        found = False 
                        for parameter2 in outtwo: 
                            if parameter == parameter2: 
                                # parameter2.edgetolist += parameter.edgetolist 
                                parameter2.edgetolist = combine_unique(parameter2.edgetolist, parameter.edgetolist) 
                                # parameter2.edgefromlist += parameter.edgefromlist 
                                parameter2.edgefromlist = combine_unique(parameter2.edgefromlist, parameter.edgefromlist) 
                                assert len(parameter2.edgefromlist) == 0 
                                found = True 
                                break 
                        if not found: 
                            outtwo.append(parameter) 
            else: 
                continue 
    if abvalid: 
        return abvalid, outlist, outtwo 
    else: 
        return False, {}, [] 
            
def all_abstract_parameters(G_structure): 
    """ Enumerate all abstract parameters in structure graph 
    
    Args: 
        G_structure (GraphStructure) 
    
    Returns: 
        outlist (Dict[int, List[AbstractParameter]]): a dictionary of abstract parameters 
        outtwo (List[InstanceParameter]): a list of instance parameters 
    """ 
    outlist = {} 
    outtwo = [] 
    for lengthleap in range(G_structure.d - 1, 0, -1): 
        for node in G_structure.layers[0]: 
            # NOTE: the following line checks the connectivity of the graph, it has to be connected to proceed 
            assert len(node.adjacent) != 0, "node {} has no adjacent nodes.".format(node.name) 
            # name = "{}'s {}".format(node.name, G_structure.layer_names[lengthleap]) 
            name = "{} in {}".format(G_structure.layer_names[lengthleap], node.name) 
            newabstractparameter = AbstractParameter(name) 
            if (lengthleap) not in outlist.keys(): 
                # outlist[G_structure.d - 1] = [newabstractparameter] 
                outlist[lengthleap] = [newabstractparameter] 
            else: 
                # outlist[G_structure.d - 1] += [newabstractparameter] 
                outlist[lengthleap] += [newabstractparameter] 
            # abvalid, outlistt, outtwoot = recursivelyaddpara(G_structure, node, newabstractparameter, G_structure.d - 1) 
            abvalid, outlistt, outtwoot = recursivelyaddpara(G_structure, node, newabstractparameter, lengthleap) 
            if abvalid: 
                if len(newabstractparameter.edgefromlist) == 0: 
                    # outlist[G_structure.d - 1].remove(newabstractparameter) 
                    outlist[lengthleap].remove(newabstractparameter) 
                    continue 
                for key in outlistt.keys(): 
                    if key in outlist.keys(): 
                        for parameter in outlistt[key]: # merging 
                            found = False 
                            if parameter in outlist[key]: 
                                parameter2 = outlist[key][outlist[key].index(parameter)] 
                                # parameter2.edgetolist += parameter.edgetolist 
                                parameter2.edgetolist = combine_unique(parameter2.edgetolist, parameter.edgetolist) 
                                parameter.edgetolist = parameter2.edgetolist 
                                # parameter2.edgefromlist += parameter.edgefromlist 
                                parameter2.edgefromlist = combine_unique(parameter2.edgefromlist, parameter.edgefromlist) 
                                parameter.edgefromlist = parameter2.edgefromlist 
                                found = True 
                                break 
                            if not found: 
                                outlist[key].append(parameter) 
                    else: 
                        outlist[key] = outlistt[key] # inserting 
                for parameter in outtwoot: 
                    if parameter in outtwo: 
                        parameter2 = outtwo[outtwo.index(parameter)] 
                        # parameter2.edgetolist += parameter.edgetolist 
                        parameter2.edgetolist = combine_unique(parameter2.edgetolist, parameter.edgetolist) 
                        parameter.edgetolist = parameter2.edgetolist 
                        # parameter2.edgefromlist += parameter.edgefromlist 
                        parameter2.edgefromlist = combine_unique(parameter2.edgefromlist, parameter.edgefromlist) 
                        parameter.edgefromlist = parameter2.edgefromlist 
                        assert len(parameter2.edgefromlist) == 0 
                    else: 
                        outtwo.append(parameter) 
    for key in outlist.keys(): 
        # NOTE: here we insert a checking system that removes unreasonable abstract parameters 
        for item in outlist[key]: 
            if len(item.edgefromlist) == 0: # NOTE: the abstract parameter isn't reasonable 
                outlist[key].remove(item) 
    
    return outlist, outtwo 
            
def drawNecessary1(G_structure, n, m): 
    """ Dependency Graph of abstract instances 

    Args:
        G_structure (StructureGraph) 
        n (int) 
        m (int) 
    
    Returns: 
        G_necessary (DependencyGraph) 
    """ 
    abstract_parameter_list, instance_parameter_list = all_abstract_parameters(G_structure) 
    # Gnece1 ← empty graph 
    G_necessary = DependencyGraph() 
    G_two = DependencyGraph() 
    updated = True 
    count = 0 
    while updated == True: 
        # updated ← false 
        updated = False 
        # for i ← d − 1, . . . , 1 do 
        for difficultylevel in range(G_structure.d - 1, 0, -1): 
            # if ∃ abstract parameter of difficulty level i in Gs that is not yet in Gnece1 then 
            if len(abstract_parameter_list[difficultylevel]) == 0: 
                continue 
            selectionidx = random.randint(0, len(abstract_parameter_list[difficultylevel]) - 1) 
            selected_parameter = abstract_parameter_list[difficultylevel][selectionidx] 
            if len(selected_parameter.edgefromlist) == 0: 
                abstract_parameter_list[difficultylevel].remove(selected_parameter) 
                continue 
            abstract_parameter_list[difficultylevel].remove(selected_parameter) 
            #     randomly pick one such abstract parameter a of difficulty level i
            #     G′ ← Gnece1+a and all instance/abstract parameters a may (recursively) depend on 
            success = G_two.add_abstractparameter(selected_parameter) 
            
            # if op(G′) ≤ n then 
            if G_two.op() <= n: 
            #     Gnece1 ← G′; updated ← true; break 
                G_necessary.add_abstractparameter(selected_parameter) 
                updated = True 
                break 
            else: 
                G_two = G_necessary.copy() 
    
    # Gnece2 ← Gnece1 ⋄ op(Gnece1) ≤ n and all instance parameters in Gnece1 have in-degree 0
    # fori←1,2,...,m−op(Gnece1)do 
    for i in range(1, m - G_necessary.op()): 
        leftoverinstanceparameters = [] 
        for instanceparameter in instance_parameter_list: 
            if instanceparameter not in G_necessary.instanceparameters: 
                leftoverinstanceparameters.append(instanceparameter) 
        
        # if there’s leftover instance parameter in Gs not yet in Gnece2, add a random one to Gnece2 
        if len(leftoverinstanceparameters) != 0: 
            selectionidx = random.randint(0, len(leftoverinstanceparameters) - 1) 
            selected_instanceparameter = leftoverinstanceparameters[selectionidx] 
            G_necessary.instanceparameters.append(selected_instanceparameter) 
            leftoverinstanceparameters.remove(selected_instanceparameter) 
    
    # return Gnece2 
    return G_necessary, abstract_parameter_list, instance_parameter_list 

def graph_checker(G_necessary): 
    for item in G_necessary.abstractparameters: 
        for item2 in item.edgefromlist: 
            assert item in item2.edgetolist 
        for item2 in item.edgetolist: 
            assert item in item2.edgefromlist 
    for item in G_necessary.instanceparameters: 
        for item2 in item.edgefromlist: 
            assert item in item2.edgetolist 
        for item2 in item.edgetolist: 
            assert item in item2.edgefromlist 

def intersection(nextone, nexttwo): 
    output = [] 
    for item in nextone: 
        if item in nexttwo: 
            output.append(item) 
    return output 

def samplingwithbias(nexttwo, nextone): 
    def weight(a, nextone): 
        g = random.gauss(0, 1) 
        return ((1 if isinstance(a, AbstractParameter) else 0) + (1 if a in nextone else 0)) * abs(g) 
    weightlist = [exp(weight(item, nextone)) for item in nexttwo] 
    
    return random.choices(nexttwo, weights = weightlist) 

def drawNecessary2(G_necessary): 
    """ Dependency Graph of abstract instances 

    Args:
        G_necessary (DependencyGraph) 
    """ 
    
    # Gnece3 ← Gnece2; Topo ← []. 
    topo: List[ParameterNode] = [] 
    # while true do 
    # for item in G_necessary.abstractparameters: 
        # print(colored("item {} edgefromlist {} edgetolist {}".format(item.name, item.edgefromlist, item.edgetolist), "green")) 
    
    while True: 
        nextone, nexttwo, output = G_necessary.next_one_two(topo) 
        # if Topo = [] then 
        if len(topo) == 0: 
            parameter0 = nexttwo[random.randint(0, len(nexttwo) - 1)] 
        else: 
            onetwointersection = intersection(nextone, nexttwo) 
            # param0 ← random parameter in Next1Gnece3 (Topo) ∩ Next2Gnece3 (Topo); 
            parameter0 = onetwointersection[random.randint(0, len(onetwointersection) - 1)] 
        topo.insert(0, parameter0) 
        
        nextone, nexttwo, output = G_necessary.next_one_two(topo) 
        # if Gnece3 \ Topo = ∅ then break 
        if len(output) == 0: 
            break 
        
        # if Next1Gnece3 (Topo) ∩ Next2Gnece3 (Topo) = ∅ then 
        if len(intersection(nextone, nexttwo)) == 0: 
            # If param0 is abstract then return failure 
            if isinstance(parameter0, AbstractParameter): 
                if isinstance(parameter0, AbstractParameterSpecial): 
                    print(colored("parameter0.isonamicparameter in {}".format((parameter0.isonamicinstanceparameter) in topo), "red")) 
                raise ValueError("param0 is abstract.") 
            # param1 ← a “random” parameter in Next2Gnece3 (Topo). 
            parameter1 = samplingwithbias(nexttwo, nextone)[0] 
            # add edge param → param to Gnece3. 
            G_necessary.add_edge(from_node = parameter1, to_node = parameter0) 
        # else if param0 is instance parameter then 
        elif isinstance(parameter0, InstanceParameter): 
            # if a probability event p0 occurs for p0 uniform chosen in (0, 1) then 
            p0 = random.random() 
            if random.random() < p0: 
                # param ← a “random” parameter in Gnece3 \ Topo. 
                parameter1 = samplingwithbias(output, nextone)[0] 
                # add edge param → param to Gnece3. 
                G_necessary.add_edge(from_node = parameter1, to_node = parameter0) 
    
    for item in G_necessary.instanceparameters: 
        assert len(item.edgefromlist) <= 1 
    
    return G_necessary, topo 
def maxopTopo(item, topo): 
    if item not in topo: 
        raise ValueError("item should be in topo.") 
    
    i = topo.index(item) 
    return min(3, max(1, i)) 

def summation_c_op(G_necessary): 
    sum_c_op = 0 
    for item in G_necessary.abstractparameters: 
        sum_c_op += item.c_op 
    for item in G_necessary.instanceparameters: 
        sum_c_op += item.c_op 
    return sum_c_op 

def drawNecessary3(G_necessary, topo, s): 
    """ Add edges to the graph 

    Args:
        G_necessary (DependencyGraph) 
        topo (List[ParameterNode]) 
        s (int) 
    """ 
    
    # cur op(a) ← op nece3 (a) for every parameter a ∈ Gnece3. 
    for item in G_necessary.abstractparameters: 
        item.c_op = len(item.edgefromlist) 
    for item in G_necessary.instanceparameters: 
        item.c_op = len(item.edgefromlist) 
    # max opTopo(a) = the maximum number of operations an instance parameter a can require. 
    for item in G_necessary.instanceparameters: 
        item.m_op = maxopTopo(item, topo) 
    # while Pa∈Gnece3 cur op(a) < s do 
    while summation_c_op(G_necessary) < s: 
        # randomly select an instance parameter a ∈ Gnece3 with cur op(a) < max op (a); 
        conditionlist = [] 
        for item in G_necessary.instanceparameters: 
            if item.c_op < item.m_op: 
                conditionlist.append(item) 
        if len(conditionlist) == 0: 
            raise ValueError("Cannot find any instance parameter that has cop < mop.") 
        selected_parameter = conditionlist[random.randint(0, len(conditionlist) - 1)] 
        # If a is found then cur op(a) ← cur op(a) + 1 else return failure. 
        selected_parameter.c_op += 1 
    # Gnece ← Gnece3 + vertex RNG. 
    # for each instance parameter a in Gnece3 do 
    for item in G_necessary.instanceparameters: 
        # pool ← RNG + all parameters in front of a in Topo. 
        pool: List[ParameterNode] = [] 
        pool.append(G_necessary.rng) 
        topoidx = topo.index(item) 
        for i in range(topoidx): 
            pool.append(topo[i]) 
        # if cur op(a) = 1 then 
        if item.c_op == 1: 
            # depnum←1or2eachw.p. 0.5; 
            dep_num = 1 if random.random() < 0.5 else 2 
        # else 
        else: 
            # dep num ← cur op(a) + 1 
            dep_num = item.c_op + 1 
        # dep num ← min{|pool|, dep num} 
        dep_num = min(len(pool), dep_num) 
        # if ∃(b→a)∈Gnece3 forsomeb∈poolthen 
        count = 0 
        for item2 in pool: 
            if item2 in item.edgefromlist: # NOTE: item <- item2 
                count += 1 
        assert count <= 1 
        if count == 1: 
            # pool ← pool \ {b} and dep num ← dep num − 1 
            pool.remove(item2) 
            dep_num -= 1 
        # if dep num = |pool| then 
        if dep_num == len(pool): 
            # addb→atoGnece forallb∈pool; 
            for item2 in pool: 
                G_necessary.add_edge(from_node = item2, to_node = item) 
        # else 
        else: 
            # with probability 0.5, add RNG → a to Gnece and dep num ← dep num − 1 
            if random.random() < 0.5: 
                G_necessary.add_edge(from_node = G_necessary.rng, to_node = item) 
                dep_num -= 1 
            # pool ← pool \ {RNG} 
            pool.remove(G_necessary.rng) 
            # add b → a to Gnece for dep num randomly select elements b in pool.
            for i in range(dep_num): 
                selected_parameter = pool[random.randint(0, len(pool) - 1)] 
                G_necessary.add_edge(from_node = selected_parameter, to_node = item) 
                pool.remove(selected_parameter) 
    # return Gnece 
    return G_necessary 

def drawUnnecessary(G_necessary, all_abstract_parameters, all_instance_parameters, G_structure): 
    """ Draw Unnecessary Graph 
    
    Args: 
        G_necessary (DependencyGraph) 
        all_abstract_parameters (Dict[int, List[AbstractParameter]]) 
        all_instance_parameters (List[InstanceParameter]) 
        G_structure (GraphStructure) 
    """ 
    
    # processing a bit 
    all_abstract_parameterslist = [] 
    for key in all_abstract_parameters.keys(): 
        all_abstract_parameterslist += all_abstract_parameters[key] 
    
    for item in G_necessary.instanceparameters: 
        if item in all_instance_parameters: 
            all_instance_parameters.remove(item) 
    
    # 1: IndList ← ∅; 
    IndList: List[InstanceParameter] = [] 
    # 2: while ∃ instance parameter in Gs not yet in Gd do 
    
    while len(all_instance_parameters) != 0: 
        # 3: K ← all params in Gd + all abstract params computable using parameters in Gd; 
        allparamdependency = [] 
        for item in G_necessary.abstractparameters: 
            allparamdependency.append(item) 
        for item in G_necessary.instanceparameters: 
            allparamdependency.append(item) 
        abstractdependentfrom = [] 
        for item in all_abstract_parameterslist: # all abstract paraameters in structure 
            for item2 in item.edgefromlist: 
                if isinstance(item2, AbstractParameter): 
                    if item2 in G_necessary.abstractparameters: 
                        abstractdependentfrom.append(item) 
                        break 
                
                if isinstance(item2, InstanceParameter): 
                    if item2 in G_necessary.instanceparameters: 
                        abstractdependentfrom.append(item) 
                        break 
        K: List[ParameterNode] = [] 
        K = G_necessary.abstractparameters + G_necessary.instanceparameters + abstractdependentfrom 
        # 4: randomly select an instance parameter a in Gs not yet in Gd; and add a to Gd; 
        selected_parameter = all_instance_parameters[random.randint(0, len(all_instance_parameters) - 1)] 
        # assert (selected_parameter not in G_necessary.instanceparameters) 
        if selected_parameter not in G_necessary.instanceparameters: 
            G_necessary.instanceparameters.append(selected_parameter) 
        else: 
            # raise ValueError("selected parameter is already in G_necessary.") 
            continue 
        all_instance_parameters.remove(selected_parameter) 
        pool: List[ParameterNode] = [] 
        
        # 5: if with half probability then 
        if random.random() < 0.5: 
            # 6: pool ← IndList ∪ {RNG}; IndList ← IndList ∪ {a}; 
            pool = IndList + [G_necessary.rng] 
            IndList.append(selected_parameter) 
        # 7: else 
        else: 
            # 8: pool ← K ∪ {RNG}; 
            pool = K + [G_necessary.rng] 
        # 9: depnum←1 
        dep = 1 
        # 10: while dep num < min{4, |pool|} do 
        while dep < min(4, len(pool)): 
            # 11: with 0.5 probability, dep num ← dep num + 1; otherwise break 
            if random.random() < 0.5: 
                dep += 1 
            else: 
                break 
        selected: List[ParameterNode] = [] 
        # 12: if dep num = |pool| then 
        if dep == len(pool): 
            # 13: selected ← pool 
            selected = pool 
        # 14: else 
        else: 
            # 15: selected ← {}
            # 16: with probability 0.5, add selected = {RNG} and dep num ← dep num − 1 
            if random.random() < 0.5: 
                selected.append(G_necessary.rng) 
                dep -= 1 
            # 17: pool ← pool \ {RNG} 
            pool.remove(G_necessary.rng) 
            # 18: selected ← selected ∪ dep num random elements from pool 
            selected += random.sample(pool, dep) 
        
        # 19: for each b ∈ selected do 
        for item in selected: 
            # 20: If b ̸∈ Gd then recursively add b and its dependencies to Gd; 
            inside = False 
            
            if isinstance(item, AbstractParameter): 
                if item not in G_necessary.abstractparameters: 
                    G_necessary.add_abstractparameter(item) 
            
            if isinstance(item, InstanceParameter): 
                if item not in G_necessary.instanceparameters: 
                    assert len(item.edgefromlist) == 0 
                    G_necessary.instanceparameters.append(item) 
            
            # 21: Add b → a to Gd. 
            G_necessary.add_edge(from_node = item, to_node = selected_parameter) 
            
            if item in all_instance_parameters: 
                all_instance_parameters.remove(item) 
        
        for item in G_necessary.instanceparameters: 
            if item in all_instance_parameters: 
                all_instance_parameters.remove(item) 
    
    # 22: return Gd 
    return G_necessary 

def preparenoiseforreverse(G_necessary, topo): 
    for item in G_necessary.instanceparameters: 
        if item not in topo: 
            allowitem = True 
            itemstoremove = [] 
            for item2 in item.edgefromlist: 
                if item2 in topo: 
                    if allowitem: 
                        allowitem = False 
                    else: 
                        item2.edgetolist.remove(item) 
                        itemstoremove.append(item2) 
                elif isinstance(item2, InstanceParameter): 
                    item2.edgetolist.remove(item) 
                    itemstoremove.append(item2) 
                else: 
                    continue 
            for item2 in itemstoremove: 
                item.edgefromlist.remove(item2) 
        else: 
            continue 
    
def num_op(topo): 
    """ Number of operations in the solution text 
    
    Args: 
        topo (List[ParameterNode]): 
    
    Returns: 
        int: number of operations 
    """ 
    
    countop = 0 
    for item in topo: 
        if not isinstance(item, AbstractParameterSpecial): 
            countop += len(item.edgefromlist) 
        else: 
            assert len(item.edgefromlist) == 1 
    return countop 

def standard_name(name, sentence_begins = False): 
    preff = "The " if sentence_begins else "the " 
    if name.split(" ")[0] == "average" or name.split(" ")[0] == "total": 
        return preff + name 
    else: 
        return preff + "number of " + name 

def generate_sentence(G_necessary, item, number_range = 23): 
    # text = "The {} equals".format(item.name) 
    text = "{} equals".format(standard_name(item.name, sentence_begins = True)) 
    # pool←{b∈Gd:∃(b→a)∈Gd}. 
    pool = [] 
    for item2 in item.edgefromlist: 
        pool.append(item2) 
    # if RNG ∈ pool then 
    if G_necessary.rng in pool: 
        # str ← str + “ [random int between 0 and 22]”; and pool ← pool \ {RNG} 
        plusortimes = random.random() < 0.5 
        if plusortimes: 
            randomnumbergenerate = random.randint(1, number_range - 1) 
        else: 
            randomnumbergenerate = random.randint(2, number_range - 1) 
        text += " {}".format(randomnumbergenerate) 
        item.value = randomnumbergenerate 
        pool.remove(G_necessary.rng) 
        # If |pool| > 0, str ← str + “ more than” or “ times” each with probability 0.5. 
        if len(pool) > 0: 
            item.rngv = randomnumbergenerate 
            if plusortimes: 
                # text += " more than" 
                text += " plus" 
                item.rngnot = "+" 
            else: 
                text += " times" 
                item.rngnot = "*" 
    # if |pool| = 1 then 
    if len(pool) == 1: 
        # str ← str + “ [name of b]” for pool = {b}. 
        text += " {}".format(standard_name(pool[0].name)) 
    # else if |pool| = |{b, c}| = 2 then 
    elif len(pool) == 2: 
        # str ← str + “ the sum of [b] and [c]” or “ the difference of [b] and [c]” each w.p. 0.5. 
        if random.random() < 0.5: 
            text += " the sum of {} and {}".format(standard_name(pool[0].name), standard_name(pool[1].name)) 
            item.notation = "+" 
        else: 
            text += " the difference between {} and {}".format(standard_name(pool[0].name), standard_name(pool[1].name)) 
            item.notation = "-" 
    # else 
    elif len(pool) > 2: 
        # str ← str + “ the sum of .., .., and ..” with a random order of all elements from pool. 
        text += " the sum of " 
        for item in pool[: -1]: 
            text += "{}, ".format(standard_name(item.name)) 
        text += "and {}".format(standard_name(pool[-1].name)) 
    else: 
        if G_necessary.rng not in item.edgefromlist: 
            item.value = random.randint(1, number_range - 1) 
            text += " {}".format(item.value) 
    return text 

def generate_description_structure(G_structure): 
    text = "" 
    for l in range(G_structure.d): 
        text += "Layer {} has category {}".format(l, G_structure.layer_names[l]) 
        text += "; " 
        text += "On layer, the specific types of {} are ".format(G_structure.layer_names[l]) 
        text += ", ".join([item.name for item in G_structure.layers[l]]) 
        text += ";\n" 
    
    for l in range(G_structure.d - 1): 
        text += "For types on layer {}, we define the connections as follows: ".format(l) 
        for item in G_structure.layers[l]: 
            text += "type {} has the following types in the next layer {}".format(item, G_structure.layer_names[l + 1]) 
            sub_types = [] 
            for item2 in item.adjacent: 
                if item2.leveli > l: 
                    sub_types.append(item2.name) 
            text += " " + ", ".join(sub_types) 
            text += "; " 
        text += "\n" 
    
    return text 
    
def problem_in_text(G_necessary, number_range = 23): 
    sentences = [] 
    
    for instanceparameter in G_necessary.instanceparameters: 
        newsentence = generate_sentence(G_necessary, instanceparameter, number_range = number_range) 
        
        sentences.append(newsentence) 
    '''
    random.shuffle(sentences) 
    
    text_output = "" 
    for sentence in sentences: 
        text_output += sentence + ". " 
    
    return text_output 
    ''' 
    return sentences 

def question_in_text(topo): 
    # returning the question in text 
    itemname = topo[-1].name 
    if "average" in itemname or "total" in itemname: 
        # text = "What is the {} of {}?".format(object_subcategory, category) 
        text = "What is the {}?".format(itemname) 
    else: 
        # category = itemname.split("'s ")[0] 
        category = itemname.split(" in ")[-1] 
        # object_subcategory = itemname.split("'s ")[1] 
        object_subcategory = itemname.split(" in ")[0] 
        text = "How many {} does {} have?".format(object_subcategory, category) 
    return text 

def ispairingvalie(item1: InstanceParameter, item2: AbstractParameter, parent_name: str) -> bool: 
    # parent_name = parent_name.split("'s ") 
    parent_name = parent_name.split(" in ") 
    # twoname = item2.name.split(" of ")[-1] 
    twoname = item2.name.split(" per ")[-1] 
    # if onename[1] == twoname[0] and parent_name[1] == twoname[1]: 
    # if twoname == item1.name and parent_name[0] == item1.name.split("'s ")[0]: 
    if twoname == item1.name and parent_name[1] == item1.name.split(" in ")[-1]: 
        return True 
    else: 
        # print("item1 {} item2 {} parent_name {}".format(item1.name, item2.name, parent_name)) 
        # print("{},{}".format(onename[1], len(onename[1]))) 
        # print("{},{}".format(twoname[0], len(twoname[0]))) 
        # print("{},{}".format(parent_name[1], len(parent_name[1]))) 
        # print("{},{}".format(twoname[1], len(twoname[1]))) 
        return False 

def solution_in_text(G_necessary, topo, mod: int = -1): 
    # the function is called early when topo is first made 
    # generate the solution 
    ch = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"] 
    ch += ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"] 
    solution_text = "" 
    
    patches = {} 
    patches_variable = {} 
    
    for item in topo: 
        if isinstance(item, AbstractParameterSpecial): 
            assert len(item.edgefromlist) == 1 
            assert item.name == item.edgefromlist[0].name 
            item.value = item.edgefromlist[0].value 
            item.variable = item.edgefromlist[0].variable 
            continue 
        
        if isinstance(item, InstanceParameter): 
            pool = [] 
            rngin = False 
            for item2 in item.edgefromlist: 
                if isinstance(item2, RNG): 
                    continue 
                pool.append(item2) 
            
            rngin = (G_necessary.rng in item.edgefromlist) 
            
            list_factor = [] 
            list_notation = [] 
            text1 = "" 
            if len(pool) == 0: # initialized 
                new_char = random.choice(ch)[0] 
                ch.remove(new_char) 
                item.variable = new_char 
                text1 += "so {} = {}".format(new_char, item.value) 
            elif len(pool) == 1: # assignment 
                new_char = random.choice(ch)[0] 
                ch.remove(new_char) 
                item.variable = new_char 
                item.value = pool[0].value 
                if not rngin: 
                    text1 += "so {} = {} = {}".format(new_char, pool[0].variable, item.value) 
                else: 
                    text1 += "{} = {} = {}".format(new_char, pool[0].variable, item.value) 
            elif len(pool) == 2: # two factors adding and subtracting 
                new_char = random.choice(ch)[0] 
                ch.remove(new_char) 
                item.variable = new_char 
                for item2 in pool: 
                    if item2.value == None: 
                        if item2.name not in patches.keys(): 
                            item2.value = patches[item2.name] 
                            item2.variable = patches_variable[item2.name] 
                        else: 
                            raise ValueError("item2.value is None but item2.name is not in patches.") 
                if not rngin: 
                    text1 += "so {} = ".format(new_char) 
                else: 
                    text1 += "{} = ".format(new_char) 
                if item.notation == "+": 
                    text1 += "{} + {} = {} + {}".format(pool[0].variable, pool[1].variable, pool[0].value, pool[1].value) 
                    item.value = pool[0].value + pool[1].value 
                    if mod > 0: 
                        item.value = item.value % mod 
                    text1 += " = {}".format(item.value) 
                elif item.notation == "-": 
                    text1 += "{} - {} = {} - {}".format(pool[0].variable, pool[1].variable, pool[0].value, pool[1].value) 
                    item.value = pool[0].value - pool[1].value 
                    if mod > 0: 
                        item.value = item.value % mod 
                    text1 += " = {}".format(item.value) 
                else: 
                    raise ValueError("item.notation is not recognized.") 
            elif len(pool) > 2: 
                for item2 in pool: 
                    if item2.value == None: 
                        if item2.name not in patches.keys(): 
                            item2.value = patches[item2.name] 
                            item2.variable = patches_variable[item2.name] 
                        else: 
                            raise ValueError("item2.value is None. ") 
                for idx in range(1, len(pool)): # running multiple factors 
                    if idx == 1: 
                        new_char = random.choice(ch)[0] 
                        ch.remove(new_char) 
                        text1 += "{} = {} + {}".format(new_char, pool[idx - 1].variable, pool[idx].variable) 
                        item.variable = new_char 
                        item.value = pool[idx - 1].value + pool[idx].value 
                        if mod > 0: 
                            item.value = item.value % mod 
                        text1 += " = {} + {} = {}; ".format(pool[idx - 1].value, pool[idx].value, item.value) 
                    else: 
                        new_char = random.choice(ch)[0] 
                        ch.remove(new_char) 
                        if idx == len(pool) - 1 and not rngin: 
                            text1 += "so " 
                        text1 += "{} = {} + {}".format(new_char, item.variable, pool[idx].variable) 
                        item.variable = new_char 
                        if idx == len(pool) - 1: 
                            if mod > 0: 
                                text1 += " = {} + {} = {}".format(item.value, pool[idx].value, (item.value + pool[idx].value) % mod) 
                            else: 
                                text1 += " = {} + {} = {}".format(item.value, pool[idx].value, item.value + pool[idx].value) 
                        else: 
                            if mod > 0: 
                                text1 += " = {} + {} = {}; ".format(item.value, pool[idx].value, (item.value + pool[idx].value) % mod) 
                            else: 
                                text1 += " = {} + {} = {}; ".format(item.value, pool[idx].value, item.value + pool[idx].value) 
                        item.value = item.value + pool[idx].value 
                        if mod > 0: 
                            item.value = item.value % mod 
            
            # rng first 
            if G_necessary.rng in item.edgefromlist and item.rngnot != None: 
                new_char = random.choice(ch)[0] 
                ch.remove(new_char) 
                text1 += "; " 
                text1 += "so " 
                if item.rngnot == "+": 
                    if mod > 0: 
                        text1 += "{} = {} + {} = {} + {} = {}. ".format(new_char, item.rngv, item.variable, item.rngv, item.value, (item.rngv + item.value) % mod) 
                        item.value = (item.rngv + item.value) % mod 
                    else: 
                        text1 += "{} = {} + {} = {} + {} = {}. ".format(new_char, item.rngv, item.variable, item.rngv, item.value, item.rngv + item.value) 
                        item.value = item.rngv + item.value 
                    item.variable = new_char 
                elif item.rngnot == "*": 
                    if mod > 0: 
                        text1 += "{} = {} * {} = {} * {} = {}. ".format(new_char, item.rngv, item.variable, item.rngv, item.value, (item.rngv * item.value) % mod) 
                        item.value = (item.rngv * item.value) % mod 
                    else: 
                        text1 += "{} = {} * {} = {} * {} = {}. ".format(new_char, item.rngv, item.variable, item.rngv, item.value, item.rngv * item.value) 
                        item.value = item.rngv * item.value 
                    item.variable = new_char 
                else: 
                    raise ValueError("item.rngnot is not recognized.") 
                solution_text += "Define {} as {}; ".format(item.name, item.variable) 
                solution_text += text1 
            else: 
                solution_text += "Define {} as {}; ".format(item.name, item.variable) 
                solution_text += text1 + ". " 
            # print("item name {} item value {} ".format(item.name, item.value)) 
            if item.name not in patches.keys(): 
                patches[item.name] = item.value 
                patches_variable[item.name] = item.variable 
        else: 
            list_pair = [] 
            list_factor = [] 
            text1 = "" 
            accum_pair = [] 
            # print(colored("item name {} edgefromlist {}".format(item.name, item.edgefromlist), "cyan")) 
            
            for item2 in item.edgefromlist: 
                if isinstance(item2, InstanceParameter): 
                    list_factor.append(item2) 
                elif isinstance(item2, AbstractParameter): 
                    found = False 
                    for item3 in item.edgefromlist: 
                        if isinstance(item3, InstanceParameter): 
                            if ispairingvalie(item3, item2, item.name): 
                                # print("item2 {} item3 {} found a pair".format(item2.name, item3.name)) 
                                list_pair.append((item2, item3)) 
                                if item3 in list_factor: 
                                    list_factor.remove(item3) 
                                found = True 
                    assert found == True, "we have to find a pair for a abstract parameter." 

            # print(colored("under item {} len(list_pair) {} len(list_factor) {}".format(item.name, len(list_pair), len(list_factor)), "cyan")) 
            
            for pairr in list_pair: 
                abpara, inpara = pairr 
                new_char = random.choice(ch)[0] 
                ch.remove(new_char) 
                if len(list_pair) == 1 and len(list_factor) == 0: 
                    text1 += "so " 
                if abpara.value == None: 
                    if abpara.name in patches.keys(): 
                        # print("abpara name {} abpara value {} patches[abpara.name] {}".format(abpara.name, abpara.value, patches[abpara.name])) 
                        abpara.value = patches[abpara.name] 
                        abpara.variable = patches_variable[abpara.name] 
                    else: 
                        raise ValueError("abpara value is None. ") 
                if inpara.value == None: 
                    if inpara.name in patches.keys(): 
                        inpara.value = patches[inpara.name] 
                        inpara.variable = patches_variable[inpara.name] 
                    else: 
                        raise ValueError("inpara value is None. ") 
                # print("{} {}".format(abpara.name, abpara.value)) 
                if mod > 0: 
                    text1 += "{} = {} * {} = {} * {} = {}; ".format(new_char, inpara.variable, abpara.variable, inpara.value, abpara.value, (inpara.value * abpara.value) % mod) 
                    accum_pair.append((new_char, (inpara.value * abpara.value) % mod)) 
                else: 
                    text1 += "{} = {} * {} = {} * {} = {}; ".format(new_char, inpara.variable, abpara.variable, inpara.value, abpara.value, inpara.value * abpara.value) 
                    accum_pair.append((new_char, inpara.value * abpara.value)) 
            
            if len(accum_pair) == 1: 
                item.variable, item.value = accum_pair[0] 
            
            for idx in range(1, len(accum_pair)): 
                if idx == 1: 
                    pairr1 = accum_pair[idx - 1] 
                    cha, val = pairr1 
                    item.variable = cha 
                    item.value = val 
                
                pairr = accum_pair[idx] 
                cha, val = pairr 
                new_char = random.choice(ch)[0] 
                ch.remove(new_char) 
                # print("new_char {} cha {} val {} item.value {}".format(new_char, cha, val, item.value)) 
                if mod > 0: 
                    text1 += "{} = {} + {} = {} + {} = {}; ".format(new_char, item.variable, cha, item.value, val, (item.value + val) % mod) 
                    item.value = (item.value + val) % mod 
                else: 
                    text1 += "{} = {} + {} = {} + {} = {}; ".format(new_char, item.variable, cha, item.value, val, item.value + val) 
                    item.value = item.value + val 
                item.variable = new_char 
            
            if len(list_factor) == 1: 
                if list_factor[0].value == None: 
                    if list_factor[0].name in patches.keys(): 
                        list_factor[0].value = patches[list_factor[0].name] 
                        list_factor[0].variable = patches_variable[list_factor[0].name] 
                    else: 
                        raise ValueError("list_factor[0].value is None. ") 
                new_char = random.choice(ch)[0] 
                ch.remove(new_char) 
                text1 += "so " 
                text1 += "{} = {} = {}; ".format(new_char, list_factor[0].variable, list_factor[0].value) 
                
                item.value = list_factor[0].value 
                item.variable = new_char 
            else: 
                # print(colored("item name {} len(list_factor) {}".format(item.name, len(list_factor)), "cyan")) 
                for i in range(len(list_factor)): 
                    item2 = list_factor[i] 
                    if item2.value == None: 
                        if item2.name in patches.keys(): 
                            item2.value = patches[item2.name] 
                            item2.variable = patches_variable[item2.name] 
                        else: 
                            raise ValueError("previous has a none ") 
                    if item.variable == None: 
                        item.variable = item2.variable 
                        item.value = item2.value 
                        continue 
                    new_char = random.choice(ch)[0] 
                    ch.remove(new_char) 
                    if i == len(list_factor) - 1: 
                        text1 += "so " 
                    if mod > 0: 
                        text1 += "{} = {} + {} = {} + {} = {}; ".format(new_char, item.variable, item2.variable, item.value, item2.value, (item.value + item2.value) % mod) 
                        item.value = (item.value + item2.value) % mod 
                    else: 
                        text1 += "{} = {} + {} = {} + {} = {}; ".format(new_char, item.variable, item2.variable, item.value, item2.value, item.value + item2.value) 
                        item.value = item.value + item2.value 
                    item.variable = new_char 
                    # print(colored("item.name {} item.value {} item.variable {}".format(item.name, item.value, item.variable), "cyan")) 
            solution_text += "Define {} as {}; ".format(item.name, item.variable) 
            text1 = text1[: -2] + ". " 
            solution_text += text1 
            # print("item name {} item value {} ".format(item.name, item.value)) 
            if item.name not in patches.keys(): 
                # print("item name {} item value {} ".format(item.name, item.value)) 
                patches[item.name] = item.value 
                patches_variable[item.name] = item.variable 
        
    solution_text += "Answer: {}.".format(topo[-1].value) 

    return solution_text 

def dummylistinstanceparam(missingstatements): 
    sqrtnum = int(np.sqrt(missingstatements)) + 1 
    listtop = [] 
    listbot = [] 
    subcategoriess = random.sample(sorted(subcategories["Location"].keys()), int(sqrtnum // 10) + 1) 
    for subcategory in subcategoriess: 
        listtop += subcategories["Location"][subcategory] 
    
    subcategoriess = random.sample(sorted(subcategories["Animal"].keys()), int(sqrtnum // 10) + 1) 
    for subcategory in subcategoriess: 
        listbot += subcategories["Animal"][subcategory] 
    
    total_combination = [] 
    for outer in listtop: 
        for inner in listbot: 
            total_combination.append((outer, inner)) 
    
    random.shuffle(total_combination) 
    
    # total_combination = ["{}'s {}".format(outer, inner) for outer, inner in total_combination] 
    total_combination = ["{} in {}".format(inner, outer) for outer, inner in total_combination] 
    
    return total_combination[:missingstatements] 

def dummynetwokinginstanceparam(
    misslength, 
    core_graph, 
    number_range, 
    template, 
    d, 
    offset = 0, 
    topo = None, 
    lengthmessageone = None, 
): 
    # determining the number of statements we need 
    # number of added abstract parameters is x 
    # number of added instance parameters is 20 * x 
    # number of statements is roughly 20 * x part is from the core graph to the added instance parameters, the other part is the edges from the added abstract parameters to other unconnected instance parameters 
    total_add_statements = misslength // lengthmessageone # a conservative estimate 
    # total added instance parameter is total_add_statements, total added abstract parameter is total_add_statements // 10 
    added_abstract_num = total_add_statements // 20 + 1 # the approach taken is to be very conservative here 
    checkingsum = 0 
    for item in core_graph.instanceparameters: 
        checkingsum += len(item.edgefromlist) 
    for item in core_graph.abstractparameters: 
        checkingsum += len(item.edgefromlist) 
    print(colored("misslength {} total_add_statements {} added_abstract_num {}".format(misslength, total_add_statements, added_abstract_num), "cyan")) 
    
    dummyhierarchicalcate = {
        "crazy_zootopia": dummyhierarchical_categorizations, 
        "teachers_in_school": dummyhierarchical_categorizationstwo, 
        "movie_festival_awards": dummyhierarchical_categorizationsthree, 
    } 
    dummysubcategoriess = {
        "crazy_zootopia": dummysubcategories, 
        "teachers_in_school": dummysubcategoriestwo, 
        "movie_festival_awards": dummysubcategoriesthree, 
    } 
    
    list_abstract_parameters = [] 
    list_abstract_depth = [] 
    layers = {} 
    assistinglayersname = [] 
    
    additional_sentences = [] 
    havelength = 0 
    
    all_instance_parameters = [] 
    
    # step 1: get the abstract parameters 
    upperlevelname = dummyhierarchicalcate[template][0][0] 
    subcategoriess = random.sample(sorted(dummysubcategoriess[template][upperlevelname].keys()), (added_abstract_num // 20) + 1)  
    listtop = [x for subcategory in subcategoriess for x in dummysubcategoriess[template][upperlevelname][subcategory]] 
    abstractparamcount = 0 
    for nametop in listtop: 
        if d == 2: 
            catebottom = dummyhierarchicalcate[template][0][1] 
            list_abstract_depth.append(1) 
            layers[(catebottom, nametop)] = [] 
            assistinglayersname.append(nametop) 
        elif d == 3: 
            catebottom = dummyhierarchicalcate[template][0][2] 
            list_abstract_depth.append(2) 
            layers[(catebottom, nametop)] = [] 
            assistinglayersname.append(nametop) 
        abstractparamcount += 1 
        if abstractparamcount >= added_abstract_num: 
            break 
    print(colored("number of keys in layers {}".format(len(layers.keys())), "cyan")) 

    # step 2: get the instance parameters 
    instance_count = 0 
    for i, abstractparam in enumerate(layers.keys()): 
        # catebottom, nametop = abstractparam.split(" in ") 
        catebottom, nametop = abstractparam 
        depth = list_abstract_depth[i] 
        namesecond = dummyhierarchicalcate[template][0][1] # Adult Animal 
        subcategory_dummy = random.sample(sorted(dummysubcategoriess[template][namesecond].keys()), 2) 
        listbot = [] 
        for subcategory_du in subcategory_dummy: 
            listbot += [x for x in dummysubcategoriess[template][namesecond][subcategory_du]] 
        for namebot in listbot: 
            layers[abstractparam].append("{} in {}".format(namebot.lower(), nametop)) 
            instance_count += 1 
            if depth == 2: 
                if instance_count >= total_add_statements//2: 
                    print(colored("instance count {}".format(instance_count), "cyan")) 
                    break 
            else: 
                if instance_count >= total_add_statements: 
                    break 
        
        # for hard set, we need more instance parameters 
        if depth == 2: 
            namethird = dummyhierarchicalcate[template][0][2] 
            addedlist = [] 
            for item in layers[abstractparam]: 
                addedlist.append("{} per {}".format(list(dummysubcategoriess[template][namethird].values())[0][0], item)) 
            layers[abstractparam] += addedlist 
    print(colored("instance count {}".format(instance_count), "cyan")) 
    
    # step 3: connect the parameters 
    auxilarygraph = DependencyGraph() 
    for abstractparam in layers.keys(): 
        abstract_parameter_node = AbstractParameter("{} in {}".format(abstractparam[0], abstractparam[1])) 
        list_abstract_parameters.append(abstract_parameter_node) 
        for instanceparam in layers[abstractparam]: 
            instance_parameter_node = InstanceParameter(instanceparam) 
            auxilarygraph.add_edge(from_node = instance_parameter_node, to_node = abstract_parameter_node) 
            all_instance_parameters.append(instance_parameter_node) 
    print(colored("abstract count {} instance count {}".format(len(list_abstract_parameters), len(all_instance_parameters)), "cyan")) 
    # print(colored("abstract count {} instance count {}".format(len(auxilarygraph.abstractparameters), len(auxilarygraph.instanceparameters)), "cyan")) 
    
    # step 5: connect between stem and auxilarygraph 
    random.shuffle(all_instance_parameters) 
    can_select = core_graph.instanceparameters + core_graph.abstractparameters 
    for item in all_instance_parameters: 
        
        if random.random() < 0.5: 
            num_picks = 2 
        else: 
            num_picks = 1 
        
        # num_picks = 1 
        picklist = random.sample(can_select, num_picks) 
        for coreinstance in picklist: 
            auxilarygraph.add_edge(from_node = coreinstance, to_node = item) 
            item.coretoflag += 1 
    
    # step 4: connect between abstract parameters and instance parameters 
    random.shuffle(all_instance_parameters) 
    for item in all_instance_parameters: 
        assert len(item.edgetolist) == 1 # only to the abstract parameter 
        parentnode = item.edgetolist[0] 
        assert isinstance(parentnode, AbstractParameter) 
        if parentnode not in list_abstract_parameters: 
            continue 
        if item.coretoflag >= 2: 
            continue 
        pickparam = None 
        found = False 
        for abstractparam in list_abstract_parameters: 
            if item in abstractparam.edgefromlist: 
                # first, check if the item is the child of the abstract parameter 
                continue 
            else: 
                pickparam = abstractparam 
                found = True 
                break 
        
        if found: 
            auxilarygraph.add_edge(from_node = pickparam, to_node = item) 
            list_abstract_parameters.remove(pickparam) 
    
    '''
    # step 6: connect within the cluster between instance 
    for abitem in auxilarygraph.abstractparameters: 
        if abitem in core_graph.abstractparameters: 
            continue 
        instancegroup = abitem.edgefromlist 
        needgroup = [] 
        supplygroup = [] 
        for item in instancegroup: 
            if len(item.edgefromlist) == 1: 
                needgroup.append(item) 
            else: 
                supplygroup.append(item) 
        
        if len(needgroup) == 0: 
            continue 
        
        if len(supplygroup) == 0: 
            lengthonehalf = len(needgroup) // 2 
            supplygroup = needgroup[:lengthonehalf] 
            needgroup = needgroup[lengthonehalf:] 
            
        if len(supplygroup) == 0: 
            continue 
        
        print(colored("needgroup {} supplygroup {}".format(len(needgroup), len(supplygroup)), "cyan")) 
        idx = 0 
        for item in needgroup: 
            auxilarygraph.add_edge(from_node = supplygroup[idx], to_node = item) 
            idx = (idx + 1) % len(supplygroup) 
    ''' 
    
    # attach English 
    random.shuffle(all_instance_parameters) 
    for item in all_instance_parameters: 
        sentence = generate_sentence(auxilarygraph, item, number_range = number_range) 
        additional_sentences.append(sentence) 
    
    # auxilarygraph.drawmess(core_graph) 
    listfromna = [] 
    for item in all_instance_parameters: 
        listfromna.append(len(item.edgefromlist)) 
    print(colored("Statistics - mean {} max {} min {}".format(np.mean(listfromna), np.max(listfromna), np.min(listfromna)), "cyan")) 
    print(colored("number of actual statements {} misslength {} havelength {}".format(len(additional_sentences), misslength, havelength), "cyan")) 
    checkingsum2 = 0 
    for item in core_graph.instanceparameters: 
        checkingsum2 += len(item.edgefromlist) 
    for item in core_graph.abstractparameters: 
        checkingsum2 += len(item.edgefromlist) 
    print(colored("checkingsum {} checkingsum2 {}".format(checkingsum, checkingsum2), "cyan")) 
    assert checkingsum == checkingsum2 
    
    return additional_sentences 

def generate_problem_text(
    G_necessary, 
    problem_segments, 
    question_text, 
    solution_text, 
    target_length, 
    tokenizer = None, 
    number_range = None, 
    template = None, 
    d = None, 
    offset = 0, 
    topo = None, 
): 
    messagechoice = {
        "crazy_zootopia": message, 
        "teachers_in_school": messagetwo, 
        "movie_festival_awards": messagethree, 
    } 
    lengthperone = {
        "crazy_zootopia": lengthmessage, 
        "teachers_in_school": lengthmessagetwo, 
        "movie_festival_awards": lengthmessagethree, 
    } 
    messageone = messagechoice[template] 
    lengthmessageone = lengthperone[template] 
    
    # first estimate the length of the exisiting text 
    problem_text = ". ".join(problem_segments) 
    total_text = messageone # initial prompt 
    for i in range(6): 
        total_text += "Problem: " + problem_text + " Question: " + question_text + " Solution: " + solution_text + "\n" 
    total_text += "Problem: " + problem_text + " Question: " + question_text + " Solution: " 
    length = len(tokenizer.encode(total_text)) # although there is some special prompt tokens not taken in consideration, but it is fine for now 
    misslength = target_length - length 
    print(colored("target length {} length {} misslength {}".format(target_length, length, misslength), "yellow")) 
    # we get actual link and missing link here 
    list_statements = None 
    list_statements = [] 
    if target_length != None and misslength > 0: 
        # TODO: we add dummy statements here in the list of list_statements 
        list_statements += dummynetwokinginstanceparam(misslength, G_necessary, number_range, template, d, offset, topo, lengthmessageone = lengthmessageone) 
    list_statements += problem_segments 
    random.shuffle(list_statements) 
    
    text_output = "" 
    for sentence in list_statements: 
        text_output += sentence + ". " 
    
    return text_output 

def simple_aggregate(problem_segments): 
    random.shuffle(problem_segments) 
    
    text_output = "" 
    for sentence in problem_segments: 
        text_output += sentence + ". " 
    
    return text_output 

def remove_dup(topo): 
    founddup = False 
    for item in topo: 
        if isinstance(item, InstanceParameter): 
            foundnames = [] 
            itemtoremove = [] 
            for item2 in item.edgefromlist: 
                if item2.name not in foundnames: 
                    foundnames.append(item2.name) 
                else: 
                    # we need to remove the edge going from an instance parameter not from an abstract parameter 
                    founddup = True 
                    if isinstance(item2, InstanceParameter): 
                        item2.edgetolist.remove(item) 
                        itemtoremove.append(item2) 
                    else: 
                        assert isinstance(item2, AbstractParameterSpecial) 
                        item3 = item2.edgefromlist[0] 
                        assert item3 in item.edgefromlist 
                        item3.edgetolist.remove(item2) 
                        itemtoremove.append(item3) 
            for item2 in itemtoremove: 
                item.edgefromlist.remove(item2) 
    return founddup 

def detectnegative(topo): 
    for item in topo: 
        if item.value < 0: 
            return True 
        elif item.value == 0: 
            if "average" in item.name or "total" in item.name or "all" in item.name: 
                continue 
            else: 
                return True 
        elif item.value > 1e3: # number's too hard to compute 
            return True 
    return False 

def addingparameter(G_necessary, topo, instanceparameterslist): 
    newit = None 
    for item in G_necessary.instanceparameters: 
        if item not in topo: 
            newit = item 
            if random.random() < 0.5: 
                G_necessary.add_edge(from_node = item, to_node = topo[0]) 
                topo.insert(0, item) 
            else: 
                G_necessary.add_edge(from_node = topo[1], to_node = item) 
                topo.append(item) 
            break 
    if newit == None: 
        for item in instanceparameterslist: 
            if item not in topo: 
                newit = item 
                if random.random() < 0.5: 
                    G_necessary.add_edge(from_node = item, to_node = topo[0]) 
                    topo.insert(0, item) 
                else: 
                    G_necessary.add_edge(from_node = topo[1], to_node = item) 
                    topo.append(item) 
                break 

def drawAll(
    op_max, # the maximum number of operations
    ip_max, # the maximum number of instance parameters
    verbose = False, 
    mod: int = -1, 
    force = False, 
    number_range = 23, 
    strictline = -1, 
    outputlistparameters = False, 
    target_length = None, 
    template = "crazy_zootopia", 
    d = None, 
    tokenizer = None, 
    oplist = None, 
): 
    # small patch: adjusting the op_max number to better increase the practical success rate 
    if target_length != "zero_context": 
        assert tokenizer != None 
    
    if strictline != -1: 
        op_max = (strictline + 1) // 2 
    
    # 1. s ← min{t0, t1} for t0, t1 being two random integers from 1 and opmax 
    s = min([random.randint(1, op_max) for _ in range(2)]) 
    # 2. If force = true then s ← opmax. 
    if force: 
        s = op_max 
    # n ← max{t0, t1} for t0, t1 being two random integers from 1 and s 
    n = max([random.randint(1, s) for _ in range(2)]) 
    # m ← random integer between n and s 
    m = random.randint(n, s) 
    # 5: d ← a random choice among {2, 3, 4} with distribution according to softmax(weight)
    # ⋄ for weight = [−(rel − 0.2)2, −(rel − 0.5)2, −(rel − 0.8)2] for rel = s−1
    # ipmax −1
    rel = (s - 1) / (ip_max - 1) 
    weight = [-((rel - 0.1) ** 2), -((rel - 0.3)) ** 2, -((rel - 0.5)) ** 2, -((rel - 0.7)) ** 2, -((rel - 0.9)) ** 2] 
    probabilitydistribution = np.exp(weight) / np.sum(np.exp(weight)) 
    
    # here the depth options are either 2 or 3 
    # d = 2 if random.random() < 0.25 else 3 
    # 6: t0, t1 ← two random choices among {2, 3, 4} with distribution according to softmax(weight) 
    # choices going from 1 to 5? 
    # t0 = np.random.choice([2, 3, 4], p = probabilitydistribution) # t0 
    t0 = np.random.choice([1, 2, 3, 4, 5], p = probabilitydistribution) # t0 
    # t1 = np.random.choice([2, 3, 4], p = probabilitydistribution) # t1 
    t1 = np.random.choice([1, 2, 3, 4, 5], p = probabilitydistribution) # t1 
    
    if s <= 5 and d == 2: 
        t1 = max(t1, 2) 
        t0 = max(t0, 2) 
    
    # 7: w0 ← min{t0, t1} and w1 ← max{t0, t1}. 
    # print(t0, t1) 
    w0 = min(t0, t1) 
    w1 = max(t0, t1) 
    # 8: e ← min{t0, t1, (d − 1)w12} for t0, t1 being random integers between (d − 1)w0 and ipmax 
    t0 = random.randint((d - 1) * w0, ip_max) 
    t1 = random.randint((d - 1) * w0, ip_max) 
    # e = min(t0, t1, (d - 1) * (w1 ** 2)) 
    if d == 2: 
        e = min(t0, t1, (d - 1) * (w1 ** 2)) 
    elif d == 3: 
        e = min(t0, t1, (d - 2) * (w1 ** 2) + w1) 
    # 9: Gs ← DrawStructure(e, d, w0 , w1 ) 
    print("selected e {} d {} w0 {} w1 {} s {} n {} m {}".format(e, d, w0, w1, s, n, m)) 
    
    assert template in ["crazy_zootopia", "teachers_in_school", "movie_festival_awards"] 
    
    G_structure = drawStructure(e, d, w0, w1, template = template) # e is number of edges, d is number of layers, w0 is the minimum number of nodes per layer, w1 is the maximum number of nodes per layer 
    if verbose: 
        G_structure.draw() 
        G_structure.save_graph_structure("graph_structure.pkl") 
    
    # G_structure = GraphStructure.load_graph_structure("graph_structure.pkl") 
    G_necessary, abstract_parameter_list, instance_parameter_list = drawNecessary1(G_structure, n, m) 
    if verbose: 
        G_necessary.draw1() 
    G_necessary, topo = drawNecessary2(G_necessary) 
    if len(topo) == 2 and random.random() < 0.5: 
        print(colored("s == 3 and len(topo) == 2", "red")) 
        addingparameter(G_necessary, topo, instance_parameter_list) 
    
    # TODO: find parameter to convert to image 
    if verbose: 
        graph_checker(G_necessary) 
        G_necessary.draw2() 
        G_necessary.visualization_topo(topo) 
    assert is_topo(topo) 
    G_necessary = drawNecessary3(G_necessary, topo, s) 
    if verbose: 
        G_necessary.draw3() 
    founddup = remove_dup(topo) 
    
    # print("founddup {}".format(founddup)) 
    G_necessary = drawUnnecessary(G_necessary, abstract_parameter_list, instance_parameter_list, G_structure) 
    if verbose: 
        G_necessary.draw4(topo) 
    # G_structure.draw() 
    
    structuredescription = generate_description_structure(G_structure) 
    # print(structuredescription) 
    problem_segments = problem_in_text(G_necessary = G_necessary, number_range = number_range) 
    question_text = question_in_text(topo) 
    solution_text = solution_in_text(G_necessary, topo, mod = mod) 
    # TODO: append image 
    if oplist is not None: 
        print("number of operations {} not in oplist {}".format(num_op(topo), oplist)) 
        assert num_op(topo) in oplist 
    # print(colored("offsets per statement {}".format(offsetsperstatement), "cyan")) 
    
    # checking for negative values 
    if detectnegative(topo): 
        raise ValueError("Negative values are detected.") 
    
    numberopscalculated = num_op(topo) 
    if strictline != -1: 
        if numberopscalculated > strictline: 
            raise ValueError("Number of operations {} is greather than the strict requirement".format(numberopscalculated)) 
    
    if target_length == "zero_context": 
        problem_text = simple_aggregate(problem_segments) 
    else: 
        converttable = {
            "4k": 4000, 
            "8k": 8000, 
            "16k": 16000, 
            "32k": 32000, 
            "64k": 64000, 
            "128k": 128000, 
        } 
        target_length = converttable[target_length] - 1000 # 1000 is the buffer 
        problem_text = generate_problem_text(
            G_necessary, 
            problem_segments, 
            question_text, 
            solution_text, 
            target_length = target_length, 
            tokenizer = tokenizer, 
            number_range = number_range, 
            template = template, 
            offset = None, 
            d = d, 
        ) 
    
    id = string_to_number(solution_text) 
    
    if outputlistparameters: 
        structabparam = G_necessary.abstractparameters 
        structinparam = G_necessary.instanceparameters 
        return problem_text, question_text, solution_text, numberopscalculated, id, structabparam, structinparam, topo 
    else: 
        return problem_text, question_text, solution_text, numberopscalculated, id 

def string_to_number(s): 
    hash_value = hash(s) 
    return hash_value % 23 

if __name__ == "__main__": 
    # problem, question, solution, num_op, id = drawAll(op_max = 1, ip_max = 20, force = True, verbose = True, mod = 10) 
    tokenizer = AutoTokenizer.from_pretrained("/netcache/huggingface/Meta-Llama-3-8B-Instruct")
    
    problem, question, solution, opnum, id = drawAll(
        op_max = 10, 
        ip_max = 10, 
        force = True, 
        verbose = True, 
        mod = -1, 
        strictline = 40, 
        target_length = "zero_context", 
        number_range = 5, 
        template = "crazy_zootopia", 
        # template = "movie_festival_awards", 
        d = 3, 
        tokenizer = tokenizer, 
    ) 

    # messageone = "Answer the questions below. Note: any Location's Adult Animal refers to sum of all types of adult animals ever mentioned for the specific location throughout the problem EXCLUDING their number of newborn children. The average number of newborn children of the same type of animal might vary across different locations. The Location's Total Newborn Children refers to the sum of the TOTAL newborn children (not average newborn children) from all adult animals mentioned for that specific location. Each question is independent of the others." 
    # message = "Answer the questions below. Note: any Location's Adult Animal refers to sum of all types of adult animals ever mentioned for the specific location throughout the problem EXCLUDING their number of newborn children. The average number of newborn children of the same type of animal might vary across different locations. The Location's Total Newborn Children refers to the sum of the TOTAL newborn children (not average newborn children) from all adult animals mentioned for that specific location. Each question is independent of the others." 
    print(message) 
    print(colored("Problem: {}".format(problem), "yellow")) 
    print(colored("Question: {}".format(question), "cyan")) 
    print(colored("Solution: {}".format(solution), "green")) 
    print("Number of operations: ", opnum) 
    print("Length of Problem: ", len(tokenizer(problem)["input_ids"])) 
    print(string_to_number(solution)) 
