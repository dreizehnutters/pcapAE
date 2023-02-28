import copy
import pickle
import torch
import types
from . import layers
from . import rules
Rules = rules.Rules

def flatten_model(module):
    '''
    flatten modul to base operation like Conv2, Linear, ...
    '''
    modules_list = []
    for m_1 in module.children():

        if len(list(m_1.children())) == 0:
            modules_list.append(m_1)
        else:
            modules_list = modules_list + flatten_model(m_1)
    return modules_list


def copy_module(module):
    '''
    sometimes copy.deepcopy() does not work
    '''
    module = copy.deepcopy(pickle.loads(pickle.dumps(module)))
    module._forward_hooks.popitem()  # remove hooks from module copy
    module._backward_hooks.popitem()  # remove hooks from module copy
    return module

def redefine_nn(model, rule, input_lowest, input_highest):
    '''
    go over model layers and overload chosen instance methods (e.g. forward()).
    New methods come from classes of layers module
    '''
    rule_func = Rules(rule)
    list_of_layers = dir(layers) #list of redefined layers in layers module
    for num, module in enumerate(flatten_model(model)):
        if  module.__class__.__name__ in list_of_layers:
            local_class = module.__class__ #current layer class
            layer_module_class = layers.__getattr__(local_class.__name__) # get same redefined layer class
            list_of_methods = [attr for attr in dir(layer_module_class) if attr[:2] != '__'] #methods which  was redefined
            for l in list_of_methods:
                #overload object method from https://stackoverflow.com/questions/394770/override-a-method-at-instance-level
                setattr(module, l, types.MethodType(getattr(layer_module_class, l), module)) #set redefined methods to object
            if num == 0:
                setattr(module, 'rule_func', Rules('z_box_no_bias', lowest=input_lowest, highest=input_highest)) #first layer always z_box
            else:
                setattr(module, 'rule_func', rule_func)
    return model
