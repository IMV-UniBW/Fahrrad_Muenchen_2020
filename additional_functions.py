# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:28:15 2021

@author: katha
"""

def remove_brackets(name):
    new_name = name
    while '[' in new_name:
        if new_name.find('[') >= 0:
            i_1 = new_name.find('[')
            i_2 = new_name.find(']')
            new_name = new_name[:i_1] + new_name[i_2+1:]
    return new_name

def make_variable_list(names):
    # remove square bracketsp
    new_names = []
    for p in names:
        new_name = remove_brackets(p)
        if new_name != 'Intercept':
            new_names.append(new_name)
    # remove doubles
    unique_names = list(set(new_names))
    return unique_names

def make_formula(my_list):
    formula = 'gesamt3 ~' + '+'.join(my_list)
    return formula

def remove_higher_order_interactions(names):
    new_names = names.copy()
    for p in names:
        if p.count(':') > 1:
            new_names.remove(p)
    return new_names