#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 19:04:39 2024

AN APPROXIMATION

"""

def generate_ivol_table(n_lists, n_elements, start_value, decay_rate, row_decay):
    """
    AN APPROXIMATION OF IVOL TABLE
    
    Generates a list of lists with n_lists number of lists and n_elements number of elements in each list.
    The values gradually decrease based on the start_value, decay_rate, and row_decay.
    
    :param n_lists: Number of lists
    :param n_elements: Number of elements in each list
    :param start_value: Starting value for the first element in the first list
    :param decay_rate: Rate at which the values decay within each list
    :param row_decay: Rate at which the starting value decays for each new list
    :return: List of lists of floating point values
    """
    data = []
    for i in range(n_lists):
        current_list = []
        base_value = start_value - i * row_decay  # Decrease the starting value for each new list
        for j in range(n_elements):
            # Decrease values for each element in the list
            value = base_value - j * decay_rate
            current_list.append(value)
        data.append(current_list)
    
    return data
