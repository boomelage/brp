#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 
Created on Sat Aug 31 03:51:15 2024

"""

import concurrent.futures

def threadpooler(functions):
    """
    Executes a list of functions in a thread pool with error handling.

    Parameters:
    functions (list): A list of functions to be executed.

    Returns:
    dict: A nested dicitonary with the key 'outcome' giving the function's
    result which is, in turn, accessed by the function's name as a string
    """
    results = {}

    def wrapper(func):
        try:
            result = func()
            return {func.__name__: {'outcome': result}}
        except Exception as e:
            return {func.__name__: {'outcome': e}}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_func = {executor.submit(wrapper, func): func for func in functions}
        for future in concurrent.futures.as_completed(future_to_func):
            func = future_to_func[future]
            try:
                result = future.result()
            except Exception as e:
                results[func.__name__] = {'outcome': e}
            else:
                results.update(result)
    
    return results



# functions = [test_func_1, test_func_2, test_func_3]
# results = threadpooler(functions)

        