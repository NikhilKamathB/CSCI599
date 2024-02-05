###################################################################################################
# Miscelaneous utilities for the project goes here
###################################################################################################

def intify(item: list) -> list:
    """
    Convert a list of strings to a list of integers.
    Input parameters:
        - item: list of strings
    Output:
        - list of integers
    """
    return list(map(int, item))

def floatify(item: list) -> list:
    """
    Convert a list of strings to a list of floats.
    Input parameters:
        - item: list of strings
    Output:
        - list of floats
    """
    return list(map(float, item))