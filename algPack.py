#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
project2.py: A package of algorithms for matrices of experiments, dna strings, 
             and vectors of lab tests.
"""

__author__      = "Jessica Stothers"

import sys 

def test(did_pass):
    """ Print the result of a test. """
    linenum = sys._getframe(1).f_lineno # Get the caller's line number.
    if did_pass:
        msg = "Test at line {0} ok.".format(linenum)
    else:
        msg = ("Test at line {0} FAILED.".format(linenum))
    print(msg)
    
def test_suite():
    """ 
    Run the suite of tests for code in this module (this file).
    """
    print("colMean tests:")
    test(colMean([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2) == 4.0)
    test(colMean([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 9) == None)
    test(colMean([[2, 4, 6], [-1, -2, -3], [1, 2, 3]], 2) == 2.0)
    test(colMean([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 0) == 0.0)
    test(colMean([[-1, -2, -3, -4], [-5, -6, -7, -8], [-9, -10, -11, -12]], 3) == -8.0)
    test(colMean([[0], [-1], [100]], 0) == 33.0)
    test(colMean([[200, 40, 6000], [-10, -2000, -300], [10, 2, 30000]], 2) == 11900.0)
    test(colMean([[2, 2, 2], [2, 2, 2], [2, 2, 2]], 1) == 2.0)
    test(colMean([[2, 4, 6], [-100, -200, -300], [1, 2, 3]], 2) == -97.0)
    test(colMean([[-2, -4, -6], [100, 200, 300], [-1, -2, -3]], 2) == 97.0)
    test(colMean([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 0) == 0.0)
    test(abs(colMean([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2) - 4.0) < .0000001 )
    print("colMode tests:")
    test(colMode([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2) == 3)
    test(colMode([[1,2,3], [222,2,3], [4,5,6]], 1) == 2)
    test(colMode([[1,2,3], [222,2,3], [4,5,6]], 0) == None)
    test(colMode([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 4) == None)
    test(colMode([[-2, -4, -6], [-1, -2, -3], [-1, -2, -3]], 0) == -1)
    test(colMode([[0, -4, -6], [-1, -2, -3], [0, -2, -3]], 0) == 0)
    test(colMode([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 1) == 0)
    test(colMode([[2, 4, 6], [1, 2, 3], [1, 2, 3], [6, 6, 6], [3, 3, 6]],  2) == 6) 
    test(colMode([[2, 4, 6], [1, 2, 6], [1, 2, 3], [6, 6, 6], [3, 3, 3]],  2) == 6) 
    test(colMode([[2, 4, 6], [1, 2, 3], [1, 2, 3], [6, 6, 6], [3, 3, 5]],  2) == (3, 6)) # This fails since my function can't handle multiple modes
    print("colSD tests")
    test(colSD([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2) == 1.7320508075688772)
    test(colSD([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 0) == 0.0)
    test(colSD([[-2, -4], [-1, -2], [-1, -2]], 1) == 1.1547005383792515)
    test(colSD([[-2, -4], [1, 2], [0, 0]], 1) == 3.055050463303893)
    test(colSD([[-2], [1], [0]], 0) == 1.5275252316519465)
    print("colStandardize tests:")
    test(colStandardize([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2) == [[2, 4, [1.1547005383792517]], [1, 2, [-0.5773502691896258]], [1, 2, [-0.5773502691896258]]])
    test(colStandardize([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 8) == None)
    test(colStandardize([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 0) == None)
    test(colStandardize([[-2, 4, 6], [-1, 2, 3], [-1, 2, 3]], 0) == [[[-1.1547005383792517], 4, 6], [[0.5773502691896256], 2, 3], [[0.5773502691896256], 2, 3]])
    test(colStandardize([[-2, 4, 6], [1, 2, 3], [-2, 2, 3]], 0) == [[[-0.5773502691896258], 4, 6], [[1.1547005383792517], 2, 3], [[-0.5773502691896258], 2, 3]])
    test(colStandardize([[1,2,3,4], [5,6,7,8], [-9,-10,-11,-12]], 3) == [[1, 2, 3, [0.3779644730092272]], [5, 6, 7, [0.7559289460184544]], [-9, -10, -11, [-1.1338934190276817]]])
    test(colStandardize([[1,1,1], [1,1,1], [1,1,1], [1,1,1]], 1) == None)
    test(colStandardize([[1,2,3], [4,5,6], [7,8,9], [-1,-2,-3], [-4,-5,-6], [-7,-8,-9]], 1) == [[1, [0.32791291789197646], 3], [4, [0.8197822947299411], 6], [7, [1.3116516715679059], 9], [-1, [-0.32791291789197646], -3], [-4, [-0.8197822947299411], -6], [-7, [-1.3116516715679059], -9]])
    test(colStandardize([[1,2,3]], 1) == None)
    test(colStandardize([[1,2,3], [2,1,2]], 1) == [[1, [0.7071067811865475], 3], [2, [-0.7071067811865475], 2]])
    test(colStandardize([[-2, -4, -6], [-1, -2, -3], [-1, -2, -3]], 2) == [[-2, -4, [-1.1547005383792517]], [-1, -2, [0.5773502691896258]], [-1, -2, [0.5773502691896258]]])
    test(colStandardize([[2], [5]], 0) == [[[-0.7071067811865476]], [[0.7071067811865476]]])
    print("colMax tests")
    test(colMax([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2) == 6)
    test(colMax([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 0) == 0)
    test(colMax([[-2, -4, -6], [-1, -2, -3], [-1, -2, -3]], 2) == -3)
    test(colMax([[0, 0 ,0], [1, 2, 3], [-1, -2, -3]], 2) == 3)
    test(colMax([[0], [1], [-1]], 0) == 1)
    test(colMax([[0]], 0) == 0)
    print("colMin tests")
    test(colMin([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2) == 3)
    test(colMin([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 0) == 0)
    test(colMin([[-2, -4, -6], [-1, -2, -3], [-1, -2, -3]], 2) == -6)
    test(colMin([[0, 0 ,0], [1, 2, 3], [-1, -2, -3]], 2) == -3)
    test(colMin([[0], [1], [-1]], 0) == -1)
    test(colMin([[0]], 0) == 0)
    print("colMinMaxNormalize tests:")
    test(colMinMaxNormalize([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2) == [[2, 4, 1], [1, 2, 0], [1, 2, 0]])
    test(colMinMaxNormalize([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 7) == None)
    test(colMinMaxNormalize([[2, 0, 6], [1, 0, 3], [1, 0, 3]], 1) == None)
    test(colMinMaxNormalize([[-2, 0, 6], [-1, 0, 3], [-1, 0, 3]], 0) == [[0, 0, 6], [1, 0, 3], [1, 0, 3]])
    test(colMinMaxNormalize([[-2, 0, 6], [1, 0, 3], [-1, 0, 3]], 0) == [[0, 0, 6], [1, 0, 3], [0, 0, 3]])
    test(colMinMaxNormalize([[-2, 0, 6]], 0) == None)
    test(colMinMaxNormalize([[-2, 0, 6, 5], [2, 2, 2, 2]], 3) == [[-2, 0, 6, 1], [2, 2, 2, 0]])
    test(colMinMaxNormalize([[-2, 0, 6, 5], [2, 2, 2, 2], [-1, -200, -3, -50000]], 3) == [[-2, 0, 6, 1], [2, 2, 2, 0], [-1, -200, -3, 0]])
    test(colMinMaxNormalize([[-2, 0, 6, 5], [2, 2, 2, 2], [-1, -200, -3, -50000]], 0) == [[0, 0, 6, 5], [1, 2, 2, 2], [0, -200, -3, -50000]])
    test(colMinMaxNormalize([[5, 5, 5],[5, 5, 5], [5, 5, 5]], 0) == None)
    test(colMinMaxNormalize([[-2, -4, -6], [-1, -2, -3], [-1, -2, -3]], 2) == [[-2, -4, 0], [-1, -2, 1], [-1, -2, 1]])
    test(colMinMaxNormalize([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 0) == None)
    print("mutation tests:")
    test(mutation("ACTCGG", 0, "G") == "GCTCGG")
    test(mutation("ACTCGG", 7, "G") == None)
    test(mutation("", 0, "G") == None)
    test(mutation("ACTCGG", 5, "A") == "ACTCGA")
    test(mutation("ACTCGG", 2, "A") == "ACACGG")
    test(mutation("ACTCGGACTCGGACTCGG", 10, "A") == "ACTCGGACTCAGACTCGG")
    test(mutation("ACTCGG", 3, "C") == "ACTCGG")
    test(mutation("CCCCCCC", 5, "B") == "CCCCCBC")
    test(mutation("CCCCCCC", 1, "B") == "CBCCCCC")
    test(mutation("ACTCGG", 0, "") == "CTCGG")
    test(mutation("CCCCCCC", 7, "B") == None)
    test(mutation("C", 0, "B") == "B")
    test(mutation("ACGGGTTTACTTGGAACTTTGGGAAAACTGAAAAGGGTTATATATATGGGCAAC", 40, "A") == "ACGGGTTTACTTGGAACTTTGGGAAAACTGAAAAGGGTTAAATATATGGGCAAC")
    print("insertion tests:")
    test(insertion ("ACTCGG", 6, "AGC") == "ACTCGGAGC")
    test(insertion("ACT", 1, "GGGGG") == "AGGGGGCT")
    test(insertion ("ACTCGG", 7, "AGC") == None)
    test(insertion ("ACTCGG", 3, "AGC") == "ACTAGCCGG")
    test(insertion ("ACTCGG", 3, "") == "ACTCGG")
    test(insertion ("", 0, "GAT") == "GAT")
    test(insertion ("", 1, "GAT") == None)
    test(insertion ("GAT", 1, "ACTCGG") == "GACTCGGAT")
    test(insertion ("GAT", 1, "A") == "GAAT")
    test(insertion ("G", 1, "ACTCGGAGC") == "GACTCGGAGC")
    test(insertion("ACT", 2, "ACGGGTTTACTTGGAACTTTGGGAAAACTGAAAAGGGTTATATATATGGGCAAC") == "ACACGGGTTTACTTGGAACTTTGGGAAAACTGAAAAGGGTTATATATATGGGCAACT")
    print("deletion tests:")
    test(deletion("ACTCGG", 5, 2) == "ACTCG")
    test(deletion("ACTCGG", 1, 2) == "ACGG")
    test(deletion("ACTCGG", 7, 2) == None)
    test(deletion("ACTCGG", 6, 1) == None)
    test(deletion("ACTCGG", 0, 1) == "CTCGG")
    test(deletion("ACTCGG", 0, 5) == "G")
    test(deletion("ACTCGG", 0, 6) == "")
    test(deletion("", 0, 6) == None)
    test(deletion("ACTCGG", 0, 0) == "ACTCGG")
    test(deletion("ACTCGG", 3, 0) == "ACTCGG")
    test(deletion("ACGGGTTTACTTGGAACTTTGGGAAAACTGAAAAGGGTTATATATATGGGCAAC", 30, 20) == "ACGGGTTTACTTGGAACTTTGGGAAAACTGCAAC")
    print("euclideanDistance tests:")
    test(euclideanDistance([3, 1], [6, 5]) == 5.0)
    test(euclideanDistance([0, 0], [3, 4]) == 5.0)
    test(euclideanDistance([0, 0], [0, 0]) == 0.0)
    test(euclideanDistance([2, 4], [-3, 8]) == 6.4031242374328485)
    test(euclideanDistance([0,0], [7,6]) == 9.219544457292887)
    test(euclideanDistance([-3,-1], [-6,-5]) == 5.0)
    test(euclideanDistance([-3,-1], []) == None)
    test(euclideanDistance([3, 1, 4], [6, 5, 6]) == 5.385164807134504)
    test(euclideanDistance([3, 1, 4, 2, 1, 7, 800, 20, 5, -11, 50], [6, 5, 6, 0, 0, -3, -1111111, 4, 2, 6666, 9]) == 1111931.0483703564)
    test(euclideanDistance([3], [6]) == 3.0)
    test(abs(euclideanDistance([3, 1], [6, 5]) - 5) < .0000001)
    test(abs(euclideanDistance([0, 0], [3, 4]) - 5) < .0000001)
    test(euclideanDistance([3, 6, 1, 2, 8, 2, 1], [3, 6, 1, 2, 8, 2, 1]) == 0)
    print("normalizeVector tests:")
    test(normalizeVector([6, 8]) == [0.6, 0.8])
    test(normalizeVector([]) == [])
    test(normalizeVector([1]) == [1.0])
    test(normalizeVector([-1, -6, -3, -8, -1]) == [-0.0949157995752499, -0.5694947974514994, -0.2847473987257497, -0.7593263966019992, -0.0949157995752499])
    test(normalizeVector([0, 0, 0]) == None)
    test(normalizeVector([0, -1, 0]) == [0.0, -1.0, 0.0])
    test(normalizeVector([5]) == [1.0])
    test(normalizeVector([5, 1, 2, 4, 6, -9, 7, 0, 333, 2, 2]) == [0.015000142502030658, 0.0030000285004061315, 0.006000057000812263, 0.012000114001624526, 0.01800017100243679, -0.027000256503655184, 0.02100019950284292, 0.0, 0.9990094906352418, 0.006000057000812263, 0.006000057000812263])
    test(normalizeVector([1, -1]) == [0.7071067811865475, -0.7071067811865475])
    test(normalizeVector([5, 600]) == [0.008333043996551019, 0.9999652795861221])
    test(abs(normalizeVector([6, 8])[0] - .6) < .0000001 )
    test(abs(normalizeVector([6, 8])[1] - .8) < .0000001 )
    test(normalizeVector([25,2,7,1,-5,12]) ==  [0.8585035246793065, 0.06868028197434452, 0.2403809869102058, 0.03434014098717226, -0.1717007049358613, 0.4120816918460671])

def colMean(m, col):
    """
    If col is valid return the mean of values in column col, else print "col out of bounds"
    and return.
    :param m: a matrix of numbers represented as a list of lists
    :param col: an integer that represents a valid column index of m
    :return: the float value that is the mean of the values in column col of m
    Example:
    >>> colMean([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2)
    4.0
    """
    try:
        result = 0
        for i in range(len(m)):
            result += m[i][col]
        return result/len(m)
    except: 
        print("col out of bounds")

def colMode(m, col):
    """
    If col is valid return the mode of values in col else print "col out of bounds" and return.
    :param m: a matrix of integers represented as a list of lists
    :param col: an integer that represents a valid column index of m
    :return: the integer value that is the mode of the values in column col of m
    Example:
    >>> colMode([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2)
    3
    """
    try:
        col_list = []
        count_list = []
        index = 0
        for i in range(len(m)): # make list of matrix column
            col_list += [m[i][col]] 
        for elem in col_list: # make list of column counts 
            count = 0
            for x in col_list:
                if elem == x:
                    count += 1
            count_list += [count]
        for i in range(len(count_list)): # find max item in list  
            if count_list[i] > count_list[index]:
                index = i
        if count_list[index] > 1:
            return col_list[index]
        else: 
            print("There is no mode.")
    except:
        print("col out of bounds")

def colSD(m, col): # my added function
    """
    Return the standard deviation of values in col.
    :param m: a matrix of integers represented as a list of lists
    :param col: an integer that represents a valid column index of m
    :return: the float value that is the standard deviation of the values in column 
    col of m
    Example:
    >>> colSD([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2)
    1.4142135623730951
    """
    sq_mean_list = []
    mean = colMean(m, col)
    for i in range(len(m)):
        sq_mean_list += [(m[i][col] - mean)**2]
    mean_of_sq = 0
    for n in sq_mean_list:
        mean_of_sq += n
    mean_of_sq /= len(sq_mean_list) - 1 # remove -1 to make it population sd
    sd = mean_of_sq**0.5
    return sd

def colStandardize(m, col): 
    """
    If col is valid return a new matrix identical to m except that the values in col are
    standardized, else print "col out of bounds" and return.
    :param m: a matrix of numbers represented as a list of lists
    :param col: an integer that represents a valid column index of m
    :return: a new matrix of the contents of m, with values in column col standardized
    Example:
    >>> colStandardize([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2)
    [[2, 4, 1.155], [1, 2, -0.577], [1, 2, -0.577]]
    """
    try:
        new_matrix = m
        mean = colMean(m, col)
        sd = colSD(m, col)
        for i in range(len(m)):
            new_matrix[i][col] = [(m[i][col] - mean) / sd]
        return new_matrix
            
    except:
        print("col out of bounds")

def colMax(m, col): # my added function
    """
    Return the maximum of values in col.
    :param m: a matrix of numbers represented as a list of lists
    :param col: an integer that represents a valid column index of m
    :return: the value which is the highest value in column col
    Example:
    >>> colMax([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2)
    6
    """
    num = m[0][col]
    for i in range(len(m)):
        if num < m[i][col]:
            num = m[i][col]
    return num

def colMin(m, col): # my added function
    """
    Return the minimum of values in col.
    :param m: a matrix of numbers represented as a list of lists
    :param col: an integer that represents a valid column index of m
    :return: the value which is the lowest value in column col
    Example:
    >>> colMax([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2)
    3
    """
    num = m[0][col]
    for i in range(len(m)):
        if num > m[i][col]:
            num = m[i][col]
    return num

def colMinMaxNormalize(m, col):
    """
    If col is valid return a new matrix identical to m except that the values in col are
    Normalized, else print "col out of bounds" and return.
    :param m: a matrix of numbers represented as a list of lists
    :param col: an integer that represents a valid column index of m
    :return: a new matrix of the contents of m with values in column col normalized between 0 and 1
    Example:
    >>> colMinMaxNormalize([[2, 4, 6], [1, 2, 3], [1, 2, 3]], 2)
    [[2, 4, 1], [1, 2, 0], [1, 2, 0]]
    """
    try:
        new_matrix = m
        col_min = colMin(m, col)
        col_max = colMax(m, col)
        col_range = col_max - col_min
        for i in range(len(m)):
            new_matrix[i][col] = int(((m[i][col]) - col_min) / col_range)
        return new_matrix
    except: 
        print("col out of bounds")

def mutation(dna, index, newNT):
    """
    If index is valid return a string with that represents a SNP (single nucleotide
    polymorphism) of dna, else print "index out of bounds" and return.
    :param dna: a string
    :param index: an integer such that 0 <= index < len(dna)
    :param newNT: a string to replace the character at index
    :return: a string composed of the characters of dna with the value at index replaced with newNT
    Example:
    >>> mutation("ACTCGG", 0, "G")
    "GCTCGG"
    """
    try:
        dna_list = []
        new_DNA = ''
        for ind in range(len(dna)):
            dna_list += [dna[ind]]
        dna_list[index] = newNT
        for i in dna_list:
            new_DNA += i
        return new_DNA
    except: 
        print("index out of bounds")

def insertion (dna, index, newNTs):
    """
    If index is valid return a string that represents an insertion mutation of dna,
    else print "index out of bounds" and return.
    :param dna: a string
    :param index: an integer such that 0 <= index <= len(dna)
    :param newNTs: a string to insert into dna at position index
    :return: a string composed of the characters of dna with the value at index replaced with newNT
    Examples:
    >>> insertion ("ACTCGG", 6, "AGC")
    "ACTCGGAGC"
    >>> insertion ("ACTCGG", 7, "AGC")
    "Index out of bounds"
    """
    try:
        dna_start = ''
        dna_rest = ''
        s = 0
        while s < index:
            dna_start += dna[s]
            s += 1
        while index < len(dna):
            dna_rest += dna[index]
            index += 1
        new_dna = dna_start + newNTs + dna_rest
        return new_dna
    except: 
        print("Index out of bounds")


def deletion(dna, index, numNTdeleted ):
    """
    If index is valid return a string that represents a deletion mutation of dna,
    else print "index out of bounds" and return.
    :param dna: a string
    :param index: an integer such that 0 <= index < len(dna)
    :param numNTdeleted: integer indicating how many characters to delete
    :return: a string composed of the characters of dna with up to numNTdeleted beginning at position index.
    Examples:
    >>> deletion("ACTCGG", 5, 2)
    "ACTCG"
    >>> deletion("ACTCGG", 1, 2)
    "ACGG”
    """
    new_dna = ''
    if len(dna) > index:
        for i in range(len(dna)):
            if i < index or i >= (index + numNTdeleted):
                new_dna += dna[i]
        return new_dna
    else:
            print("index out of bounds")

def euclideanDistance(v1, v2):
    """
    If vector is valid return the euclidean distance between vectors, else
    print "vector out of bounds" and return.
    :param v1: a vector of numbers represented as a list
    :param v2: a vector of numbers represented as a list
    :return: the float value that is the Euclidean distance between v1 and v2
    Examples:
    >>> euclideanDistance([3, 1], [6, 5])
    5.0
    >>> euclideanDistance([0, 0], [3, 4])
    5.0
    """
    try:
        vector_diff = []
        vector_sum = 0
        for i in range(len(v1)):
            vector_diff += [(v2[i]-v1[i])**2]
        for n in vector_diff:
            vector_sum += n
        return (vector_sum)**0.5
    except:
        print("index out of bounds")

def normalizeVector(v):
    """
    If vector is valid return a new vector that is vector v normalized, else 
    print "cannot normalize a zero vector" and return.
    :param v: a vector of numbers represented as a list
    :return: a new vector equivalent to v scaled to length 1 (ie: a unit vector)
    Example:
    >>> normalizeVector([6, 8])
    [0.6, 0.8]
    """
    try: 
        norm = 0
        result = []
        for n in v:
            norm += n**2
        norm = norm**0.5
        for n in v:
            result += [n/norm]
        return result
    except:
        print("cannot normalize a zero vector")

if __name__ == "__main__":
    test_suite()