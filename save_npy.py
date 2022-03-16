# -*- coding:utf-8 -*-
import numpy as np
import time
from collections import deque
import math
import re
import math


def ConvertELogStrToValue(eLogStr):
    """
    convert string of natural logarithm base of E to value
    return (convertOK, convertedValue)
    eg:
    input:  -1.1694737e-03
    output: -0.001169
    input:  8.9455025e-04
    output: 0.000895
    """

    (convertOK, convertedValue) = (False, 0.0)
    foundEPower = re.search("(?P<coefficientPart>-?\d+\.\d+)e(?P<ePowerPart>[-+]\d+)", eLogStr, re.I)
    # print "foundEPower=",foundEPower
    if (foundEPower):
        coefficientPart = foundEPower.group("coefficientPart")
        ePowerPart = foundEPower.group("ePowerPart")
        # print "coefficientPart=%s,ePower=%s"%(coefficientPart, ePower)
        coefficientValue = float(coefficientPart)
        ePowerValue = float(ePowerPart)
        # print "coefficientValue=%f,ePowerValue=%f"%(coefficientValue, ePowerValue)
        # math.e= 2.71828182846
        # wholeOrigValue = coefficientValue * math.pow(math.e, ePowerValue)
        wholeOrigValue = coefficientValue * math.pow(10, ePowerValue)

        # print "wholeOrigValue=",wholeOrigValue;

        (convertOK, convertedValue) = (True, wholeOrigValue)
    else:
        (convertOK, convertedValue) = (False, 0.0)

    return (convertOK, convertedValue)


def save_npy(file_path,save_file):
    li = []
    with open(file_path, 'r') as f:
        for line in f:
            li.append(ConvertELogStrToValue(line.strip())[1])
    np.save(save_file, np.array(li))
def save_npy2(file_path,save_file):
    li = []
    with open(file_path, 'r') as f:
        for line in f:
            li.append(float(line.strip()))
    np.save(save_file, np.array(li))
if __name__ == "__main__":
    save_npy2('./data/Query2.txt','./data/query_cj2.npy')
    # save_npy('Data_new.txt','./data/data_cj1.npy')