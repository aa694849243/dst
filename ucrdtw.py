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


import time
from collections import deque


class UCR_DTW(object):

    def __init__(self, data_file=None, query_file=None, m=128, R=0.05):
        self.fp = open(data_file, 'r')  # data file path
        self.qp = open(query_file, 'r')  # query file path
        self.result = open("lb_kim.result", 'w')
        self.bsf = float('inf')  # best-so-far
        self.m = m  # the size of the query
        self.t = [None] * (self.m * 2)  # candidate C sequence
        self.q = [0] * self.m  # query array
        self.order = [None] * self.m  # new order of the query
        self.u, self.l = [None] * self.m, [None] * self.m  # LB_Keogh Upper,Lower bound
        self.qo = [None] * self.m
        self.uo = [None] * self.m
        self.lo = [None] * self.m
        self.tz = [None] * self.m  # ???????????????C
        self.cb = [0] * self.m  # the cumulative lower bound
        self.cb1 = [0] * self.m  # the cumulative lower bound
        self.cb2 = [0] * self.m  # the cumulative lower bound
        self.u_d = [None] * self.m
        self.l_d = [None] * self.m

        """
        for every EPOCH points, all cummulative values, such as ex(sum),ex2,
        will be restarted for reducing the floating point error
        """
        self.EPOCH = 100000  # for ???flush out??? any accumulated floating error

        self.d = None  # distance
        self.ex, self.ex2, self.mean, self.std = 0.0, 0.0, 0.0, 0.0
        self.r = int(self.m * R)  # LB_Keogh window,warping windows for LB_Keogh lower bound
        self.loc = 0
        self.t1, self.t2 = None, None  # start clock,end clock for time cost calculation
        self.kim = 0
        self.keogh = 0  # LB_Keogh_EQ
        self.keogh2 = 0  # LB_Keogh_EC
        self.buffer = [0.0] * self.EPOCH
        self.u_buff = [0.0] * self.EPOCH
        self.l_buff = [0.0] * self.EPOCH

    def print_result(self, i):
        print("Location: ", self.loc)
        print("Distance: ", math.sqrt(self.bsf))
        print("Data Scanned: ", i)
        print("Total Execution Time: ", (self.t2 - self.t1), " sec")
        print("Pruned by LB_Kim   : %.2f" % ((self.kim / i) * 100), '%')
        print("Pruned by LB_Keogh : %.2f" % ((self.keogh / i) * 100), '%')
        print("Pruned by LB_Keogh2: %.2f" % ((self.keogh2 / i) * 100), '%')
        print("DTW Calculation    : %.2f" % ((100 - self.kim + self.keogh + self.keogh2) / i * 100), '%')

    def main_run(self):
        self.t1 = time.time()  # start the clock

        self.Q_normalize()
        # ?????????????????????Q?????????lower bound???create envelop of the query,LB_Keogh_EQ self.l and self.u
        self.lower_upper_lemire(self.q, self.m, self.r, self.l, self.u)

        self.sort_query_order()

        i = 0  # current index of the data in current chunk of size EPOCH
        j = 0  # the starting index of the data in the circular array, t
        it = 0  # ????????????self.buffer??????????????????self.EPOCH,????????????????????????1??????????????????0
        done = False
        while not done:
            # read buffer of size EPOCH or when all data has been read
            ep = self.buffer_init(it)

            # data are read in chunk of size EPOCH
            # when there is nothing to read, the loop is end
            if ep <= self.m - 1:
                # ??????data file????????????????????????m-1???????????????????????????????????????????????????
                break

            # ?????????buffer???????????????normalization??????(????????????lb_keogh_data_cumulative??????????????????????????????)?????????????????????chunk??????lower bound
            self.lower_upper_lemire(self.buffer, ep, self.r, self.l_buff, self.u_buff)  # LB_Keogh_EC lower bound??????

            # ????????????online z-normalize
            ex, ex2, mean, std = 0.0, 0.0, 0.0, 0.0
            for i in range(ep):
                # A bunch of data has been read and pick one of them at a time to use
                d = self.buffer[i]
                # Calcualte sum and sum square
                ex += d
                ex2 += d * d

                # t is a circular array for keeping current data
                self.t[i % self.m] = d
                # double the size for avoiding using module "%" operator
                self.t[(i % self.m) + self.m] = d
                if i < self.m - 1:
                    continue

                # start the task when there are more than m-1 points in the current chunk
                # ??????(5)
                mean, std = self.Compute_Mean_Std(ex, ex2)

                # ????????????????????????????????????
                # compute the start location of the data in the current circular array,self.t??????????????????
                j = (i + 1) % self.m
                # the start location of the data in the current chunk??????self.buffer????????????
                I = i - (self.m - 1)

                # LB_KimFL
                lb_kim = self.lb_kim_hierarchy(self.t, self.q, j, self.m, mean, std, self.bsf)
                print("lb_kim:%f best_so_far:%f" % (lb_kim, self.bsf), file=self.result)
                # ??????lower bound??????
                if lb_kim < self.bsf:
                    # LB_Keogh_EQ
                    lb_k = self.lb_keogh_cumulative(self.order, self.t, self.uo, self.lo, self.cb1, j, self.m, mean,
                                                    std, self.bsf)
                    print("lb_k:%f best_so_far:%f" % (lb_k, self.bsf), file=self.result)
                    if lb_k < self.bsf:
                        for k in range(self.m):
                            # z-normalization of t will be computed on the fly
                            self.tz[k] = (self.t[(k + j)] - mean) / std  # ????????????C???????????????????????????

                        # LB_Keogh_EC
                        lb_k2 = self.lb_keogh_data_cumulative(self.order, self.tz, self.qo, self.cb2, self.l_buff[I:],
                                                              self.u_buff[I:], self.m, mean, std, self.bsf)
                        print("lb_k2:%f best_so_far:%f" % (lb_k2, self.bsf), file=self.result)
                        if lb_k2 < self.bsf:
                            # choose better lower bound between lb_keogh and lb_keogh2 to be used in early abandoning DTW
                            # Note that cb and cb2 will be cumulative summed here
                            self.LB_Keogh_for_abandon_DTW(lb_k, lb_k2)
                            self.Early_Abandon_DTW(it, i)
                        else:
                            self.keogh2 += 1
                    else:
                        self.keogh += 1
                else:
                    self.kim += 1

                # reduce obsolute points from sum and sum square,??????(5)????????????
                ex -= self.t[j]
                ex2 -= self.t[j] * self.t[j]
            if ep < self.EPOCH:
                done = True
            else:
                it += 1
            print("#" * 20, it, ep, "#" * 20)

        i = it * (self.EPOCH - self.m + 1) + ep
        self.fp.close()
        self.result.close()
        self.t2 = time.time()
        self.print_result(i)

    def buffer_init(self, it):
        # read first m-1 points
        if it == 0:
            # ???data file??????m-1???????????????buffer
            for k in range(self.m - 1):
                try:
                    line = self.line_to_float(next(self.fp))
                    self.buffer[k] = line
                except:
                    break
        else:
            for k in range(self.m - 1):
                # ????????????m-1??????????????????buffer?????????
                self.buffer[k] = self.buffer[self.EPOCH - self.m + 1 + k]

        # fill the rest of self.buffer
        # read buffer of size EPOCH or when all data has been read
        ep = self.m - 1
        while ep < self.EPOCH:
            try:
                line = self.line_to_float(next(self.fp))  # ???data file??????????????????buffer???
                self.buffer[ep] = line
            except:
                break
            ep += 1
        return ep

    def Compute_Mean_Std(self, ex, ex2):
        mean = ex / self.m
        std = ex2 / self.m
        std = math.sqrt(std - mean * mean)
        return mean, std

    def Early_Abandon_DTW(self, it, i):
        # compute DTW and early abandoning if possible
        dist = self.dtw(self.tz, self.q, self.cb, self.m, self.r, self.bsf)
        print("dtw-dist:%f best_so_far:%f" % (dist, self.bsf), file=self.result)
        if dist < self.bsf:
            # update bsf
            # loc is the real starting locatin of the nearest neighbor in the file
            self.bsf = dist
            self.loc = it * (self.EPOCH - self.m + 1) + i - self.m + 1

    def LB_Keogh_for_abandon_DTW(self, lb_k, lb_k2):
        # ????????????LB_Keogh?????????early abandoning DTW
        if lb_k > lb_k2:
            self.cb[self.m - 1] = self.cb1[self.m - 1]
            for k in range(self.m - 2, -1, -1):
                self.cb[k] = self.cb[k + 1] + self.cb1[k]
        else:
            self.cb[self.m - 1] = self.cb2[self.m - 1]
            for k in range(self.m - 2, -1, -1):
                self.cb[k] = self.cb[k + 1] + self.cb2[k]

    def line_to_float(self, line):
        # return ConvertELogStrToValue(line.strip())[1]
        return float(line.strip())
    def sort_query_order(self):
        """
        ??????????????????Q?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        """
        Q_tmp = {}
        # ????????????????????????????????????order
        # sort the query one time by abs(z-norm(q[i]))
        for i in range(self.m):
            Q_tmp[i] = abs(self.q[i])
        sorted_index = [item[0] for item in sorted(Q_tmp.items(), key=lambda x: x[1], reverse=True)]
        # also create another arrays for keeping sorted envelop
        self.order = sorted_index
        for i, ind in enumerate(sorted_index):
            self.qo[i] = self.q[ind]
            self.uo[i] = self.u[ind]
            self.lo[i] = self.l[ind]

    def Q_normalize(self):
        """
        ???Query?????????????????????
        """
        i = 0
        ex = 0.0
        ex2 = 0.0
        while i < self.m:
            line = self.line_to_float(next(self.qp))
            ex += line
            ex2 += line ** 2
            self.q[i] = line
            i += 1

        # [Query z-normalize] Do z-normalize the query, keep in same array, self.q
        mean, std = self.Compute_Mean_Std(ex, ex2)
        # print("[Q_normalize]ex=%f,ex2=%f,mean=%f,std=%f"%(ex,ex2,mean,std),file=self.result)
        for i in range(self.m):
            old = self.q[i]
            self.q[i] = (self.q[i] - mean) / std
            # print("old_q[i]=%f;q[%d]=%f"%(old,i,self.q[i]),file=self.result)
        self.qp.close()

    def lower_upper_lemire(self, t, lenght, r, l, u):
        """
        Finding the envelop of min and max value for LB_Keogh_EQ
        Implementation idea is introducted by Danial Lemire in his paper
        "Faster Retrieval with a Two-Pass Dynamic-Time-Warping Lower Bound",Pattern Recognition 42(9) 2009
        ?????????
            t: the query
            lenght: query lenght
            r: window
            l: lower
            u: upper
        """
        for ind in range(lenght):
            l[ind] = min(t[(ind - r if ind - r >= 0 else 0):(ind + r + 1 if ind + r + 1 < lenght else lenght)])
            u[ind] = max(t[(ind - r if ind - r >= 0 else 0):(ind + r + 1 if ind + r + 1 < lenght else lenght)])

    def lb_kim_hierarchy(self, t, q, j, lenght, mean, std, bsf=float('inf')):
        """
        Calculate quick lower bound
        Usually, LB_Kim take time O(m) for finding top,bottom,first and last
        however, because of z-normalization the top and bottom cannot give significant benefits
        and using the first and last points can be computed in constant time
        the prunning power of LB_Kim is non-trivial,especially when the query is not long,say in lenght 128
        ?????????
            t: C-unnormalized
            q: Q-normalized
            j: t?????????????????????
            lenght: q?????????
            mean: ??????
            std: ??????
            bsf: best-so-far
        """
        # 1 point at front and back
        x0 = (t[j] - mean) / std  # ????????????????????????????????????
        y0 = (t[(lenght - 1 + j)] - mean) / std  # ???????????????????????????????????????
        # ?????????first???last?????????????????????????????????
        lb = self.dist(x0, q[0]) + self.dist(y0, q[lenght - 1])
        if lb >= bsf: return lb

        # 2 points at front
        x1 = (t[j + 1] - mean) / std
        d = min(self.dist(x1, q[0]), self.dist(x0, q[1]))
        d = min(d, self.dist(x1, q[1]))
        lb += d
        if lb >= bsf: return lb

        # 2 points at back
        y1 = (t[(lenght - 2 + j)] - mean) / std
        d = min(self.dist(y1, q[lenght - 1]), self.dist(y0, q[lenght - 2]))
        d = min(d, self.dist(y1, q[lenght - 2]))
        lb += d
        if lb >= bsf: return lb

        # 3 points at front
        x2 = (t[(j + 2)] - mean) / std
        d = min(self.dist(x0, q[2]), self.dist(x1, q[2]))
        d = min(d, self.dist(x2, q[2]))
        d = min(d, self.dist(x2, q[1]))
        d = min(d, self.dist(x2, q[0]))
        lb += d
        if lb >= bsf: return lb

        # 3 points at back
        y2 = (t[(lenght - 3 + j)] - mean) / std
        d = min(self.dist(y0, q[lenght - 3]), self.dist(y1, q[lenght - 3]))
        d = min(d, self.dist(y2, q[lenght - 3]))
        d = min(d, self.dist(y2, q[lenght - 2]))
        d = min(d, self.dist(y2, q[lenght - 1]))
        lb += d

        return lb

    def lb_keogh_cumulative(self, order, t, uo, lo, cb, j, lenght, mean, std, best_so_far=float('inf')):
        """
        LB_Keogh 1: Create Envelop for the query
        Note that because the query is known, envelop can be created once at the begenining.

        ?????????
            order: sorted indices for the query
            uo,lo: upper and lower envelops for the query,which already sorted
            t    : C (candidate subsequence) a circular array keeping the current data
            j    : index of the starting location in t
            cb   : (output)current bound at each position.It will be used later for early abandoning in DTW
        """
        lb = 0
        print("lb_keogh_cumulative:mean=%f;std=%f" % (mean, std), file=self.result)

        for i in range(lenght):
            if lb >= best_so_far:
                break

            x = (t[order[i] + j] - mean) / std  # ???????????????
            d = 0
            # ????????????????????????Q???lower bound???????????????
            if x > uo[i]:
                d = self.dist(x, uo[i])
            elif x < lo[i]:
                d = self.dist(x, lo[i])
            lb += d
            cb[order[i]] = d  # ????????????????????????????????????????????????Early Abandoning of DTW ??????
            print(
                "lb_keogh_cumulative:j=%d;order[i]=%d,q[order[i]]=%f;x=%f;d=%f" % (j, order[i], self.q[order[i]], x, d),
                file=self.result)
        return lb

    def lb_keogh_data_cumulative(self, order, tz, qo, cb, l, u, lenght, mean, std, best_so_far=float('inf')):
        '''
        LB_Keogh 2: Create Envelop for the data
        Note that the envelops have bean created (in main function) when each data point has been read.

        ??????:
            order: ?????????????????????Q????????????????????????index??????
            tz:Z-normalized C
            qo:sorted query
            cb:???cb2???current bound at each position.Used later for early abandoning in DTW
            l,u:lower and upper envelop of the current data(l_buff,u_buff)
        '''
        lb = 0
        for i in range(lenght):
            if lb >= best_so_far:
                break
            uu = (u[order[i]] - mean) / std  # z-normalization??????lower bound?????????????????????(???????????????chunk?????????????????????????????????)
            ll = (l[order[i]] - mean) / std
            d = 0
            if qo[i] > uu:
                d = self.dist(qo[i], uu)
            elif qo[i] < ll:
                d = self.dist(qo[i], ll)
            lb += d
            cb[order[i]] = d
        return lb

    def dtw(self, A, B, cb, m, r, bsf=float('inf')):
        """
        Calculate Dynamic Time Wrapping distance
        A,B: data and query
        cb: cummulative bound used for early abandoning
        r: size of Sakoe-Chiba warpping band
        """
        x, y, z, min_cost = 0.0, 0.0, 0.0, 0.0

        # instead of using matrix of size O(m^2) or O(mr),we will reuse two array of size O(r)
        cost = [float('inf')] * (2 * r + 1)
        cost_prev = [float('inf')] * (2 * r + 1)
        for i in range(m):
            k = max(0, r - i)
            min_cost = float('inf')

            for j in range(max(0, i - r), min(m - 1, i + r) + 1):
                # Initialize all row and column
                if i == 0 and j == 0:
                    cost[k] = self.dist(A[0], B[0])
                    min_cost = cost[k]
                    k += 1
                    continue

                if j - 1 < 0 or k - 1 < 0:
                    y = float('inf')
                else:
                    y = cost[k - 1]
                if i - 1 < 0 or k + 1 > 2 * r:
                    x = float('inf')
                else:
                    x = cost_prev[k + 1]
                if i - 1 < 0 or j - 1 < 0:
                    z = float('inf')
                else:
                    z = cost_prev[k]

                # Classic DTW calculation
                cost[k] = min(min(x, y), z) + self.dist(A[i], B[j])
                # Find minimum cost in row for early abandoning(possibly to use column instead of row)
                if cost[k] < min_cost:
                    min_cost = cost[k]
                k += 1

            # we can abandon early if the current cummulative distance with lower bound together are larger than bsf
            if i + r < m - 1 and min_cost + cb[i + r + 1] >= bsf:
                return min_cost + cb[i + r + 1]

            # Move current array to previous array
            cost, cost_prev = cost_prev, cost
        k -= 1

        # the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array
        final_dtw = cost_prev[k]
        return final_dtw

    def dist(self, x, y):
        return (x - y) ** 2


if __name__ == "__main__":
    model = UCR_DTW('Data_new.txt', './data/Query2.txt')
    model.main_run()
