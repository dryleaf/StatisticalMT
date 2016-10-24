#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Josimar H. Lopes,
# Master of EEIE, NUT
#from __future__ import division
import math
import codecs
import sys
import collections
from decimal import Decimal, getcontext, ROUND_HALF_UP


# defines decimal precision and roundings
getcontext().prec = 4
getcontext().rounding = ROUND_HALF_UP

class IBModel2:
    """Class to implement IBM model 2"""

    def __init__(self):
        """Initialize Variables"""
        filePointer = open(sys.argv[1]) if len(sys.argv) >= 2 else sys.stdin
        sentencePairs = [line.strip().split('|') for line in filePointer.readlines()]
        self.trainingCorpus = [(f.split(), e.split()) for (f, e) in sentencePairs]
        print "Sentence Pairs\n", sentencePairs
        print "Word Pairs\n", self.trainingCorpus
        self.e_a = list()


    def initializationIBModel1(self):
        """Method to Initialize the parameter estimation for IBM model 2"""

        # number of english words in trainingCorpus
        e_words = set()
        for (fs, es) in self.trainingCorpus:
            for e in es:
                e_words.add(e)

        # initialize with uniform random distribution
        t = collections.defaultdict(lambda: Decimal(Decimal(1)/Decimal(len(e_words))))
        q = collections.defaultdict(Decimal)
        delta = collections.defaultdict(Decimal)

        # EM IBM model 1 algorithm
        for iterations in range(3):  # usually requires less iterations for convergence depends on the size of corpus
            count_ef = collections.defaultdict(Decimal)
            count_e = collections.defaultdict(Decimal)
            count_ji = collections.defaultdict(Decimal)
            count_i = collections.defaultdict(Decimal)
            total_t = collections.defaultdict(Decimal)

            k = 0
            for (fs, es) in self.trainingCorpus:
                m = len(fs)
                l = len(es)
                for i in range(m):
                    for j in range(-1, l):  # include NULL at position -1
                        #compute total_t for delta
                        total_t[i] = Decimal()
                        for lj in range(-1, l):  # include NULL at position -1
                            total_t[i] += t[(fs[i], 'NULL' if lj == -1 else es[lj])]  # to include null at position -1
                        delta[k, i, j] = t[(fs[i], 'NULL' if j == -1 else es[j])] / total_t[i]
                        count_ef[('NULL' if j == -1 else es[j], fs[i])] += delta[(k, i, j)]
                        count_e[('NULL' if j == -1 else es[j])] += delta[(k, i, j)]
                        count_ji[(j, i, l, m)] += delta[(k, i, j)]
                        count_i[(i, l, m)] += delta[(k, i, j)]
                k += 1

            # parameter estimation for t and q
            for (e, f) in count_ef.keys():
                t[(f, e)] = count_ef[(e, f)] / count_e[e]

            for (j, i, l, m) in count_ji.keys():
                q[(j, i, l, m)] = count_ji[(j, i, l, m)] / count_i[(i, l, m)]


        return t, q

    def key_fun(key):
        ''' default_factory function for keydefaultdict '''
        i, j, l_e, l_f = key
        return Decimal("1") / Decimal(l_f + 1)

    def emIBModel2(self, t, q):
        """Estimation Maximization algorithm of IBM model 2"""

        delta = collections.defaultdict(Decimal)

        # EM IBM model 2 algorithm
        for iterations in range(2):  # usually requires less iterations for convergence depends on the size of corpus
            count_ef = collections.defaultdict(Decimal)
            count_e = collections.defaultdict(Decimal)
            count_ji = collections.defaultdict(Decimal)
            count_i = collections.defaultdict(Decimal)
            total_t = collections.defaultdict(Decimal)

            k = 0
            for (fs, es) in self.trainingCorpus:
                m = len(fs)
                l = len(es)
                for i in range(m):
                    for j in range(-1, l):  # include NULL at position -1
                        #compute total_t for delta
                        total_t[i] = Decimal()
                        for lj in range(-1, l):  # include NULL at position -1
                            total_t[i] += q[(lj, i, l, m)] * t[(fs[i], 'NULL' if lj == -1 else es[lj])]  # include NULL at position -1
                        delta[k, i, j] = (q[(j, i, l, m)] * t[(fs[i], 'NULL' if j == -1 else es[j])]) / total_t[i]
                        count_ef[('NULL' if j == -1 else es[j], fs[i])] += delta[(k, i, j)]
                        count_e[('NULL' if j == -1 else es[j])] += delta[(k, i, j)]
                        count_ji[(j, i, l, m)] += delta[(k, i, j)]
                        count_i[(i, l, m)] += delta[(k, i, j)]
                k += 1

            # parameter estimation for t and q
            for (e, f) in count_ef.keys():
                t[(f, e)] = count_ef[(e, f)] / count_e[e]

            for (j, i, l, m) in count_ji.keys():
                q[(j, i, l, m)] = count_ji[(j, i, l, m)] / count_i[(i, l, m)]

        return t, q

    def recoveringAlignments(self, t, q):
        """Given sentence pairs, we can recover the most likely alignments."""

        print "\n\nRECOVERING ALIGNMENTS\n"

        esents = "the blue house"
        fsents = "a mansão azul"

        tp = collections.defaultdict(Decimal)

        for (fs, es) in self.trainingCorpus:  # only 1 example
            a = collections.defaultdict(Decimal)
            l = len(es)
            m = len(fs)
            for i in range(m):
                temp_qt = 0
                pair = ""
                save_i = 0
                save_j = 0
                for j in range(-1, l):  # include NULL at position -1
                    qt = q[(j, i, l, m)] * t[(fs[i], 'NULL' if j == -1 else es[j])]
                    if temp_qt < qt:
                        temp_qt = qt
                        pair = 'NULL' if j == -1 else es[j]  # include NULL at position -1
                        save_j = j
                        save_i = i
                a[(save_i, fs[i], save_j, pair, m)] = temp_qt
                tp[(fs[i], save_j, pair, m)] = temp_qt

            alignments = ""
            for (i, f, j, e, m), val in sorted(a.iteritems(), key=lambda (k, v): k[0], reverse=False):
                print("{}|{}, {}|{}, {}={}    ".format(i+1, f, j+1, e, m, val)),
                alignments += f + '/' + e + ' '
            print"\n", alignments, "\n"

        print "TRANSLATION PROBABILITIES"
        for (f, j, e, m), val in tp.items():
                print("({}, {}|{}, m={}) = {}    ".format(f, j+1, e, m, val))

        return tp


    def displayParameters(self, t, q):
        """Prints Estimated Parameters t and q"""

        print "\n\nProb. of Alignments/distortion q(j|i,l,m):"
        for (j, i, l, m), val in q.items():
            print("({} | {}, {}, {}\t{})".format(j+1, i+1, l, m, val))

        print "\nTranslation Probabilities t(f|e):"
        for (f, e), val in t.items():
            print("{} {}\t{}".format(f, e, val))
        """
        print >>outfile, "Lexical translation parameter values"
        for (f, e), val in t.items():
            print >>outfile, "%s\t%s\t%f" % (f, e, val)

        print >>outfile, "\n%s" % ("="*40)
        for (j, i, l, m), val in q.items():
            print >>outfile, "tgtpos=%d\tsrcpos=%d\ttgtlen=%d\tsrclen=%d\t%f" % (j+1, i+1, l, m, val)
        outfile.close()"""

    def languageModel(self):
        """Calculates the Language Model for bi-grams"""

        lm = collections.defaultdict(Decimal)
        count_ij = collections.defaultdict(Decimal)
        counti = collections.defaultdict(Decimal)

        for (f, e) in self.trainingCorpus:
            prev = "<s>"
            for w in e:
                count_ij[(prev, w)] += 1
                counti[prev] += 1
                prev = w
            count_ij[(prev, "</s>")] += 1
            counti[prev] += 1

        print "\nLanguage Model"
        for (wi_1, wi), val in count_ij.items():
            lm[(wi, wi_1)] = val/counti[wi_1]
            print "({}|{}) = {}".format(wi, wi_1, lm[(wi, wi_1)])

        return lm

    def decoding(self, lm, tp):
        """Decoder uses the noise-channel approach to decode from Portuguese to English"""
        print "DECODING"

        inp = open('input.txt', 'r')

        k = 0
        for f in inp:
            prev = '<s>'
            print "f = ", f
            M = len(f.split())
            self.e_a.append([])
            ppos = -1  # setting for '<s>' instances in language model
            j = 0
            for w in f.split():
                temp_tp = max((value, ew, a, m) for (fw, a, ew, m), value in tp.iteritems() if w == fw and M == m)
                print "\ntp = ({}, {}, {}, m={})".format(temp_tp[0], temp_tp[1], temp_tp[2], temp_tp[3])
                temp_lm = max((value, ewi, ewi_1) for (ewi, ewi_1), value in lm.iteritems() if ewi_1 == prev)
                print "lm = ({}, {})={}".format(temp_lm[1], temp_lm[2], temp_lm[0])

                if temp_lm[0] * temp_tp[0] > 0 and temp_tp[1] != prev:
                    #if ppos == temp_tp[2]:
                    self.e_a[k].insert(temp_tp[2], temp_tp[1])
                    #else:
                    #    self.e_a[k].insert(temp_tp[2], temp_tp[1])

                ppos = temp_tp[2]
                prev = temp_tp[1]
                j += 1
            k += 1
        inp.close()

        out = open('decoded.txt', 'w')
        for s in self.e_a:
            for w in s:
                print >> out, "{}".format(w),
            print >> out
        out.close()

    def evaluation(self):
        """Evaluates System translated vs Reference translations"""

        dec = open('decoded.txt', 'r')
        ref = open('reference.txt', 'r')
        dec_input = [sent.split() for sent in dec.readlines()]
        ref_input = [sent.split() for sent in ref.readlines()]


        precision = []
        recall = []
        fmeasure = []

        ref_length = len(ref_input)
        dec_length = len(dec_input)

        for i in range(ref_length):
            r_row_length = len(ref_input[i])
            d_row_length = len(dec_input[i])
            correct = 0
            for j in range(r_row_length):
                if ref_input[i][j] == dec_input[i][j]:
                    correct += 1

            temp_p = correct / float(d_row_length)
            temp_r = correct / float(r_row_length)
            print "cor = ", correct, "len_ref = ", r_row_length, "len_dec = ", d_row_length
            print "precision = ", temp_p, "recall = ", temp_r

            precision.append(temp_p)
            recall.append(temp_r)
            fmeasure.append((temp_p * temp_r) / float((temp_p + temp_r) / 2))

        print sum(precision)/len(precision)
        print sum(recall)/len(recall)
        print sum(fmeasure)/len(recall)

        dec.close()
        ref.close()







if __name__ == '__main__':
    ibmodel2 = IBModel2()
    t, q = ibmodel2.initializationIBModel1()
    t, q = ibmodel2.emIBModel2(t, q)
    ibmodel2.displayParameters(t, q)
    tp = ibmodel2.recoveringAlignments(t, q)
    lm = ibmodel2.languageModel()
    ibmodel2.decoding(lm, tp)
    ibmodel2.evaluation()
