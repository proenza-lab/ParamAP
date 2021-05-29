#!/usr/bin/env python3

'''
Copyright 2021 The Regents of the University of Colorado

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

Author:     Christian Rickert <christian.rickert@cuanschutz.edu>
Title:      ParamAP (parametrization of sinoatrial myocyte action potentials)

DOI:        https://doi.org/10.5281/zenodo.823742
URL:        https://github.com/christianrickert/ParamAP/
'''


# imports

# runtime
import fnmatch
import functools
import gc
import math
import os
import sys

# numpy
import numpy as np

# scipy
import scipy.signal as sp_sig
import scipy.spatial as sp_spat
import scipy.stats as sp_stat

# matplotlib
import matplotlib.backends.backend_pdf as mpbp
import matplotlib.pyplot as mpp


# variables

APEXT = 0.5  # margin of ap extension (%)
FFRAME = 0.5  # time interval for filtering (ms)
POLYNOM = 2  # polynomial order used for filtering


# functions

def askboolean(dlabel="custom boolean", dval=True):
    """Returns a boolean provided by the user."""
    if dval:  # True
        dstr = "Y/n"
    else:  # False
        dstr = "y/N"
    while True:
        uchoice = input(dlabel + " [" + dstr + "]: ") or dstr
        if uchoice in ["Y/n", "Y", "y", "Yes", "yes"]:
            print("True\n")
            return True  # break
        elif uchoice in ["y/N", "N", "n", "No", "no"]:
            print("False\n")
            return False  # break
        else:
            continue


def askext(dlabel="custom extension", dext='atf'):
    """Returns a file extention provided by the user."""
    while True:
        uext = str(input("Enter " + dlabel + " [" + dext + "]: ")).lower() or dext
        if uext not in ["dat", "log", "pdf"] and len(uext) == 3:
            print(uext + "\n")
            return uext  # break
        else:
            print("Invalid file extension!\n")
            continue


def askunit(dlabel="custom unit", daxis='', dunit=''):
    """Returns a unit provided by the user."""
    while True:
        uunit = input("Enter " + dlabel + " [" + dunit + "]: ") or dunit
        if daxis in ["x", "X"]:
            if uunit in ["ms", "s"]:
                print(uunit + "\n")
                return uunit  # break
            else:
                print("Invalid unit for X-axis!\n")
                continue
        elif daxis in ["y", "Y"]:
            if uunit in ["mV", "V"]:
                print(uunit + "\n")
                return uunit  # break
            else:
                print("Invalid unit for Y-axis!\n")
                continue


def askvalue(dlabel="custom value", dval=1.0, dunit="", dtype="float"):
    """Returns a value provided by the user."""
    while True:
        try:
            uval = float(input("Enter " + dlabel + " [" + str(dval) + "]" + dunit + ": ") or dval)
            break
        except ValueError:
            print("Non-numerical input!\n")
            continue
    if dtype == "float":  # default
        pass
    elif dtype == "int":
        uval = int(round(uval))
    print(str(uval) + "\n")
    return uval


def getfiles(path='/home/user/', pattern='*'):
    """Returns all files in path matching the pattern."""
    abspath = os.path.abspath(path)
    for fileobject in os.listdir(abspath):
        filename = os.path.join(abspath, fileobject)
        if os.path.isfile(filename) and fnmatch.fnmatchcase(fileobject, pattern):
            yield os.path.join(abspath, filename)


def getneighbors(origin_i=np.empty(0), vicinity=np.empty(0), origin_x=np.empty(0), origin_y=np.empty(0), hwidth=float("inf"), fheight=0.0, limit=None, within=float("inf"), bads=False):
    """Returns all nearest-neighbors in ascending (i.e. increasing distance) order."""
    neighbors = np.zeros(0)
    badorigins = np.zeros(0)
    vicinity_kdt = sp_spat.KDTree(list(zip(vicinity, np.zeros(vicinity.size))))  # KDTree for the nearest neighbors search
    for origin in origin_i:
        neighbor_left, neighbor_right = False, False
        for position in vicinity_kdt.query([origin, 0.0], k=limit, distance_upper_bound=within)[1]:  # return nearest neighbors in ascending order
            if not neighbor_left or not neighbor_right:
                neighbor = vicinity[position]
                if (abs(origin_x[origin]-origin_x[neighbor]) <= hwidth) and (abs(origin_y[origin]-origin_y[neighbor]) >= fheight):  # relative criteria for minima left and right of maximum
                    if not neighbor_left and (neighbor < origin):  # criteria for minimum left of maximum only
                        neighbors = np.append(neighbors, neighbor)
                        neighbor_left = True
                    elif not neighbor_right and (neighbor > origin):  # criteria for minimum right of maximum only
                        neighbors = np.append(neighbors, neighbor)
                        neighbor_right = True
                else:  # odd origins with missing neighbors
                    badorigins = np.append(badorigins, np.argwhere(origin == origin_i))
    neighbors = np.sort(np.unique(neighbors))  # unique elements only
    if neighbors.size <= 1:  # missing neighbor
        if neighbor_left:
            neighbors = np.append(neighbors, 0.0)  # append neighbor_right
        if neighbor_right:
            neighbors = np.insert(neighbors, 0, 0.0)  # insert neighbor_left
    badorigins = np.sort(np.unique(badorigins))
    return (neighbors.astype(int), badorigins.astype(int)) if bads else (neighbors.astype(int))


def getrunavg(xdata=np.empty(0), xinterval=FFRAME, porder=POLYNOM):
    """Returns the running average count based on a given time interval."""
    tmprun = int(round(xinterval/(xdata[1]-xdata[0])))
    while tmprun <= porder:  # prevents filtering
        tmprun += 1
    return (tmprun) if tmprun % 2 else (tmprun + 1)  # odd number


def getpartitions(pstart=0, pstop=100, pint=5, pmin=10):
    """Returns a partition list in percent to segment an interval."""
    plist = []
    for part_l in list(range(int(pstart), int(pstop)+int(pint), int(pint))):
        for part_r in list(range(int(pstart), int(pstop)+int(pint), int(pint))):
            if part_r > part_l and part_r-part_l >= int(pmin):  # no duplication or empty partitions, minimum size
                plist.append([part_l, part_r])  # return statement removes the outmost list
    return plist


def getbestlinearfit(xaxis=np.empty(0), yaxis=np.empty(0), xmin=0.0, xmax=1.0, pstart=0, pstop=100, pint=1, pmin=10):
    """Returns the best linear fit from segments of an interval."""
    bst_r = 0  # regression coefficient
    seg_i = np.argwhere((xaxis >= xmin) & (xaxis <= xmax)).ravel()  # analyzing partial segment only
    seg_t = xaxis[seg_i[-1]]-xaxis[seg_i[0]]  # full interval from partial segment
    seg_m, seg_n, seg_r = 0.0, 0.0, 0.0
    for partition in getpartitions(pstart, pstop, pint, pmin):
        seg_i = np.argwhere((xaxis >= (avgfmin_x[0]+(seg_t*partition[0]/100))) & (xaxis <= (avgfmin_x[0]+(seg_t*partition[1]/100)))).ravel()  # 'ravel()' required for 'sp_stat.linregress()'
        seg_x = xaxis[seg_i]
        seg_y = yaxis[seg_i]
        seg_m, seg_n, seg_r = sp_stat.linregress(seg_x, seg_y)[0:3]  # tuple unpacking and linear regression of partial ap segment
        if math.pow(seg_r, 2.0) >= math.pow(bst_r, 2.0):
            bst_m, bst_n, bst_r = seg_m, seg_n, seg_r
            bst_i, bst_x, bst_y = seg_i, seg_x, seg_y
            # print(partition[0], " - ", partition[1], " : ", str(partition[1]-partition[0]), " ~ ", str(math.pow(bst_r, 2.0)))  # shows progress, but is slow!
    return (bst_i, bst_x, bst_y, bst_m, bst_n, bst_r)


def mpp_setup(title='Plot title', xlabel='Time [ ]', ylabel='Voltage [ ]'):
    """Provides a title and axes labels to a Matplotlib plot."""
    mpp.title(title)
    mpp.xlabel(xlabel)
    mpp.ylabel(ylabel)


def readfile(inputfile='name'):
    """ read an atf file into a numpy array """
    defux = ["ms", "s"]
    defuy = ["mV", "V"]
    inpunits = False
    try:
        with open(inputfile, 'r') as in_data:
            full_record = []  # python list
            for _ in range(0, 2):
                full_record.append(in_data.readline().strip().split())  # first and second record
            full_record.append([in_data.readline().strip() for line in range(0, int(full_record[1][0]))])  # optional record
            full_record.append([in_data.readline().strip().split('\t')])  # title record
            full_record.append(np.genfromtxt(in_data, dtype='float', delimiter='\t', unpack=True))  # data record, use: 'np.loadtxt()' for limited memory usage, but lower speed
    except FileNotFoundError:
        print("Error! File not found.")
        read_result = None
        raise
    else:
        inpuxy = full_record[3][0]
        inpux = str(inpuxy[0]).split()[-1][1:-2]
        inpuy = str(inpuxy[1]).split()[-1][1:-2]
        if inpux in defux and inpuy in defuy:
            inpunits = True
        inp_xy = full_record[4]
        read_result = inp_xy, inpunits, inpux, inpuy
    return read_result


def toggle_mpp_imode(activate=True):
    """Toggle matplotlib interactive mode on or off."""
    inter_active = mpp.isinteractive()
    if activate and not inter_active:
        mpp.ion()  # turn interactive mode on
    elif not activate and inter_active:
        mpp.ioff()  # turn interactive mode off


# main routine

AUTHOR = "Copyright 2021 The Regents of the University of Colorado"
SEPARBOLD = 79*'='
SEPARNORM = 79*'-'
SOFTWARE = "ParamAP"
VERSION = "version 1.3,"  # (2021-05-29)
WORKDIR = SOFTWARE  # working directory for parameterization
print('{0:^79}'.format(SEPARBOLD) + os.linesep)
GREETER = '{0:<{w0}}{1:<{w1}}{2:<{w2}}'.format(SOFTWARE, VERSION, AUTHOR, w0=len(SOFTWARE)+1, w1=len(VERSION)+1, w2=len(AUTHOR)+1)
INTERMEDIATELINE = '{0:}'.format("University of Colorado, Anschutz Medical Campus")
DISCLAIMER = "ParamAP is distributed in the hope that it will be useful, but it comes without\nany guarantee or warranty.  This program is free software; you can redistribute\nit and/or modify it under the terms of the GNU General Public License:"
URL = "https://www.gnu.org/licenses/gpl-2.0.en.html"
print('{0:^79}'.format(GREETER) + os.linesep)
print('{0:^79}'.format(INTERMEDIATELINE) + os.linesep)
print('{0:^79}'.format(DISCLAIMER) + os.linesep)
print('{0:^79}'.format(URL) + os.linesep)
print('{0:^79}'.format(SEPARBOLD) + os.linesep)

# customize use case
AUTORUN = askboolean("Use automatic mode?", False)
SERIES = askboolean("Run time series analysis?", False)
APMODE = askboolean("Analyze action potentials?", True)
print('{0:^79}'.format(SEPARNORM))

# set up working directory
WORKPATH = os.path.abspath(WORKDIR)
if not os.path.exists(WORKPATH):
    os.mkdir(WORKPATH)
print("FOLDER:\t" + WORKPATH + "\n")

FILE = 0  # file
EXTENSION = askext(dlabel="custom file type", dext='atf')  # file extension used to filter files in working directory
if SERIES:
    AVG_FRAME = askvalue(dlabel="analysis frame time", dval=5000.0, dunit=' ms')  # time interval for series analysis (ms)
ATFFILES = getfiles(path=WORKDIR, pattern=("*." + EXTENSION))
for ATFFILE in ATFFILES:  # iterate through files
    name = os.path.splitext(os.path.split(ATFFILE)[1])[0]
    print('{0:^79}'.format(SEPARNORM))
    print("FILE:\t" + str(name) + os.linesep)

    AP_AMP = 50.0  # minimum acceptable ap amplitude (mV)
    AP_HWD = 250.0  # maximum acceptable ap half width (ms)
    AP_MAX = 50.0  # maximum acceptable ap value (mV)
    AP_MIN = -10.0  # minimum acceptable ap value (mV)
    MDP_MAX = -50.0  # maximum acceptable mdp value (mV)
    MDP_MIN = -90.0  # minimum acceptable mdp value (mV)
    WM_DER = 1.0  # window multiplier for derivative filtering
    WM_MAX = 4.0  # window multiplier for maximum detection
    WM_MIN = 16.0  # window multiplier for minimum detection

    # read file raw data
    sys.stdout.write(">> READING... ")
    sys.stdout.flush()
    RAW_XY, UNITS, UNIT_X, UNIT_Y = readfile(ATFFILE)
    if not UNITS:  # missing or incomplete units from header
        print("\n")
        UNIT_X = askunit(dlabel="X-axis unit", daxis="X", dunit=UNIT_X)
        UNIT_Y = askunit(dlabel="Y-axis unit", daxis="Y", dunit=UNIT_Y)
        sys.stdout.write(1*"\t")
    TOMS = 1000.0 if UNIT_X == "s" else 1.0
    RAW_XY[0] *= TOMS  # full X-axis, UNIT_X = "ms"
    raw_x = RAW_XY[0]  # partial X-axis for time series analysis
    TOMV = 1000.0 if UNIT_Y == "V" else 1.0
    RAW_XY[1] *= TOMV  # full Y-axis, UNIT_Y = "mV"
    raw_y = RAW_XY[1]  # partial Y-axis for time series analysis
    runavg = getrunavg(RAW_XY[0])  # used for filtering and peak detection
    ipg_t = RAW_XY[0][1]-RAW_XY[0][0]  # time increment for interpolation grid
    if not APMODE:  # avoid noise artifacts in beat detection mode
        runavg = 10.0*runavg+1
        WM_MAX *= 1.5
        WM_MIN = WM_MAX
    avg_start = RAW_XY[0][0]  # interval start for averaging
    avg_stop = RAW_XY[0][-1]  # interval stop for averaging
    sys.stdout.write(8*"\t" + "   [OK]\n")
    sys.stdout.flush()

    while True:  # repeat data analysis for current file
        STARTPDF = True  # overwrite existing file
        SEGMENT = 0.0
        while True:  # time series analysis

            try:
                # create raw data plot
                sys.stdout.write(">> PLOTTING... ")
                sys.stdout.flush()
                mpp_setup(title="Raw data: " + name, xlabel='Time (ms)', ylabel='Voltage (mV)')
                mpp.plot(raw_x, raw_y, '0.75')  # raw data (grey line)

                if STARTPDF:
                    pdf_file = mpbp.PdfPages(os.path.join(WORKDIR, name + ".pdf"), keep_empty=False)  # multi-pdf file
                    STARTPDF = False   # append existing file
                mpp.tight_layout()  # avoid label overlaps

                if SEGMENT == 0.0:
                    mpp.savefig(pdf_file, format='pdf', dpi=600)  # save before .show()!
                sys.stdout.write(8*"\t" + "   [OK]\n")
                sys.stdout.flush()

                if not AUTORUN:
                    toggle_mpp_imode(activate=True)
                    mpp.show()

                # set parameters for averaging
                sys.stdout.write(">> SETTING... ")
                sys.stdout.flush()

                if not AUTORUN:
                    print("\n")
                    if SEGMENT == 0.0:  # initialize values
                        avg_start = askvalue(dlabel="analysis start time", dval=avg_start, dunit=' ms')
                        avg_stop = askvalue(dlabel="analysis stop time", dval=avg_stop, dunit=' ms')
                    AP_MAX = askvalue(dlabel="upper limit for maxima", dval=AP_MAX, dunit=' mV')
                    AP_MIN = askvalue(dlabel="lower limit for maxima", dval=AP_MIN, dunit=' mV')
                    MDP_MAX = askvalue(dlabel="upper limit for minima", dval=MDP_MAX, dunit=' mV')
                    MDP_MIN = askvalue(dlabel="lower limit for minima", dval=MDP_MIN, dunit=' mV')
                    if APMODE:
                        AP_HWD = askvalue(dlabel="maximum peak half width", dval=AP_HWD, dunit=' ms')
                        AP_AMP = askvalue(dlabel="minimum peak amplitude", dval=AP_AMP, dunit=' mV')
                    runavg = askvalue(dlabel="running average window size", dval=runavg, dunit='', dtype='int')
                    WM_DER = askvalue(dlabel="window multiplier for derivative", dval=WM_DER, dunit='')
                    WM_MAX = askvalue(dlabel="window multiplier for maxima", dval=WM_MAX, dunit='')
                    WM_MIN = askvalue(dlabel="window multiplier for minima", dval=WM_MIN, dunit='')
                mpp.clf()  # clear canvas

                if SEGMENT == 0.0:  # set first frame
                    tmp_start = avg_start + (SEGMENT*AVG_FRAME if SERIES else 0.0)
                    tmp_stop = (tmp_start + AVG_FRAME) if SERIES else avg_stop
                    raw_i = np.argwhere((RAW_XY[0] >= tmp_start) & (RAW_XY[0] <= tmp_stop)).ravel()
                    raw_x = RAW_XY[0][raw_i[0]:raw_i[-1]+1]
                    raw_y = RAW_XY[1][raw_i[0]:raw_i[-1]+1]

                sys.stdout.write(("" if AUTORUN else 1*"\t") + 8*"\t" + "   [OK]\n")
                sys.stdout.flush()

                # filter noise of raw data with Savitzky-Golay
                sys.stdout.write(">> FILTERING... ")
                sys.stdout.flush()
                rawf_y = sp_sig.savgol_filter(raw_y, runavg, POLYNOM, mode='nearest')
                sys.stdout.write(7*"\t" + "   [OK]\n")
                sys.stdout.flush()

                # detect extrema in filtered raw data
                sys.stdout.write(">> SEARCHING... ")
                sys.stdout.flush()

                if AUTORUN:  # use unrestricted dataset (slower)

                    # detect maxima in filtered raw data
                    tmpavg = int(round(WM_MAX*runavg)) if int(round(WM_MAX*runavg)) % 2 else int(round(WM_MAX*runavg))+1
                    rawfmax_iii = np.asarray(sp_sig.argrelmax(rawf_y, order=tmpavg)).ravel()  # unfiltered maxima
                    rawfmax_x = raw_x[rawfmax_iii]
                    rawfmax_y = rawf_y[rawfmax_iii]

                    # detect minima in filtered raw data
                    tmpavg = int(round(WM_MIN*runavg)) if int(round(WM_MIN*runavg)) % 2 else int(round(WM_MIN*runavg))+1
                    rawfmin_iii = np.asarray(sp_sig.argrelmin(rawf_y, order=tmpavg)).ravel()  # unfiltered minima
                    rawfmin_x = raw_x[rawfmin_iii]
                    rawfmin_y = rawf_y[rawfmin_iii]
                    sys.stdout.write(7*"\t" + "   [OK]\n")
                    sys.stdout.flush()

                else:  # use restricted dataset (faster)

                    # detect maxima in filtered raw data
                    tmpmax_x = raw_x[np.intersect1d(np.argwhere(rawf_y >= AP_MIN), np.argwhere(rawf_y <= AP_MAX))]
                    tmpmax_y = rawf_y[np.intersect1d(np.argwhere(rawf_y >= AP_MIN), np.argwhere(rawf_y <= AP_MAX))]
                    tmpavg = int(round(WM_MAX*runavg)) if int(round(WM_MAX*runavg)) % 2 else int(round(WM_MAX*runavg))+1
                    rawfmax_iii = np.asarray(sp_sig.argrelmax(tmpmax_y, order=tmpavg)).ravel()  # unfiltered maxima
                    rawfmax_ii = np.asarray(np.where(np.in1d(raw_x.ravel(), np.intersect1d(raw_x, tmpmax_x[rawfmax_iii]).ravel()).reshape(raw_x.shape))).ravel()  # back to full dataset
                    rawfmax_x = raw_x[rawfmax_ii]
                    rawfmax_y = rawf_y[rawfmax_ii]

                    # detect minima in filtered raw data
                    tmpmin_x = raw_x[np.intersect1d(np.argwhere(rawf_y >= MDP_MIN), np.argwhere(rawf_y <= MDP_MAX))]
                    tmpmin_y = rawf_y[np.intersect1d(np.argwhere(rawf_y >= MDP_MIN), np.argwhere(rawf_y <= MDP_MAX))]
                    tmpavg = int(round(WM_MIN*runavg)) if int(round(WM_MIN*runavg)) % 2 else int(round(WM_MIN*runavg))+1
                    rawfmin_iii = np.asarray(sp_sig.argrelmin(tmpmin_y, order=tmpavg)).ravel()  # unfiltered minima
                    rawfmin_ii = np.asarray(np.where(np.in1d(raw_x.ravel(), np.intersect1d(raw_x, tmpmin_x[rawfmin_iii]).ravel()).reshape(raw_x.shape))).ravel()
                    rawfmin_x = raw_x[rawfmin_ii]
                    rawfmin_y = rawf_y[rawfmin_ii]
                    sys.stdout.write(7*"\t" + "   [OK]\n")
                    sys.stdout.flush()

                # analyze and reduce extrema in filtered raw data
                sys.stdout.write(">> REDUCING... ")
                sys.stdout.flush()
                rawfmax_m = np.mean(rawfmax_y)  # rough estimate due to assignment errors
                rawfmin_m = np.mean(rawfmin_y)
                rawfmaxmin_m = (rawfmax_m + rawfmin_m) / 2.0  # center between unreduced maxima and minima within limits (may differ from average of AVGMAX and AVGMIN)

                if AUTORUN:  # estimate range for reduction of extrema

                    # reduce maxima from unrestricted dataset
                    rawfmax_ii = np.argwhere(rawfmax_y >= rawfmaxmin_m).ravel()  # use center to discriminate between maxima and minima
                    rawfmax_x = rawfmax_x[rawfmax_ii]
                    rawfmax_y = rawfmax_y[rawfmax_ii]
                    rawfmax_std = np.std(rawfmax_y, ddof=1)  # standard deviation from the (estimated) arithmetic mean
                    AP_MAX = np.mean(rawfmax_y) + 4.0 * rawfmax_std  # 99% confidence interval
                    AP_MIN = np.mean(rawfmax_y) - 4.0 * rawfmax_std
                    rawfmax_ii = functools.reduce(np.intersect1d, (rawfmax_iii, np.argwhere(rawf_y >= AP_MIN), np.argwhere(rawf_y <= AP_MAX)))
                    rawfmax_x = raw_x[rawfmax_ii]
                    rawfmax_y = rawf_y[rawfmax_ii]

                    # reduce minima from unrestricted dataset
                    rawfmin_ii = np.argwhere(rawfmin_y <= rawfmaxmin_m)
                    rawfmin_x = rawfmin_x[rawfmin_ii].ravel()
                    rawfmin_y = rawfmin_y[rawfmin_ii].ravel()
                    rawfmin_std = np.std(rawfmin_y, ddof=1)
                    MDP_MAX = np.mean(rawfmin_y) + 4.0 * rawfmin_std
                    MDP_MIN = np.mean(rawfmin_y) - 4.0 * rawfmin_std
                    rawfmin_ii = functools.reduce(np.intersect1d, (rawfmin_iii, np.argwhere(rawf_y >= MDP_MIN), np.argwhere(rawf_y <= MDP_MAX)))
                    rawfmin_x = raw_x[rawfmin_ii]
                    rawfmin_y = rawf_y[rawfmin_ii]

                if APMODE:  # check extrema for consistency - reduce maxima
                    badmax_ii = np.zeros(0)
                    badmin_ii = np.zeros(0)
                    rawfmin_i, badmax_ii = getneighbors(rawfmax_ii, rawfmin_ii, raw_x, rawf_y, AP_HWD, AP_AMP, bads=True)
                    rawfmax_i = np.delete(rawfmax_ii, badmax_ii)
                    rawfmin_i = rawfmin_i.astype(int)  # casting required for indexing

                    # check extrema for boundary violations - reduce maxima and minima
                    while True:  # rough check, assignment happens later
                        if rawfmax_i[0] < rawfmin_i[0]:  # starts with a maximum
                            rawfmax_i = rawfmax_i[1:]
                            continue
                        elif rawfmin_i[1] < rawfmax_i[0]:  # starts with two minima
                            rawfmin_i = rawfmin_i[1:]
                            continue
                        elif rawfmax_i[-1] > rawfmin_i[-1]:  # ends with a maximum
                            rawfmax_i = rawfmax_i[0:-1]
                            continue
                        elif rawfmin_i[-2] > rawfmax_i[-1]:  # ends with two minima
                            rawfmin_i = rawfmin_i[0:-1]
                            continue
                        else:
                            break
                    rawfmax_x = raw_x[rawfmax_i]  # filtered and extracted maxima
                    rawfmax_y = rawf_y[rawfmax_i]

                    # assign minima to corresponding maxima - reduce minima
                    minmaxmin = np.asarray([3*[0] for i in range(rawfmax_i.size)])  # [[min_left_index, max_index, min_right_index], ...]
                    rawfmin_kdt = sp_spat.KDTree(list(zip(rawfmin_i, np.zeros(rawfmin_i.size))))
                    i = 0  # index
                    for max_i in rawfmax_i:
                        MIN_LEFT, MIN_RIGHT = False, False
                        minmaxmin[i][1] = max_i
                        for order_i in rawfmin_kdt.query([max_i, 0.0], k=None)[1]:
                            min_i = rawfmin_i[order_i]
                            if not MIN_LEFT and (min_i < max_i):
                                minmaxmin[i][0] = min_i
                                MIN_LEFT = True
                            elif not MIN_RIGHT and (min_i > max_i):
                                minmaxmin[i][2] = min_i
                                MIN_RIGHT = True
                        i += 1
                    rawfmin_i = np.unique(minmaxmin[:, [0, 2]].ravel())
                    rawfmin_x = raw_x[rawfmin_i]  # filtered and extracted minima
                    rawfmin_y = rawf_y[rawfmin_i]

                    # find largest distance between left minima and maxima
                    ipg_hwl, ipg_tmp = 0.0, 0.0
                    for min_l, max_c in minmaxmin[:, [0, 1]]:
                        ipg_tmp = raw_x[max_c] - raw_x[min_l]
                        if ipg_tmp > ipg_hwl:
                            ipg_hwl = ipg_tmp

                    # find largest distance between right minima and maxima
                    ipg_hwr, ipg_tmp = 0.0, 0.0
                    for max_c, min_r in minmaxmin[:, [1, 2]]:
                        ipg_tmp = raw_x[min_r] - raw_x[max_c]
                        if ipg_tmp > ipg_hwr:
                            ipg_hwr = ipg_tmp

                else:  # beating rate
                    rawfmax_x = raw_x[rawfmax_ii]  # pre-filtered maxima
                    rawfmax_y = rawf_y[rawfmax_ii]
                    rawfmin_x = raw_x[rawfmin_ii]  # pre-filtered minima
                    rawfmin_y = rawf_y[rawfmin_ii]

                rawfmax_m = np.mean(rawfmax_y)  # refined estimate due to exlusion (ap_mode)
                rawfmin_m = np.mean(rawfmin_y)
                if rawfmax_y.size == 0:  # no APs detected
                    raise Warning
                else:  # two or more APs
                    frate = 60000.0*(rawfmax_y.size/(rawfmax_x[-1]-rawfmax_x[0])) if rawfmax_y.size > 1 else float('nan')  # AP firing rate (FR) [1/min]
                sys.stdout.write(8*"\t" + "   [OK]\n")
                sys.stdout.flush()

                # create extrema plot
                sys.stdout.write(">> PLOTTING... ")
                sys.stdout.flush()
                mpp_setup(title="Extrema: " + name, xlabel='Time (ms)', ylabel='Voltage (mV)')
                mpp.plot([raw_x[0], raw_x[-1]], [0.0, 0.0], '0.85')  # X-Axis (grey line)
                mpp.plot([raw_x[0], raw_x[-1]], [rawfmaxmin_m, rawfmaxmin_m], 'k--')  # center between unfiltered maxima and unfiltered minima, i.e. not between AVGMAX and AVGMIN (black dashed line)
                mpp.plot(raw_x, raw_y, '0.50', raw_x, rawf_y, 'r')  # raw data and averaged data (grey, red line)
                mpp.plot([raw_x[0], raw_x[-1]], [AP_MAX, AP_MAX], 'b')  # upper limit for maxima (blue dotted line)
                mpp.plot([raw_x[0], raw_x[-1]], [AP_MIN, AP_MIN], 'b:')  # lower limit for maxima (blue dotted line)
                mpp.plot([rawfmax_x, rawfmax_x], [AP_MIN, AP_MAX], 'b')  # accepted maxima (blue line)
                mpp.plot([raw_x[0], raw_x[-1]], [MDP_MIN, MDP_MIN], 'g')  # lower limit for minima (green line)
                mpp.plot([raw_x[0], raw_x[-1]], [MDP_MAX, MDP_MAX], 'g:')  # upper limit for minima (green dotted line)
                mpp.plot([rawfmin_x, rawfmin_x], [MDP_MIN, MDP_MAX], 'g')  # accepted minima (green line)
                mpp.plot([rawfmax_x[0], rawfmax_x[-1]], [rawfmax_m, rawfmax_m], 'k')  # average of maxima, time interval used for firing rate count (black line)
                mpp.plot([rawfmin_x[0], rawfmin_x[-1]], [rawfmin_m, rawfmin_m], 'k')  # average of minima (black line)
                mpp.plot(raw_x[rawfmax_ii], rawf_y[rawfmax_ii], 'bo')  # maxima (blue dots)
                mpp.plot(raw_x[rawfmin_ii], rawf_y[rawfmin_ii], 'go')  # minima (green dots)
                mpp.figtext(0.12, 0.91, "{0:<s} {1:<.4G}".format("AVGMAX (mV):", rawfmax_m), ha='left', va='center')
                mpp.figtext(0.12, 0.88, "{0:<s} {1:<.4G}".format("FR (AP/min):", frate), ha='left', va='center')
                mpp.figtext(0.12, 0.85, "{0:<s} {1:<.4G}".format("AVGMIN (mV):", rawfmin_m), ha='left', va='center')
                mpp.tight_layout()
                sys.stdout.write(8*"\t" + "   [OK]\n")
                sys.stdout.flush()

                sys.stdout.write(">> SAVING... ")
                sys.stdout.flush()
                mpp.savefig(pdf_file, format='pdf', dpi=600)
                sys.stdout.write(8*"\t" + "   [OK]\n")
                sys.stdout.flush()
                if not AUTORUN:
                    toggle_mpp_imode(activate=False)
                    mpp.show()
                mpp.clf()

                if APMODE:
                    # slice raw data segments by minima and align by maxima
                    sys.stdout.write(">> AVERAGING... ")
                    sys.stdout.flush()

                    # align ap segments by maxima, extend and average ap segments
                    ipg_max = float((1.0+APEXT)*ipg_hwr)
                    ipg_min = -float((1.0+APEXT)*ipg_hwl)
                    avg_x = np.arange(ipg_min, ipg_max, ipg_t, dtype='float64')  # interpolation grid
                    avgxsize = avg_x.size
                    avg_y = np.zeros(avgxsize, dtype='float64')  # ap average array
                    mpp.subplot2grid((4, 1), (0, 0), rowspan=3)  # upper subplot
                    TIMESTAMP = "[" + str(round(tmp_start, 2)) + "ms-" + str(round(tmp_stop, 2)) + "ms]"
                    mpp_setup(title='Analysis: ' + name + ' ' + TIMESTAMP, xlabel='Time (ms)', ylabel='Voltage (mV)')
                    mpp.plot([avg_x[0], avg_x[-1]], [0.0, 0.0], '0.85')  # X-axis
                    N = 0  # current maximum
                    for min_l, max_c, min_r in minmaxmin:  # slicing of ap segments, extend ap parts if possible
                        minext_l = int(min_l - APEXT*(max_c - min_l))  # use int for index slicing
                        minext_r = int(min_r + APEXT*(min_r - max_c))
                        # prepare ap SEGMENT
                        tmp_x = np.asarray(raw_x[:] - raw_x[max_c])  # align by maximum
                        tmp_y = np.interp(avg_x, tmp_x, raw_y[:])
                        # average ap segments
                        if N == 0:  # first average
                            avg_y = np.copy(tmp_y)
                        else:  # all other averages
                            i = 0  # array index
                            NW = (1.0/(N+1.0))  # new data weight
                            pw = (N/(N+1.0))  # previous data weight
                            for y in np.nditer(avg_y, op_flags=['readwrite']):
                                y[...] = pw*y + NW*tmp_y[i]  # integrate raw data into averaged data
                                i += 1
                        N += 1
                        mpp.plot(avg_x, tmp_y, '0.75')  # plot aligned raw data segments
                    sys.stdout.write("\t\t\t\t\t\t\t   [OK]\n")
                    sys.stdout.flush()

                    # analyze AP parameters with given criteria
                    sys.stdout.write(">> ANALYZING... ")
                    sys.stdout.flush()

                    # filter noise of averaged data with Savitzky-Golay
                    avgf_y = sp_sig.savgol_filter(avg_y, runavg, POLYNOM, mode='nearest')

                    # detect "Peak potential: Maximum potential of AP" (PP) (mV)
                    avgfmax_i = np.argwhere(avg_x == 0.0)  # data point for maximum centered
                    if not avgfmax_i.size > 0:  # data point for maximum left or right of center
                        tmpavg = int(round(WM_MAX*runavg)) if int(round(WM_MAX*runavg)) % 2 else int(round(WM_MAX*runavg))+1
                        avgfmax_ii = np.asarray(sp_sig.argrelmax(avgf_y, order=tmpavg)).ravel()  # find all maxima
                        avgfmax_i = avgfmax_ii[np.argmin(np.abs(avg_x[avgfmax_ii] - 0.0))]  # return the maximum closest to X = 0.0
                    avgfmax_x = avg_x[avgfmax_i]
                    avgfmax_y = avgf_y[avgfmax_i]
                    pp_y = float(avgfmax_y)
                    pp = pp_y

                    # detect and reduce (several) minima in filtered average data,
                    tmpavg = int(round(WM_MIN*runavg)) if int(round(WM_MIN*runavg)) % 2 else int(round(WM_MIN*runavg))+1
                    avgfmin_ii = np.asarray(sp_sig.argrelmin(avgf_y, order=tmpavg)).ravel()  # find all minima
                    avgfmin_i = getneighbors(np.asarray([avgfmax_i]), avgfmin_ii, avg_x, avgf_y, AP_HWD, AP_AMP)
                    avgfmin_x = avg_x[avgfmin_i]
                    avgfmin_y = avgf_y[avgfmin_i]

                    # determine "Maximum diastolic potential 1: Minimum potential preceding PP" (MDP1) (mV)
                    mdp1_i = avgfmin_i[0]
                    mdp1_x = avg_x[mdp1_i]
                    mdp1_y = avgf_y[mdp1_i]
                    mdp1 = mdp1_y

                    # determine "Maximum diastolic potential 2: Minimum potential following PP" (MDP2) (mV)
                    mdp2_i = avgfmin_i[-1]
                    mdp2_x = avg_x[mdp2_i]
                    mdp2_y = avgf_y[mdp2_i]
                    mdp2 = mdp2_y

                    # determine "Cycle length: Time interval MDP1-MDP2" (CL) (ms)
                    cl = float(mdp2_x - mdp1_x)

                    # determine "Action potential amplitude: Potential difference of PP minus MDP2" (APA) (mV)
                    apa = pp - mdp2

                    # determine "AP duration 30: Time interval at 30% of maximum repolarization" (APD30) (ms)
                    apd30_l = (pp - 0.30*apa)  # threshold value
                    apd30_i = functools.reduce(np.intersect1d, (np.argwhere(avgf_y > apd30_l), np.argwhere(avg_x >= mdp1_x), np.argwhere(avg_x <= mdp2_x)))
                    apd30_x = (avg_x[apd30_i[0]-1], avg_x[apd30_i[-1]+1])  # equal to or smaller than apd30_l
                    apd30_y = (avgf_y[apd30_i[0]-1], avgf_y[apd30_i[-1]+1])
                    apd30 = float(apd30_x[-1] - apd30_x[0])

                    # determine "AP duration 50: Time interval at 50% of maximum repolarization" (APD50) (ms)
                    apd50_l = (pp - 0.50*apa)  # threshold value
                    apd50_i = functools.reduce(np.intersect1d, (np.argwhere(avgf_y > apd50_l), np.argwhere(avg_x >= mdp1_x), np.argwhere(avg_x <= mdp2_x)))
                    apd50_x = (avg_x[apd50_i[0]-1], avg_x[apd50_i[-1]+1])  # equal to or smaller than apd50_l
                    apd50_y = (avgf_y[apd50_i[0]-1], avgf_y[apd50_i[-1]+1])
                    apd50 = float(apd50_x[-1] - apd50_x[0])

                    # determine "AP duration 90: Time interval at 90% of maximum repolarization" (APD90) (ms)
                    apd90_l = pp - 0.90*apa
                    apd90_i = functools.reduce(np.intersect1d, (np.argwhere(avgf_y > apd90_l), np.argwhere(avg_x >= mdp1_x), np.argwhere(avg_x <= mdp2_x)))
                    apd90_x = (avg_x[apd90_i[0]-1], avg_x[apd90_i[-1]+1])  # equal to or smaller than apd90_l
                    apd90_y = (avgf_y[apd90_i[0]-1], avgf_y[apd90_i[-1]+1])
                    apd90 = float(apd90_x[-1] - apd90_x[0])

                    # calculate derivative of averaged data (mV/ms)
                    avgfg_y = np.ediff1d(avgf_y)  # dY/1, differences between values
                    avgfg_y = np.insert(avgfg_y, 0, avgfg_y[0])  # preserve array size
                    avgfg_y = avgfg_y / ipg_t  # dY/dX, differences per increment

                    # filter derivative of averaged data
                    tmpavg = int(round(WM_DER*runavg)) if int(round(WM_DER*runavg)) % 2 else int(round(WM_DER*runavg))+1
                    avgfgf_y = sp_sig.savgol_filter(avgfg_y, tmpavg, POLYNOM, mode='nearest')

                    # determine "Maximum upstroke velocity: Maximum of derivative between MDP1 and PP" (MUV) (mV/ms)
                    tmpavg = int(round(WM_MAX*runavg)) if int(round(WM_MAX*runavg)) % 2 else int(round(WM_MAX*runavg))+1
                    avgfgfmax_ii = functools.reduce(np.intersect1d, (sp_sig.argrelmax(avgfgf_y, order=tmpavg), np.argwhere(avg_x >= mdp1_x), np.argwhere(avg_x <= avgfmax_x)))
                    avgfgfmax_i = getneighbors(np.asarray([avgfmax_i]), avgfgfmax_ii, avg_x, avgfgf_y)[0]  # avoid errors from large ap part extensions
                    avgfgfmax_x = avg_x[avgfgfmax_i]
                    avgfgfmax_y = avgfgf_y[avgfgfmax_i]
                    muv = float(avgfgfmax_y)

                    # determine "Maximum repolarization rate: Minimum of derivative between PP and MDP2" (MRR) (mV/ms)
                    tmpavg = int(round(WM_MIN*runavg)) if int(round(WM_MIN*runavg)) % 2 else int(round(WM_MIN*runavg))+1
                    avgfgfmin_ii = functools.reduce(np.intersect1d, (sp_sig.argrelmin(avgfgf_y, order=tmpavg), np.argwhere(avg_x >= avgfmax_x), np.argwhere(avg_x <= mdp2_x)))
                    avgfgfmin_i = getneighbors(np.asarray([apd90_i[-1]+1]), avgfgfmin_ii, avg_x, avgfgf_y)[0]  # mrr or trr

                    # determine "Transient repolarization rate: Second minimum of derivative between PP and MDP2 after PP, if distinct from MRR" (TRR) (mV/ms)
                    avgfgfmin_i = np.append(avgfgfmin_i, getneighbors(np.asarray([avgfgfmax_i]), avgfgfmin_ii, avg_x, avgfgf_y)[1])  # trr only

                    avgfgfmin_x = avg_x[avgfgfmin_i]
                    avgfgfmin_y = avgfgf_y[avgfgfmin_i]
                    mrr = float(avgfgf_y[avgfgfmin_i][0])
                    if avgfgfmin_i[0] == avgfgfmin_i[1]:  # no trr
                        WITH_TRR = False
                        trr = float("nan")
                    else:
                        WITH_TRR = True
                        trr = float(avgfgf_y[avgfgfmin_i][1])  # True

                    # approximate diastolic duration in filtered derivative
                    da_i, da_x, da_y, da_m, da_n, da_r = getbestlinearfit(avg_x, avgfgf_y, mdp1_x, apd90_x[0], 10, 90, 1, 40)  # get a baseline for the derivative before exceeding the threshold

                    # determine "Threshold potential: Potential separating DD and APD." (THR) (mV)
                    thr_i = functools.reduce(np.intersect1d, (np.argwhere(avgfgf_y >= ((da_m*avg_x + da_n) + 0.5)), np.argwhere(avg_x >= avg_x[da_i[-1]]), np.argwhere(avg_x <= apd50_x[0])))[0].astype(int)  # determine baseline-corrected threshold level
                    thr_x = avg_x[thr_i]
                    thr_y = avgf_y[thr_i]
                    thr = float(thr_y)

                    # determine "Late AP duration"
                    lapd_l = thr
                    lapd_i = functools.reduce(np.intersect1d, (np.argwhere(avgf_y > lapd_l), np.argwhere(avg_x >= avgfmax_x), np.argwhere(avg_x <= mdp2_x)))
                    lapd_x = (0.5*(avg_x[lapd_i[-1]]+avg_x[lapd_i[-1]+1]))  # equal to or smaller than lapd_l
                    lapd_y = (0.5*(avgf_y[lapd_i[-1]]+avgf_y[lapd_i[-1]+1]))
                    lapd = float(mdp2_x - lapd_x)

                    # determine "Early diastolic duration: Time from MDP1 to end of linear fit for DDR" (EDD) (ms)
                    edd_i, edd_x, edd_y, edd_m, edd_n, edd_r = getbestlinearfit(avg_x, avgf_y, mdp1_x, thr_x, 10, 50, 1, 20)  # fit EDD within the threshold level determined earlier
                    edd = float(edd_x[-1]-mdp1_x)

                    # determine "Diastolic depolarization rate: Potential change rate at end of EDD" (DDR) (mV/ms)
                    ddr = float(edd_m)  # or: np.mean(avgfgf_y[edd_i])

                    # determine "Diastolic duration: EDD plus LDD" (DD) (ms)
                    dd = float(thr_x - mdp1_x)

                    # determine "Late diastolic duration: Time from end of linear fit for DDR to THR" (LDD) (ms)
                    ldd = float(thr_x - edd_x[-1])

                    # determine "Action potential duration: Time between THR and MDP2" (APD) (ms)
                    apd = float(mdp2_x - thr_x)
                    sys.stdout.write("\t\t\t\t\t\t\t   [OK]\n")
                    sys.stdout.flush()

                    # create analysis plot
                    sys.stdout.write(">> PLOTTING... ")  # the X-axis and the individual segments are already plotted during averaging
                    sys.stdout.flush()
                    mpp.plot([mdp1_x, thr_x], [mdp1_y, mdp1_y], 'k-.')  # DD (black dashed/dotted line)
                    mpp.plot([thr_x, mdp2_x], [mdp2_y, mdp2_y], 'k')  # APD (black line)

                    mpp.plot([lapd_x, mdp2_x], [lapd_y, lapd_y], 'r-.')  # LAPD (black line)
                    mpp.plot([apd50_x[0], apd50_x[1]], [apd50_y[1], apd50_y[1]], 'k')  # APD50 (black line)
                    mpp.plot([apd90_x[0], apd90_x[1]], [apd90_y[1], apd90_y[1]], 'k')  # APD90 (black line)
                    mpp.plot([mdp1_x, mdp1_x], [mdp1_y, 0.0], 'k:')  # MDP1 indicator (black dotted line)
                    mpp.plot([mdp2_x, mdp2_x], [mdp2_y, 0.0], 'k:')  # MDP2 indicator (black dotted line)
                    mpp.plot([avgfgfmax_x, avgfgfmax_x], [mdp2_y, avgf_y[avgfgfmax_i]], 'k:')  # MUV indicator (black dotted line)
                    mpp.plot([avgfgfmin_x[0], avgfgfmin_x[0]], [mdp2_y, avgf_y[avgfgfmin_i[0]]], 'k:')  # MRR indicator (black dotted line)

                    mpp.plot([edd_x[-1], edd_x[-1]], [mdp2_y, 0.0], 'k:')  # EDD/LDD separator (black dashed line)
                    mpp.plot([thr_x, thr_x], [thr_y, 0.0], 'k:')  # DD/APD upper separator (black dotted line)
                    mpp.plot([thr_x, thr_x], [mdp2_y, thr_y], 'k:')  # DD/APD lower separator (black dotted line)
                    mpp.plot(avg_x, avg_y, 'k', avg_x, avgf_y, 'r')  # averaged data and filtered averaged data (black, red lines)
                    mpp.plot(avg_x[edd_i], avgf_y[edd_i], 'g')  # best linear fit segment for DDR (green line)
                    mpp.plot(avg_x, (edd_m*avg_x + edd_n), 'k--')  # DDR (black dashed line)
                    mpp.plot([edd_x[-1]], [edd_y[-1]], 'ko')  # EDD-LDD separator (black dot)

                    mpp.plot(lapd_x, lapd_y, 'ro')  # LAPD (black dot)
                    mpp.plot([apd50_x[1]], [apd50_y[1]], 'ko')  # APD50 (black dots)
                    mpp.plot(apd90_x[1], apd90_y[1], 'ko')  # APD90 (black dots)
                    mpp.plot(thr_x, avgf_y[thr_i], 'ro')  # THR (red dot)
                    mpp.plot(avgfgfmax_x, avgf_y[avgfgfmax_i], 'wo')  # MUV (white dot)
                    mpp.plot(avgfgfmin_x[0], avgf_y[avgfgfmin_i[0]], 'wo')  # MRR (white dot)
                    if WITH_TRR:
                        mpp.plot([avgfgfmin_x[1], avgfgfmin_x[1]], [mdp2_y, avgf_y[avgfgfmin_i[1]]], 'k:')  # trr indicator (black dotted line)
                        mpp.plot(avgfgfmin_x[1], avgf_y[avgfgfmin_i[1]], 'wo')  # trr (white dot)
                    mpp.plot(avgfmax_x, pp_y, 'bo')  # PP (blue dot)
                    mpp.plot(avgfmin_x, avgfmin_y, 'go')  # MDP1, MDP2 (green dots)
                    mpp.figtext(0.12, 0.91, "{0:<s} {1:<.4G}".format("APs (#):", rawfmax_y.size), ha='left', va='center')
                    mpp.figtext(0.12, 0.88, "{0:<s} {1:<.4G}".format("FR (AP/min):", frate), ha='left', va='center')
                    mpp.figtext(0.12, 0.85, "{0:<s} {1:<.4G}".format("CL (ms):", cl), ha='left', va='center')
                    mpp.figtext(0.12, 0.82, "{0:<s} {1:<.4G}".format("DD (ms):", dd), ha='left', va='center')
                    mpp.figtext(0.12, 0.79, "{0:<s} {1:<.4G}".format("EDD (ms):", edd), ha='left', va='center')
                    mpp.figtext(0.12, 0.76, "{0:<s} {1:<.4G}".format("LDD (ms):", ldd), ha='left', va='center')
                    mpp.figtext(0.12, 0.73, "{0:<s} {1:<.4G}".format("APD (ms):", apd), ha='left', va='center')
                    mpp.figtext(0.12, 0.70, "{0:<s} {1:<.4G}".format("LAPD (ms):", lapd), ha='left', va='center')
                    mpp.figtext(0.12, 0.67, "{0:<s} {1:<.4G}".format("APD50 (ms):", apd50), ha='left', va='center')
                    mpp.figtext(0.12, 0.64, "{0:<s} {1:<.4G}".format("APD90 (ms):", apd90), ha='left', va='center')
                    mpp.figtext(0.12, 0.61, "{0:<s} {1:<.4G}".format("MDP1 (mV):", mdp1), ha='left', va='center')
                    mpp.figtext(0.12, 0.58, "{0:<s} {1:<.4G}".format("MDP2 (mV):", mdp2), ha='left', va='center')
                    mpp.figtext(0.12, 0.55, "{0:<s} {1:<.4G}".format("THR (mV):", thr), ha='left', va='center')
                    mpp.figtext(0.12, 0.52, "{0:<s} {1:<.4G}".format("PP (mV):", pp), ha='left', va='center')
                    mpp.figtext(0.12, 0.49, "{0:<s} {1:<.4G}".format("APA (mV):", apa), ha='left', va='center')
                    mpp.figtext(0.12, 0.46, "{0:<s} {1:<.4G}".format("DDR (mV/ms):", ddr), ha='left', va='center')
                    mpp.figtext(0.12, 0.43, "{0:<s} {1:<.4G}".format("MUV (mV/ms):", muv), ha='left', va='center')
                    mpp.figtext(0.12, 0.40, "{0:<s} {1:<.4G}".format("TRR (mV/ms):", trr), ha='left', va='center')
                    mpp.figtext(0.12, 0.37, "{0:<s} {1:<.4G}".format("MRR (mV/ms):", mrr), ha='left', va='center')
                    mpp.subplot2grid((4, 1), (3, 0))  # lower subplot
                    mpp_setup(title="", xlabel='Time (ms)', ylabel='(mV/ms)')
                    mpp.plot([avg_x[0], avg_x[-1]], [0.0, 0.0], '0.85')  # x axis
                    mpp.plot([avgfgfmin_x[0], avgfgfmin_x[0]], [avgfgfmin_y[0], avgfgfmax_y], 'k:')  # MRR indicator (black dotted line)
                    if WITH_TRR:
                        mpp.plot([avgfgfmin_x[1], avgfgfmin_x[1]], [avgfgfmin_y[1], avgfgfmax_y], 'k:')  # TRR indicator (black dotted line)
                    mpp.plot([thr_x, thr_x], [avgfgf_y[thr_i], avgfgfmax_y], 'k:')  # THR indicator (black dotted line)
                    mpp.plot(avg_x, avgfg_y, 'c', avg_x, avgfgf_y, 'm')  # derivative and filtered derivative
                    mpp.plot(avg_x[da_i], avgfgf_y[da_i], 'g')  # best linear fit SEGMENT for THR (green line)
                    mpp.plot(avg_x, (da_m*avg_x + da_n), 'k--')  # best linear fit for THR (black dashed line)
                    mpp.plot(thr_x, avgfgf_y[thr_i], 'ro')  # THR (red dot)
                    mpp.plot(avgfgfmax_x, avgfgfmax_y, 'bo')  # derivative maximum (blue dot)
                    mpp.plot(avgfgfmin_x, avgfgfmin_y, 'go')  # derivative minima (green dots)
                    sys.stdout.write(8*"\t" + "   [OK]\n")
                    sys.stdout.flush()

                    # data summary
                    sys.stdout.write(">> SAVING... ")
                    sys.stdout.flush()
                    avg_file = os.path.join(WORKDIR, name + "_" + TIMESTAMP + "_avg.dat")
                    UHEADER = "" +\
                        "Analysis start time: " + 4*"\t" + str(tmp_start) + " ms\n" + \
                        "Analysis stop time:" + 4*"\t" + str(tmp_stop) + " ms\n" + \
                        "Upper limit for maxima:" + 3*"\t" + str(AP_MAX) + " mV\n" + \
                        "Lower limit for maxima:" + 3*"\t" + str(AP_MIN) + " mV\n" + \
                        "Upper limit for minima:" + 3*"\t" + str(MDP_MAX) + " mV\n" + \
                        "Lower limit for minima:" + 3*"\t" + str(MDP_MIN) + " mV\n" + \
                        "Maximum peak half width:" + 3*"\t" + str(AP_HWD) + " ms\n" + \
                        "Minimum peak amplitude:" + 3*"\t" + str(AP_AMP) + " mV\n" + \
                        "Running average window size:" + 2*"\t" + str(runavg) + "\n" + \
                        "Window multiplier for derivative:" + "\t" + str(WM_DER) + "\n" + \
                        "Window multiplier for maxima:" + 2*"\t" + str(WM_MAX) + "\n" + \
                        "Window multiplier for minima:" + 2*"\t" + str(WM_MIN) + "\n" + \
                        "Time (ms)" + "\t" + "Averaged signal (mV)" + "\t" + "Filtered average (mV)"
                    np.savetxt(avg_file, np.column_stack((avg_x, avg_y, avgf_y)), fmt='%e', delimiter='\t', header=UHEADER)
                    mpp.tight_layout()
                    mpp.savefig(pdf_file, format='pdf', dpi=600)
                    sum_file = os.path.join(WORKDIR, "ParamAP.log")
                    NEWFILE = not bool(os.path.exists(sum_file))
                    with open(sum_file, 'a') as targetfile:  # append file
                        if NEWFILE:  # write header
                            targetfile.write("{0:s}\t{1:s}\t{2:s}\t{3:s}\t{4:s}\t{5:s}\t{6:s}\t{7:s}\t{8:s}\t{9:s}\t{10:s}\t{11:s}\t{12:s}\t{13:s}\t{14:s}\t{15:s}\t{16:s}\t{17:s}\t{18:s}\t{19:s}\t{20:s}\t{21:s}".format(
                                             "File ( )", "Start (ms)", "Stop (ms)", "APs (#)", "FR (AP/min)", "CL (ms)", "DD (ms)", "EDD (ms)", "LDD (ms)", "APD (ms)", "LAPD (ms)", "APD50 (ms)", "APD90 (ms)", "MDP1 (mV)", "MDP2 (mV)", "THR (mV)", "PP (mV)", "APA (mV)", "DDR (mV/ms)", "MUV (mV/ms)", "TRR (mV/ms)", "MRR (mV/ms)") + "\n")
                        targetfile.write("{0:s}\t{1:4G}\t{2:4G}\t{3:4G}\t{4:4G}\t{5:4G}\t{6:4G}\t{7:4G}\t{8:4G}\t{9:4G}\t{10:4G}\t{11:4G}\t{12:4G}\t{13:4G}\t{14:4G}\t{15:4G}\t{16:4G}\t{17:4G}\t{18:4G}\t{19:4G}\t{20:4G}\t{21:4G}".format(
                                          name, tmp_start, tmp_stop, rawfmax_y.size, frate, cl, dd, edd, ldd, apd, lapd, apd50, apd90, mdp1, mdp2, thr, pp, apa, ddr, muv, trr, mrr) + "\n")
                        targetfile.flush()
                    sys.stdout.write(8*"\t" + "   [OK]\n")
                    sys.stdout.flush()
                    if not AUTORUN:
                        mpp.show()

            except IndexError as ierr:  # check running average and window multiplier
                sys.stdout.write("\n" + 9*"\t" + "   [ER]")
                print("\r   ## Run failed. Detection of extrema or threshold failed.")
            except PermissionError as perr:  # file already opened or storage read-only
                sys.stdout.write("\n" + 9*"\t" + "   [ER]")
                print("\r   ## Run failed. File access denied by system.")
            except Warning as werr:  # increase averaging window time
                sys.stdout.write("\n" + 9*"\t" + "   [ER]")
                print("\r   ## Run failed. Identification of action potentials failed.")
            except Exception as uerr:  # unknown
                sys.stdout.write("\n" + 9*"\t" + "   [UN]")
                print("\r   ## Run failed. Error was: {0}".format(uerr) + ".")
            except KeyboardInterrupt as kerr:  # user canceled this file
                sys.stdout.write("\n" + 9*"\t" + "   [KO]")
                print("\r   ## Run skipped. Canceled by user.")
            if SERIES:  # check for next frame
                if tmp_stop + AVG_FRAME <= avg_stop:
                    SEGMENT += 1.0
                    tmp_start = avg_start + SEGMENT*AVG_FRAME  # prepare next frame for preview
                    tmp_stop = tmp_start + AVG_FRAME
                    raw_i = np.argwhere((RAW_XY[0] >= tmp_start) & (RAW_XY[0] <= tmp_stop)).ravel()
                    raw_x = RAW_XY[0][raw_i[0]:raw_i[-1]+1]
                    raw_y = RAW_XY[1][raw_i[0]:raw_i[-1]+1]
                    print()
                    print("RUN:\t" + str(int(SEGMENT + 1)) + "/" + str(math.floor((avg_stop-avg_start)/AVG_FRAME)))
                    print()
                else:  # not enough data left in file
                    break
            else:  # no time series analysis
                break

        if not AUTORUN:  # check for next file
            print()
            NEXTFILE = askboolean("Continue with next file?", True)
            if NEXTFILE:
                break
            else:  # re-run current file
                raw_x = RAW_XY[0]  # recover original rawdata
                raw_y = RAW_XY[1]
                continue
        else:  # autorun
            break

    # housekeeping after each file
    FILE += 1
    sys.stdout.write(">> CLEANING... ")
    sys.stdout.flush()
    pdf_file.close()  # close multi-pdf file and remove if empty
    mpp.clf()  # clear canvas
    gc.collect()  # start garbage collection to prevent memory fragmentation
    sys.stdout.write(8*"\t" + "   [OK]\n")
    sys.stdout.flush()

# print summary
print('{0:^79}'.format(SEPARBOLD))
SUMMARY = "End of run: " + str(FILE) + str(" files" if FILE != 1 else " file") + " processed."
print('{0:^79}'.format(SUMMARY))
print('{0:^79}'.format(SEPARBOLD) + os.linesep)

WAIT = input("Press ENTER to end this program.")
