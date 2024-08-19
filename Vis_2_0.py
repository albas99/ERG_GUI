import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pywt
import cv2

from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks



def find_components(cwt):
    img_gray = ((255/10)*np.abs(np.log(np.abs(cwt) + 1))).astype('uint8')
    method=cv2.THRESH_TOZERO
    min_threshold = 25
    max_threshold = 255
    _, mask = cv2.threshold(img_gray,min_threshold,
                        max_threshold,method+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8) # kernel
    opening = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    ret, markers = cv2.connectedComponents(opening.astype('uint8'))
    return markers


def plot_wavelet(pdf, data, w='gaus8', D=1, f=[1. / 1, 1. / 100], re=False):
    fig = plt.figure()
    fig.set_figheight(7.5)
    fig.set_figwidth(30)
    plt.plot(data, 'k', linewidth=8)
    plt.title(person, **axis_font)
    plt.xlabel('ms', **axis_font)
    plt.ylabel('\u03BCV', **axis_font)
    locs, labels = plt.xticks()
    plt.xticks(ticks=locs[1:-1], labels=locs[1:-1]/2 , **axis_font)
    plt.yticks( **axis_font)
    plt.grid()
    plt.tight_layout()
    pdf.savefig(fig)

    plt.close(fig)

    f1 = f[0]
    f2 = f[-1]
    fc = pywt.central_frequency(w)
    a1 = fc / (D * f1)
    a2 = fc / (D * f2)
    lna1 = np.log(a1)
    lna2 = np.log(a2)
    ln_a = np.linspace(lna1, lna2, 300)
    A = np.exp(ln_a)
    F = fc / (D * A)
    addition = 10
    data_begin = np.mean(data[0]) * np.ones([addition, ])
    data_end = np.mean(data[-1:]) * np.ones([addition, ])

    temp_data = np.hstack([data_begin, data, data_end])

    dA = np.diff(A)
    dA = dA.tolist()
    dA.append(dA[-1])

    cwt, frq = pywt.cwt(temp_data, A, w)

    cwt = cwt[:, addition: addition + len(data)]
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(30)
    plt.imshow(np.log(np.abs(cwt) + 1), cmap='jet', aspect='auto', vmax=4.5)
    # plt.colorbar()
    locs, labels = plt.yticks()
    ind = [int(20 * loc / A[-1]) for loc in locs]
    plt.yticks(locs[1:-1], np.round(F[np.array(locs[1:-1]).astype('int')], 2), **axis_font)
    plt.ylabel('Hz', **axis_font)
    plt.xlabel('ms', **axis_font)
    locs, labels = plt.xticks()
    plt.xticks(ticks=locs[1:-1], labels=locs[1:-1] /2, **axis_font)
    plt.yticks(**axis_font)
    plt.tight_layout()
    plt.title(person, **axis_font)
    pdf.savefig(fig)
    plt.close(fig)

    markers = find_components(cwt)
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(30)
    plt.imshow(markers, aspect='auto')
    plt.ylabel('Hz', **axis_font)
    plt.xlabel('ms', **axis_font)
    plt.yticks(locs[1:-1], np.round(F[np.array(locs[1:-1]).astype('int')], 2), **axis_font)
    plt.xticks(ticks=locs[1:-1], labels=locs[1:-1]/2 , **axis_font)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    labels = np.unique(markers)

    features = []

    for l in labels[1:]:
        zero = np.zeros(cwt.shape)
        zero[markers == l] = 1
        temp_result = np.log(np.abs(cwt) + 1) * zero
        max = np.max(temp_result)
        max_f_, max_t_ = np.unravel_index(np.argmax(temp_result), temp_result.shape)
        indeces = np.nonzero(temp_result)
        min_f, max_f = np.max(indeces[0]), np.min(indeces[0])
        min_t, max_t = np.min(indeces[1]), np.max(indeces[1])
        median = np.median(temp_result[indeces])
        mean = np.mean(temp_result[indeces])
        zero_ = np.zeros(cwt.shape)
        zero_[temp_result > 0.9 * max] = 1
        temp_result_ = zero_ * temp_result
        indeces_ = np.nonzero(temp_result_)
        min_f90, max_f90 = np.max(indeces_[0]), np.min(indeces_[0])
        min_t90, max_t90 = np.min(indeces_[1]), np.max(indeces_[1])

        if ((max_t - min_t) > 10) * ((min_f - max_f) > 10):
            features.append(
                [l, max, F[max_f_], max_t_/2.0, median, mean, F[min_f], F[max_f], min_t/2.0, max_t/2.0, F[min_f90], F[max_f90],
                 min_t90/2.0, max_t90/2.0])
        else:
            print('sorry ' + str(l))

    dA = np.diff(A)
    dA = dA.tolist()
    dA.append(dA[-1])

    V = np.transpose(np.multiply(np.transpose(cwt), dA / pow(A, 1.5)))

    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(30)
    icwt = sum(V.transpose())

    peaks, _ = find_peaks(-np.abs(icwt[F < 75]), distance=15)

    features2 = F[F < 75][peaks][::-1]

    plt.loglog(F[F < 75], np.abs(icwt[F < 75]), 'b', linewidth=9)
    plt.loglog(F[F > 75], np.abs(icwt[F > 75]), 'r', linewidth=9)
    plt.loglog(F[F < 75][peaks], np.abs(icwt[F < 75][peaks]), 'g*', markersize=15)
    plt.ylim([0.0001, 100])
    plt.grid()
    plt.xlabel('Hz', **axis_font)
    plt.ylabel('\u03BCV', **axis_font)
    plt.xticks(**axis_font)
    plt.yticks(**axis_font)
    plt.title(person, **axis_font)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(30)
    ICWT1 = (sum(V[F < 75]))  # [2*len(data):3*len(data)]
    # ICWT2 = (sum(V[(15<F)*(F<100)]))#[2*len(data):3*len(data)]
    ICWT3 = (sum(V[F > 75]))  # [2*len(data):3*len(data)]
    ax = plt.subplot(2, 1, 1)
    plt.plot(ICWT1, 'b', linewidth=8)
    plt.title(person)
    plt.grid()
    plt.xlabel('ms', **axis_font)
    plt.ylabel('\u03BCV', **axis_font)
    locs, labels = plt.xticks()
    plt.xticks(ticks=locs[1:-1], labels=locs[1:-1]/2, **axis_font)
    plt.yticks(**axis_font)
    ax = plt.subplot(2, 1, 2)
    plt.plot(ICWT3, 'r', linewidth=8)
    plt.grid()
    plt.xlabel('ms', **axis_font)
    plt.ylabel('\u03BCV', **axis_font)
    locs, labels = plt.xticks()
    plt.xticks(ticks=locs[1:-1], labels=locs[1:-1]/2 , **axis_font)
    plt.yticks(**axis_font)
    # plt.plot(ICWT3,'k', linewidth=0.7)
    plt.grid()

    plt.grid()
    plt.tight_layout()
    pdf.savefig(fig)

    plt.close(fig)

    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(30)
    ax = plt.subplot(2, 1, 2)
    plt.plot(ICWT3 + ICWT1, '--k', linewidth=9, alpha=0.75)
    plt.plot(ICWT3, 'r', linewidth=8, alpha=0.5)
    # plt.plot(ICWT2,'g', linewidth=8,alpha=0.5)
    plt.plot(ICWT1, 'b', linewidth=8, alpha=0.5)
    plt.grid()
    plt.xlabel('ms', **axis_font)
    plt.ylabel('\u03BCV', **axis_font)
    locs, labels = plt.xticks()
    plt.xticks(ticks=locs[1:-1], labels=locs[1:-1]/2 , **axis_font)
    plt.yticks(**axis_font)
    ax = plt.subplot(2, 1, 1)
    plt.plot(data, 'k', linewidth=10)
    plt.title(person)
    plt.grid()
    plt.xlabel('ms', **axis_font)
    plt.ylabel('\u03BCV', **axis_font)
    locs, labels = plt.xticks()
    plt.xticks(ticks=locs[1:-1], labels=locs[1:-1]/2 , **axis_font)
    plt.yticks(**axis_font)

    plt.tight_layout()

    pdf.savefig(fig)

    plt.close(fig)

    if re:
        return cwt
    else:
        return np.vstack(features), features2, data, np.abs(icwt),ICWT1, ICWT3


import matplotlib as mpl

params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w"}
plt.rcParams.update(params)

plt.rcParams['axes.titlepad'] =  25
mpl.rcParams['axes.linewidth'] = 2

plt.style.use('seaborn')

axis_font = {'fontname':'Times New Roman', 'size':'16'}

DF=pd.read_csv('Data_2_Pandas_MaxERG',delimiter='\t')

DF['Old/Young']=[x<18 for x in DF['Age']]
DF['Group']=[(int(x)+1+(int(y=='норм'))) for x,y in zip(DF['Old/Young'],DF['Диагноз'])]



indexes = [str(x) for x in DF['#']]

temp_ = []
a = ''
for x in indexes:
    if x in temp_:
        a = '.1'
    if x+a in temp_:
        a = '.2'
    temp_.append(x+a)
    a = ''

DF['#_'] = temp_

DF = DF.sort_values(by = 'Group')

DF_old = DF[DF['Old/Young']==False]

DF_signals = pd.read_csv('MaxErg', delimiter='\t')

temp = [str(x) for x in DF_old['#_']]

DF_cut = DF_signals[temp_][0:200]

feat_names = ['#','Возраст','Диагноз',
              'Амплитуда А волны','Амплитуда Б Волны','Латентность А волны','Латентность Б волны',
              'Пики частот',
              'Номер Сегмента','Максимум Вейвлет','Частота Максимума', "время максимума",
              "медианное значение", "среднее зрачение",
              "частота начала", "частота конца", "время начала", "время конца",
              "частота начала 90%", "частота конца 90%", "время начала 90%", "время конца 90%",]

features_all = []
dummy = ['', '', '', '', '', '', '']
dummy1 = np.array(['', '', '', '', '', '', '', '', '', '', '', '', '', ''])

D=0.5/1000

database = {}
with PdfPages('10.pdf') as pdf:
    for person in DF['#_']:
        signal_collection = {}
        feat_person = DF[DF['#_'] == person][['#_', 'Age', 'Диагноз', 'Amp. a-wave', 'Amp. b-wave', 'Time a-wave', 'Time b-wave']].values
        print(person)
        features1, features2, signal,spectr,wave1,wave2 = plot_wavelet(pdf, DF_cut[person].values, D=D,
                                            f=[0.1 / D, 0.5 * 0.0625 / (len(DF_cut[person].values) * D)])

        s = max(features1.shape[0], features2.shape[0])

        signal_collection['signal'] = signal
        signal_collection['spectr'] = spectr
        signal_collection['wave1'] = wave1
        signal_collection['wave2'] = wave2

        for i in range(s - 1):
            feat_person = np.vstack([feat_person, dummy])

        while s - features2.shape[0] > 0:
            features2 = np.append(features2, np.nan)

        while s - features1.shape[0] > 0:
            features1 = np.vstack([features1, dummy1])

        features_all.append(np.hstack([feat_person, features2.reshape((len(features2), 1)), features1]))
        database[person] = signal_collection

DF_All = pd.DataFrame(np.vstack(features_all), columns=feat_names)

DF_All.to_csv('data3.csv')

np.save('database.npy', database)