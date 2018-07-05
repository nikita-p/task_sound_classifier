import wave
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
import os
import sys

class Sound:
    def __init__(self, filename):
        f = wave.open(filename)
        (nchannels, sampwidth, framerate, nframes, comptype, compname) = f.getparams()
        content = f.readframes(nframes)
        #duration = nframes/framerate
        self.samples = np.frombuffer(content, dtype=np.int16)
        self.dT = 1/framerate
        self.fourier = np.abs( np.fft.rfft(self.samples) )
        
    def getDuration(self):
        return self.samples.size * self.dT
        
    def getResult(self):
        length = 21
        answer = np.zeros(length)
        answer[0] = np.max(np.abs(self.samples))/( np.sum(np.abs(self.samples))/10000 )
        xs = np.linspace(0, 1, self.fourier.size) * (np.pi / self.dT)
        const = 2500
        sumF = np.sum(self.fourier)
        
        for i in range( 0, min( int(np.max(xs)//const), length-1) ):
            answer[i+1] = np.mean( self.fourier[(xs>=i*const) & (xs<(i+1)*const)] ) / (sumF/10000)
    
        return answer


def get_filenames_and_rights():
    names = list()
    Ys = list()
    f = open('data_v_7_stc/meta/meta.txt', 'r')
    for line in f:
        t = line.split('\t')
        names.append(t[0])
        Ys.append(t[-1])
    
    return (names, Ys)
    
    
def FindMetrics():
    names, Ys = get_filenames_and_rights()
    fileWr = open('data.txt','w')
    fileWr.write('peak\tomg1\tomg2\tomg3\tomg4\tomg5\tomg6\tomg7\tomg8\tomg9\tomg10\tomg11\tomg12\tomg13\tomg14\tomg15\tomg16\tomg17\tomg18\tomg19\tomg20\tspecies\n')
    result = list()
    for name,y in zip(names,Ys):
        if not('time_stretch' in name):
            continue
        print( name )
        g = Sound('data_v_7_stc/audio/' + name)
        arr = g.getResult()
        #print(arr[0])
        [ fileWr.write( '%.2f\t' % j ) for j in arr]
        fileWr.write(y)
    fileWr.close()
    return

def Learning(filename='data.txt'):
    data = pd.read_csv(filename, sep='\t')
    X = data.drop('species', axis=1)
    Y = data['species']
    
    #Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y)
    
    modelForest = RandomForestClassifier(n_estimators=200, max_depth=40, criterion='entropy', max_features=None)
    modelForest.fit(X, Y)
    #modelForest.fit(Xtrain, Ytrain)
    #Ypred = modelForest.predict(Xtest)
    #print(metrics.classification_report(Ypred, Ytest))
    joblib.dump(modelForest, 'model.pkl')
    return
    
def Prediction(audioLibrary='data_v_7_stc/test/'):
    modelForest = joblib.load('model.pkl')
    files = os.listdir(audioLibrary)
    xs = list()
    ans = open('answers.txt', 'w')
    files = files[:100:5]
    for f in files:
        #print(f)
        audio = Sound(audioLibrary+f)    
        xs.append(audio.getResult())
    ys = modelForest.predict( np.array(xs) )
    probes = np.max( modelForest.predict_proba( np.array(xs) ), axis=1 )
    r, s = 0, 0
    for f,y,probe in zip(files,ys,probes):
        print(f, probe, y )
        if (y in f):
            r+=1
        if not('unknown' in f):
            s+=1
        ans.write( (f + '\t' + str(probe) + '\t' + y + '\n') )
    print('Final:', r/s )
    return

if __name__=="__main__":
    pars = sys.argv
    
    if("-h" in pars):
        print("Keys:\n-m - find parameters\n-l - model learning\n-p - make prediction")
    elif("-m" in pars):
        FindMetrics()
    elif("-l" in pars):
        Learning()
    elif("-p" in pars):
        Prediction()
    else:
        print("Error. Add key -h to get help")
