import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

############1.Linear Regression##############
#SH = pd.DataFrame(np.zeros((51,1),dtype=float),columns=['SHIT'])
#SH.drop([0], axis=0,inplace=True)

### Import the dataset 

#df = pd.read_csv("train.csv")
df = pd.read_csv("train.csv", index_col=0) # delete first column
# df.head()
# df.info()

df = pd.get_dummies(df)
#print(df[:5])

"""
label =['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason',
        'guardian','schoolsup','famsup','paid','activities','nursery',
        'higher','internet','romantic','cat']

for i in label:
    d = pd.get_dummies(df[i])
    df.drop([i], axis=1,inplace=True) # inplace=True (delete original data)
    res = pd.concat([df, d], axis=1)  # axis 0: (do with column); 1: (do with row)
    df = res

# print(df[:5])
# print(u.columns)
"""

### (a) normalization
ndf = df

for i in df.columns:
    if i != "G3":
        n = ndf[i].values
        mu = n.mean() # mean
        std = n.std() # standard deviations

        normalized = (n - mu) / std
        ndf[i] = normalized

### (b) RMSE

train_set = ndf[:800].copy()
test_set = ndf[800:1000].copy()

G3_train =train_set['G3'].values
G3_train = G3_train.reshape(800,1)
train_set.drop(['G3'], axis=1,inplace=True)
x_train = train_set.values


tm1 = np.dot(x_train.T,x_train)
tm1 = np.linalg.pinv(tm1)
tm2 = np.dot(x_train.T,G3_train)
weight = np.dot(tm1,tm2)

G3_test = test_set['G3'].values
G3_test = G3_test.reshape(200,1)
test_set.drop(['G3'], axis=1,inplace=True)
x_test = test_set.values

y_test = np.dot(x_test,weight)

tmp1 = y_test - G3_test
tmp2 = np.square(tmp1) 
tmp3 = tmp2.mean()
RMSE = np.sqrt(tmp3)
print(RMSE)

### (c) RMSE-2 reg

temp1 = np.dot(x_train.T,x_train)
addterm = np.identity(temp1.shape[0],dtype=float)
temp1 = temp1 + addterm/2
temp1 = np.linalg.pinv(temp1)
temp2 = np.dot(x_train.T,G3_train)
wt=np.dot(temp1,temp2)
#print(wt)
y_test2 = np.dot(x_test,wt)
tp1 = y_test2 - G3_test
tp2 = np.square(tp1) 
tp3 = tp2.mean()
RMSE2 = np.sqrt(tp3)
print(RMSE2)


### (d) RMSE-3 + "bias"  : y = W^T * x + b

#bias = np.ones((0,200),dtype=float)
bias = pd.DataFrame(np.ones((201,1),dtype=float),columns=['BIAS'])
bias.drop([0], axis=0,inplace=True)
bias = bias.values
#print(bias)
wbias = pd.DataFrame(np.ones((801,1),dtype=float),columns=['WBIAS'])
wbias.drop([0], axis=0,inplace=True)
wbias = wbias.values
#print(wbias)

x_train2 = np.column_stack((x_train, wbias)) ## x_train + bias(800,1)
tq1 = np.dot(x_train2.T,x_train2)
addterm = np.identity(tq1.shape[0],dtype=float)
tq1 = tq1 + addterm
tq1 = np.linalg.pinv(tq1)
tq2 = np.dot(x_train2.T,G3_train)
wt_2 = np.dot(tq1,tq2)
#print(wt_2)
x_test2 = np.column_stack((x_test, bias))  ## x_test + bias(200,1)
#print(np.dot(wt_2,SH))
#print(np.dot(x_test2,SH))
#y_test3 = np.dot(x_test,wt) + bias
y_test3 = np.dot(x_test2,wt_2) 
#print(y_test3)
tm1 = y_test3 - G3_test
tm2 = np.square(tm1) 
tm3 = tm2.mean()
RMSE3 = np.sqrt(tm3)
print(RMSE3)
#print(y_test3)

### (e) Bayesian Linear Regression

## um = (X^T*X + alpha*I)^(-1) * X^T * y

t1 = np.dot(x_train2.T,x_train2)
alphaI = np.identity(t1.shape[0],dtype=float)
t1 = t1 + alphaI
t1 = np.linalg.pinv(t1)
t2 = np.dot(x_train2.T,G3_train)
weit = np.dot(t1,t2)
#print(weit)

y_test4 = np.dot(x_test2,weit)
te1 = y_test4 - G3_test
te2 = np.square(te1) 
te3 = te2.mean()
RMSE4 = np.sqrt(te3)
print(RMSE4)

### (f) PLOT

y_range = range(0,200)

"""
plt.plot(y_range, G3_test, color='blue', label="Ground Truth")
plt.plot(y_range, y_test, color='orange', label="Linear Regression")
plt.plot(y_range, y_test2, color='green', label="Linear Regression (reg)")
plt.plot(y_range, y_test3, color='red', label="Linear Regression (r/b)")
plt.plot(y_range, y_test4, color='purple', label="Bayesian Linear Regression")


plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.legend(loc='best')
plt.show()
"""
############2.Classication##############

### (a)
cdf = ndf.copy()

df1 = pd.DataFrame(np.zeros((1001,1),dtype=int),columns=['G3F'])
df1.drop([0], axis=0,inplace=True)

for j in range(1,1001):
    if cdf['G3'][j] - 10 >= 0:
        df1['G3F'][j] = 1
    else:
        df1['G3F'][j] = 0

res = pd.concat([cdf,df1],axis=1)
res.drop(['G3'], axis=1,inplace=True)
#print(res)

train_sets = res[:800].copy()
test_sets = res[800:1000].copy()

G3F_train = train_sets['G3F'].values
G3F_train = G3F_train.reshape(800,1)
train_sets.drop(['G3F'], axis=1,inplace=True)
xs_train = train_sets.values
xs_train2 = np.column_stack((xs_train, wbias)) ## xs_train + bias(800,1)

G3F_test = test_sets['G3F'].values
G3F_test = G3F_test.reshape(200,1)
test_sets.drop(['G3F'], axis=1,inplace=True)
xs_test = test_sets.values
xs_test2 = np.column_stack((xs_test, bias))  ## xs_test + bias(200,1)

tep1 = np.dot(xs_train2.T,xs_train2)
addterms = np.identity(tep1.shape[0],dtype=float)
tep1 = tep1 + addterms
tep1 = np.linalg.pinv(tep1)
tep2 = np.dot(xs_train2.T,G3F_train)
weights = np.dot(tep1,tep2)
#print(weights)

ys_test = np.dot(xs_test2,weights)
#print(ys_test)
#print(len(ys_test))
#"""
threshold = 0.1
p1_test = ys_test.copy()
for k in range(0,200):
    if ys_test[k] > threshold:
        p1_test[k] = 1
    else:
        p1_test[k] = 0

threshold = 0.5
p2_test = ys_test.copy()
for l in range(0,200):
    if ys_test[l] >= threshold:
        p2_test[l] = 1
    else:
        p2_test[l] = 0

threshold = 0.9
p3_test = ys_test.copy()
for m in range(0,200):
    if ys_test[m] >= threshold:
        p3_test[m] = 1
    else:
        p3_test[m] = 0

#print(p1_test)
#print(p2_test)
#print(p3_test)
#"""

### (b)  Logistic regression
# hyper parameters
LR = 0.001
EPOCH = 10000

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

train_weight = pd.DataFrame(np.ones((1,62),dtype=int))
train_weight.drop([0], axis = 1,inplace=True)

"""
update = G3F_train - sigmoid(np.dot(xs_train2,train_weight.T))
#print(np.dot(update,SH))
tpe = np.dot(update.T,xs_train2) * (-1)
#print(np.dot(tpe,SH))
train_weight = train_weight - LR * tpe
#print(train_weight)
py_test = sigmoid(np.dot(xs_test2,train_weight.T))
print(py_test)
CrossEntropy = (-1) * (G3F_test.T * np.log(py_test) + 
                        (1 - G3F_test.T) * np.log(1 - py_test) )
loss = CrossEntropy.mean()
print(loss)

"""

bloss = 100.0
record = 0
# update weight

#"""
for epoch in range(EPOCH):
    #print(epoch)
    ## update weight
    update = G3F_train - sigmoid(np.dot(xs_train2,train_weight.T)) ##(800,1)
    tpe = np.dot(update.T,xs_train2) * (-1) ##(1,61)
    train_weight = train_weight - LR * tpe ##(1,61)
    #print(train_weight)
    ## update loss function
    py_test = sigmoid(np.dot(xs_test2,train_weight.T)) ##(200*1)
    CrossEntropy = (-1) * (G3F_test.T * np.log(py_test) + 
                        (1 - G3F_test.T) * np.log(1 - py_test) )
    loss = CrossEntropy.mean()
    if epoch == 6000:
        break
    #print(loss)
#"""   


"""
if record == 1:
    best_weight = train_weight.copy()
    best_loss = bloss
    tpy_test = sigmoid(np.dot(xs_test2,best_weight.T) )
"""

#"""
best_weight = train_weight.copy()
best_loss = bloss
tpy_test = sigmoid(np.dot(xs_test2,best_weight.T) )
#"""

#print(best_loss)
#print(tpy_test)

thresholds = 0.1
pt1_test = tpy_test.copy()
for n in range(0,200):
    if tpy_test[n] >= thresholds:
        pt1_test[n] = 1
    else:
        pt1_test[n] = 0

thresholds = 0.5
pt2_test = tpy_test.copy()
for u in range(0,200):
    if tpy_test[u] >= thresholds:
        pt2_test[u] = 1
    else:
        pt2_test[u] = 0

thresholds = 0.9
pt3_test = tpy_test.copy()
for v in range(0,200):
    if tpy_test[v] >= thresholds:
        pt3_test[v] = 1
    else:
        pt3_test[v] = 0

#print(pt1_test)
#print(pt2_test)
#print(pt3_test)

### (c) plot confusion matrices for (a), (b) threshold is set to 0.5,
"""
if G3F_test[8] == p2_test[8]:
    print("HHHAAA")
    if G3F_test[8] < 0.5:
         print("SHIY")   

print(G3F_test[8])
print(p2_test[8])
"""
#-------------linear regression, threshold = 0.5
TP5_LI = 0
TN5_LI = 0
FP5_LI = 0
FN5_LI = 0
#-------------logistic regression, threshold = 0.5
TP5_LO = 0
TN5_LO = 0
FP5_LO = 0
FN5_LO = 0

for f in range(0,200):
    if G3F_test[f] > 0.5: # actual = 1
        if G3F_test[f] == p2_test[f]: # predict = 1
            TP5_LI = TP5_LI + 1
        else: # predict = 0
            FN5_LI = FN5_LI + 1
    elif G3F_test[f] < 0.5: # actual = 0
        if G3F_test[f] == p2_test[f]: # predict = 0
            TN5_LI = TN5_LI + 1
        else: # predict = 1
            FP5_LI = FP5_LI +1
#"""
print("TP5_LI = ",TP5_LI)
print("TN5_LI = ",TN5_LI)
print("FP5_LI = ",FP5_LI)
print("FN5_LI = ",FN5_LI)
#"""
for f in range(0,200):
    if G3F_test[f] > 0.5: # actual = 1
        if G3F_test[f] == pt2_test[f]: # predict = 1
            TP5_LO = TP5_LO + 1
        else: # predict = 0
            FN5_LO = FN5_LO + 1
    elif G3F_test[f] < 0.5: # actual = 0
        if G3F_test[f] == pt2_test[f]: # predict = 0
            TN5_LO = TN5_LO + 1
        else: # predict = 1
            FP5_LO = FP5_LO +1
#"""
print("TP5_LO = ",TP5_LO)
print("TN5_LO = ",TN5_LO)
print("FP5_LO = ",FP5_LO)
print("FN5_LO = ",FN5_LO)
#"""
## confusion matrix(heatmap)

array5 = [[FN5_LO,FP5_LO],
        [TN5_LO,TP5_LO]]
#print(array5)
df_cm5 = pd.DataFrame(array5, index = ["predict = 0","predict = 1"],
                  columns = ["true = 0","true = 1"])    
plt.figure(figsize = (10,7))          
sn.set(font_scale=1.4)
sn.heatmap(df_cm5, annot=True,annot_kws={"size": 16}, vmin=0, vmax=200, fmt='g')

#sn.heatmap(df_cm, annot=True, vmin=0, vmax=200, fmt='g')  

### (d) Repeat (c) threshold is set to 0.9.

#-------------linear regression, threshold = 0.9
TP9_LI = 0
TN9_LI = 0
FP9_LI = 0
FN9_LI = 0
#-------------logistic regression, threshold = 0.9
TP9_LO = 0
TN9_LO = 0
FP9_LO = 0
FN9_LO = 0

for f in range(0,200):
    if G3F_test[f] > 0.5: # actual = 1
        if G3F_test[f] == p3_test[f]: # predict = 1
            TP9_LI = TP9_LI + 1
        else: # predict = 0
            FN9_LI = FN9_LI + 1
    elif G3F_test[f] < 0.5: # actual = 0
        if G3F_test[f] == p3_test[f]: # predict = 0
            TN9_LI = TN9_LI + 1
        else: # predict = 1
            FP9_LI = FP9_LI +1
#"""
print("TP9_LI = ",TP9_LI)
print("TN9_LI = ",TN9_LI)
print("FP9_LI = ",FP9_LI)
print("FN9_LI = ",FN9_LI)
#"""
for f in range(0,200):
    if G3F_test[f] > 0.5: # actual = 1
        if G3F_test[f] == pt3_test[f]: # predict = 1
            TP9_LO = TP9_LO + 1
        else: # predict = 0
            FN9_LO = FN9_LO + 1
    elif G3F_test[f] < 0.5: # actual = 0
        if G3F_test[f] == pt3_test[f]: # predict = 0
            TN9_LO = TN9_LO + 1
        else: # predict = 1
            FP9_LO = FP9_LO +1
#"""
print("TP9_LO = ",TP9_LO)
print("TN9_LO = ",TN9_LO)
print("FP9_LO = ",FP9_LO)
print("FN9_LO = ",FN9_LO)
#"""

## confusion matrix(heatmap)
array9 = [[FN9_LO,FP9_LO],
        [TN9_LO,TP9_LO]]
#print(array9)
df_cm9 = pd.DataFrame(array9, index = ["predict = 0","predict = 1"],
                  columns = ["true = 0","true = 1"])   
plt.figure(figsize = (10,7))           
sn.set(font_scale=1.4)
sn.heatmap(df_cm9, annot=True,annot_kws={"size": 16}, vmin=0, vmax=200, fmt='g')

### (e)

#-------------linear regression, threshold = 0.1
TP1_LI = 0
TN1_LI = 0
FP1_LI = 0
FN1_LI = 0
#-------------logistic regression, threshold = 0.1
TP1_LO = 0
TN1_LO = 0
FP1_LO = 0
FN1_LO = 0

for f in range(0,200):
    if G3F_test[f] > 0.5: # actual = 1
        if G3F_test[f] == p1_test[f]: # predict = 1
            TP1_LI = TP1_LI + 1
        else: # predict = 0
            FN1_LI = FN1_LI + 1
    elif G3F_test[f] < 0.5: # actual = 0
        if G3F_test[f] == p1_test[f]: # predict = 0
            TN1_LI = TN1_LI + 1
        else: # predict = 1
            FP1_LI = FP1_LI +1
#"""
print("TP1_LI = ",TP1_LI)
print("TN1_LI = ",TN1_LI)
print("FP1_LI = ",FP1_LI)
print("FN1_LI = ",FN1_LI)
#"""
for f in range(0,200):
    if G3F_test[f] > 0.5: # actual = 1
        if G3F_test[f] == pt1_test[f]: # predict = 1
            TP1_LO = TP1_LO + 1
        else: # predict = 0
            FN1_LO = FN1_LO + 1
    elif G3F_test[f] < 0.5: # actual = 0
        if G3F_test[f] == pt1_test[f]: # predict = 0
            TN1_LO = TN1_LO + 1
        else: # predict = 1
            FP1_LO = FP1_LO +1
#"""
print("TP1_LO = ",TP1_LO)
print("TN1_LO = ",TN1_LO)
print("FP1_LO = ",FP1_LO)
print("FN1_LO = ",FN1_LO)
#"""
## Accuracies = (TP + TN)/(P+N)
AC1_LI = (TP1_LI + TN1_LI) / 200
AC1_LO = (TP1_LO + TN1_LO) / 200
AC5_LI = (TP5_LI + TN5_LI) / 200
AC5_LO = (TP5_LO + TN5_LO) / 200
AC9_LI = (TP9_LI + TN9_LI) / 200
AC9_LO = (TP9_LO + TN9_LO) / 200
#"""
print("AC1_LI",AC1_LI)
print("AC1_LO",AC1_LO)
print("AC5_LI",AC5_LI)
print("AC5_LO",AC5_LO)
print("AC9_LI",AC9_LI)
print("AC9_LO",AC9_LO)
#"""
## Precisons = (TP)/(TP+FP)
PC1_LI = TP1_LI / (TP1_LI + FP1_LI)
PC1_LO = TP1_LO / (TP1_LO + FP1_LO)
PC5_LI = TP5_LI / (TP5_LI + FP5_LI)
PC5_LO = TP5_LO / (TP5_LO + FP5_LO)
PC9_LI = TP9_LI / (TP9_LI + FP9_LI)
PC9_LO = TP9_LO / (TP9_LO + FP9_LO)
#"""
print("PC1_LI",PC1_LI)
print("PC1_LO",PC1_LO)
print("PC5_LI",PC5_LI)
print("PC5_LO",PC5_LO)
print("PC9_LI",PC9_LI)
print("PC9_LO",PC9_LO)
#"""
############3.Hidden Test Set##############
fd = pd.read_csv("test_no_G3.csv", index_col=0) # delete first column

fd = pd.get_dummies(fd)

## normalize
nfd = fd

for i in fd.columns:
    n = nfd[i].values
    mu = n.mean() # mean
    std = n.std() # standard deviations

    normalized = (n - mu) / std
    nfd[i] = normalized

xH_test = nfd.copy()

Tbias = pd.DataFrame(np.ones((45,1),dtype=float),columns=['TBIAS'])
Tbias.drop([0], axis=0,inplace=True)
Tbias = Tbias.values

xH_test2 = np.column_stack((xH_test, Tbias)) ## xH_test + bias(44,1)

### (a) Apply the model from 1. (d) ?? where is alpha
HT_weight = wt_2.copy()
HT_weight = np.delete(HT_weight,41,axis=0) # no 'guardian_other'
HT_weight = np.delete(HT_weight,31,axis=0) # no 'Fjob_health'
#print(len(HT_weight))
HT_test = np.dot(xH_test2,HT_weight)
#HT_test = np.dot(xH_test,HT_weight) + bias
#print(xH_test)
#print(HT_test)

ID = []
for i in range(1,45):
    ID.append(str(1000 + i))
Result1 = []
for i in range(0,44):
    Result1.append(str(HT_test.flatten()[i]))
#print(Result)

file1 = open('107062512_1.txt','w')

for i in range(0,44):
   file1.write(ID[i])
   file1.write("\t")
   file1.write(Result1[i])
   file1.write("\n")

### (b) Apply the model from 2. (b)

HT2_weight = best_weight.T.copy()
HT2_weight = HT2_weight.values

HT2_weight = np.delete(HT2_weight,41,axis=0) # no 'guardian_other'
HT2_weight = np.delete(HT2_weight,31,axis=0) # no 'Fjob_health'

HT2_test  = sigmoid(np.dot(xH_test2,HT2_weight))
#print(HT2_test)

Result2 = []
for i in range(0,44):
    Result2.append(str(HT2_test.flatten()[i]))
#print(Result2)

file2 = open('107062512_2.txt','w')

for i in range(0,44):
   file2.write(ID[i])
   file2.write("\t")
   file2.write(Result2[i])
   file2.write("\n")

