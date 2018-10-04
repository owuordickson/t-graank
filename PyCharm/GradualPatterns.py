# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:31:16 2015

@author: Olivier + modif MJL+MR 140316
"""
import csv
import bisect
import numpy as np
import time
import gc


def Trad(fileName):
    temp=[]
    with open(fileName, 'r') as f:
        reader = csv.reader(f,delimiter=' ')
        temp = list(reader)
    #print(temp)
    if temp[0][0].replace('.','',1).isdigit() or temp[0][0].isdigit():
        return [[float(temp[j][i]) for j in range(len(temp))] for i in range(len(temp[0]))]
    else:
        if temp[0][1].replace('.','',1).isdigit() or temp[0][1].isdigit():
            return [[float(temp[j][i]) for j in range(len(temp))] for i in range(1,len(temp[0]))]
        else:
            for i in range(len(temp[0])):
                print(str(i+1)+' : '+temp[0][i])
            return [[float(temp[j][i]) for j in range(1,len(temp))] for i in range(len(temp[0]))]
    
    
    

def GraankInit(T,eq=False):
    res=[]
    n=len(T[0])
    #print T
    for i in range(len(T)):
        npl=str(i+1)+'+'
        nm=str(i+1)+'-'
        tempp=np.zeros((n,n),dtype= 'bool')
        tempm=np.zeros((n,n),dtype= 'bool')
        #print i
        for j in range(n):
            for k in range(j+1,n):
                if T[i][j]>T[i][k]:
                    tempp[j][k]=1
                    tempm[k][j]=1
                else:
                    if T[i][j]<T[i][k]:
                        #print (j,k)
                        tempm[j][k]=1
                        tempp[k][j]=1
                    else:
                        if eq:
                            tempm[j][k]=1
                            tempp[k][j]=1
                            tempp[j][k]=1
                            tempm[k][j]=1
        res.append((set([npl]),tempp))
        res.append((set([nm]),tempm))
    #print res
    #print "a"
    return res

def SetMax(R):
    i=0
    k=0
    test=0
    Cb=R
    while(i<len(Cb)-1):
            test=0
            k=i+1
            while(k<len(Cb)):
                if(Cb[i].issuperset(Cb[k]) or Cb[i]==Cb[k]):
                    del Cb[k]
                else:
                    if Cb[i].issubset(Cb[k]):
                        del Cb[i]
                        test=1
                        break
                k+=1
            if test==1:
                continue
            i+=1
    return Cb

def inv(s):
    i=len(s)-1
    if s[i]=='+':
        return s[0:i]+'-'
    else:
        return s[0:i]+'+'

def APRIORIgen(R,a,n):
    res=[]
    test=1
    temp=set()
    temp2=set()
    #print"a"
    I=[]
    if(len(R)<2):
        return []
    Ck=[x[0] for x in R]
    #print"b"
    for i in range(len(R)-1):
        #print"c"
        #print len(R)
        for j in range(i+1,len(R)):
            temp=R[i][0]|R[j][0]
            invtemp={inv(x) for x in temp}
            #print invtemp
            #print"d"+str(j)
            if ((len(temp)==len(R[0][0])+1) and (not (I!=[] and temp in I)) and (not (I!=[] and invtemp in I))):
                test=1
                #print "e"
                for k in temp:
                    temp2=temp-set([k])
                    invtemp2={inv(x) for x in temp2}
                    if not temp2 in Ck and not invtemp2 in Ck:
                        test=0
                        break
                if test==1:
                    m=R[i][1]*R[j][1]
                    t=float(np.sum(m))/float(n*(n-1.0)/2.0)
                    if t >a:
                        res.append((temp,m))
                I.append(temp)
                gc.collect()
                    
    #print "z"
    return res
            
def Graank(T,a,eq=False):
    res=[]
    res2=[]
    temp=0
    n=len(T[0])
    G=GraankInit(T,eq)
    #print G
    for i in G:
        temp=float(np.sum(i[1]))/float(n*(n-1.0)/2.0)
        if temp<a:
            G.remove(i)
#        else:
#            res.append(i[0])
    while G!=[]:
        G=APRIORIgen(G,a,n)
        #print G
        i=0
        while i<len(G) and G!=[]:
            temp=float(np.sum(G[i][1]))/float(n*(n-1.0)/2.0)
            #print temp
            if temp<a:
                del G[i]
            else:
                #print i
                z=0
                while z <(len(res)-1):
                    if res[z].issubset(G[i][0]):
                        del res[z]
                        del res2[z]
                    else:
                        z=z+1
                res.append(G[i][0])
                res2.append(temp)
                i+=1
                #print G
                #res=SetMax(res)
#                j=0
#                k=0
#                test=0
#        while(j<len(res)-1):
#                test=0
#                k=j+1
#                while(k<len(res)):
#                    if(res[j].issuperset(res[k]) or res[j]==res[k]):
#                        del res[k]
#                        del res2[k]
#                    else:
#                        if res[j].issubset(res[k]):
#                            del res[j]
#                            del res2[j]
#                            test=1
#                            break
#                    k+=1
#                if test==1:
#                    continue
#                j+=1
        #print res
    return res,res2


        
def main(filename1,supmin1,eq=False):
    D1,S1=Graank(Trad(filename1),supmin1,eq)
    print('D1 : '+filename1)
    for i in range(len(D1)):
        print(str(D1[i])+' : '+str(S1[i]))

def fuse(L):
    Res=L[0][:][:4000]
    for j in range(len(L[0])):
        for i in range(1,len(L)):
            Res[j]=Res[j]+L[i][j][:4000]
    return Res

def fuseTrad(L):
    temp=[]
    for i in L:
        temp.append(Trad(i))
    return fuse(temp)
    
    
def allComb(L,supmin,eq=False):
    ll=L
    lT=[]
    for i in ll:
        lT.append(Trad(i))
    lT.append(fuseTrad(ll))
    ll.append('all')
    ld=[]
    for i in range(len(ll)):
        D1,S1=Graank(lT[i],supmin,eq)
        print(ll[i])
        for i in range(len(D1)):
            print(str(D1[i])+' : '+str(S1[i]))
        ld.append(D1)
    for i in range(len(ll)-1):
        for j in range(i+1,len(ll)):
            Res=MBDLL(ld[i],ld[j])
            print(str(ll[j])+' vs '+str(ll[i]))
            for a in range(len(Res)):
                print("Bordure "+str(a)+ " : L="+str(Res[a][0])+", R="+str(Res[a][1]))
            Res=MBDLL(ld[j],ld[i])
            print(str(ll[i])+' vs '+str(ll[j]))
            for a in range(len(Res)):
                print("Bordure "+str(a)+ " : L="+str(Res[a][0])+", R="+str(Res[a][1]))
                
def getSupp(T,s,eq=False):
    n=len(T[0])
    res=0
    for i in range(len(T[0])):
        for j in range(i+1,len(T[0])):
            temp=1
            tempinv=1
            for k in s:
                x=int(k[0:(len(s)-1)])-1
                if(k[len(s)-1]=='+'):
                    if(T[x][i]>T[x][j]):
                        tempinv=0
                    else:
                        if(T[x][i]<T[x][j]):
                            temp=0
                else:
                    if(T[x][i]<T[x][j]):
                        tempinv=0
                    else:
                        if(T[x][i]>T[x][j]):
                            temp=0
                if(T[x][i]==T[x][j] and not eq):
                    temp=0
                    tempinv=0
            res=res+temp+tempinv
    return float(res)/float(n*(n-1.0)/2.0)
                        

#main('FluTopicData-testsansdate-blank.csv',0.5,False)
main('ndvi_file.csv',0.5,False)