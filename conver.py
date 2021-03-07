import pandas as pd
import re
import numpy as np

class conversion():
    def input(self, ilist):
        ### input is the list of words from test file
        s1 = re.sub(r"[']",'',ilist[1:-1])
        lid = []
        for i,xx in enumerate(s1.split(',')):
            for j,x in enumerate(xx):
                lid.append(i)
        self.rinput = s1.split(',')
        self.lid = lid
        self.output = ['O']*len(self.rinput)
        return re.sub(',','',s1)
       
    def assign(self, attribute,start,end=-1):
        atta = attribute.upper().replace(' ',"_")
        idx = self.lid[start]
        self.output[idx] = "B_"+atta
        idx = self.lid[end]
        self.output[idx] = "I_"+atta

    def getoutput(self):
        ## TODO
        ### iterate over length of attributes 
        ## using assign function 
        return self.output
