import re
import pandas as pd
import numpy as np


# clean dataset

clean_qus=[]
clean_ans=[]
word=[]
df=pd.read_csv('final.csv')
value=0
for index,row in df.iterrows():
    new=str(row['qus']).lower()

    new1 = str(row['ans']).lower()


    replaced = re.sub("[?/!@.)({}[#$%~^&*=,'-:;1234567890]", "", new)
    replaced=replaced.replace("'t",' not')
    replaced = replaced.replace("'s", ' is')


    replaced1 = re.sub("[?/!@.)({}[#$%~^&*=,'-:;1234567890]", "", new1)
    replaced1 = replaced1.replace("'t", ' not')
    replaced1 = replaced1.replace("'s", ' is')

    if len(replaced.split(" "))<50 and len(replaced1.split(" "))<40:
        value+=1
        clean_qus.append(replaced.split(" "))
        st=replaced.split(" ")
        for x in range(0,len(st)):
            word.insert(1,st[x])

        clean_ans.append(replaced1.split(" "))
        st = replaced1.split(" ")
        for x in st:
            word.insert(1,x)

    print(index)
    if value==10000:
        break


u_characters = set(word)
print("Vocab size ->",len(u_characters)) #total unique word
char2num = dict(zip(u_characters, range(len(u_characters)))) #  A dictunary with { "word":0} char to number
#print(char2num)

char2num['<PAD>'] = len(char2num)
char2num['<GO>'] = len(char2num)
num2char = dict(zip(char2num.values(), char2num.keys()))#  A dictunary with { 0:"word"}  number to char
#print(num2char)


from sklearn.externals import joblib
joblib.dump(clean_qus,"data/input.pkl")
joblib.dump(clean_ans,"data/output.pkl")
#joblib.dump(set(word),"data/vocab.pkl")
joblib.dump(char2num,"data/char2num.pkl")
joblib.dump(num2char,"data/num2char.pkl")





