import os
import Levenshtein
import numpy as np

splice = open("splice.data")

lable = []
DNA = []

for i in splice:
    lable.append(i.split(",")[0])
    DNA.append(i.split(",")[2].replace(" ", "").replace("\n", ""))

print(DNA[1])
print(len(DNA[1]))
for i in DNA[:1000]:
    print(Levenshtein.hamming(DNA[0], i))

os.system("pause")

# 
# 