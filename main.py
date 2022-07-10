from complex_word_checker import ComplexWordCheck
import sys

file_path=sys.argv[1]
with open(file_path,"r",encoding="utf-8") as f:
    sentences=[line.replace("\n","") for line in f.readlines()]
cwc=ComplexWordCheck()
sentences=cwc.complex_check(sentences)
for sentence in sentences:
    print(sentence)