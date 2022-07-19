import re
import torch
import MeCab
import unidic,fugashi
from colorama import Fore
from transformers import BertJapaneseTokenizer
from transformers import BertForSequenceClassification


class ComplexWordCheck():
    def __init__(self):
        self.tagger = fugashi.Tagger('-d "{}"'.format(unidic.DICDIR))
        self.max_length = 5
        self.model_name="cl-tohoku/bert-base-japanese-whole-word-masking"
        self.tokenizer=BertJapaneseTokenizer.from_pretrained(self.model_name)
        self.model=BertForSequenceClassification.from_pretrained(self.model_name,num_labels=6)
        self.model.load_state_dict(torch.load("BERT_complex_word_estimator.pth",map_location=torch.device('cpu')))
        self.complex_dict=dict()
        with open("Simple_PPDM_mod.txt",encoding="utf-8") as f:
            for line in f.readlines():
                #word_info[0]:word
                #word_info[1]:difficulty
                word_info=line.replace("\n","").split("\t")
                if word_info[0] not in self.complex_dict.keys():
                    self.complex_dict[word_info[0]]=int(word_info[1])
                    
    def word_extract(self,sentense):
        #形態素解析
        result_sentense=self.tagger.parse(sentense).split("\n")
        
        edit_sentence=""
        #単語と認識しない品詞
        ignore=["助詞","助動詞","補助記号","接頭辞","接尾辞"]
        
        word_set=set()
        for word_info in result_sentense:
            
            #空文字、EOSの場合は単語として扱わない
            if word_info=='EOS' or word_info=='':
                continue
            edit_sentence+=word_info.split("\t")[0]+" "   
            #数値は単語として扱わない
            if re.compile(r'^[0-9]+$').fullmatch(word_info.split("\t")[0]):
                continue
            a=word_info.split("\t")[1].split(",")
            """
            
            a：['名詞', '普通名詞', '副詞可能', '', '', '', 'キョウ', '今日', '今日',\
            'キョー', '今日', 'キョー', '和', '""', '""', '""', '""', '""', '""', '体',\
            'キョウ', 'キョウ', 'キョウ', 'キョウ', '"1"', '"C3"', '""', '2509094191768064', '9128']
            
            """
            #単語の抽出:
            if a[0] not in ignore:
                if not re.compile(r'^[あ-ん]+$').fullmatch(word_info.split("\t")[0]):     #ひらがなでない場合はすべて抽出
                    if len(a)>6:
                        word_set.add(a[10])
                    else:
                        word_set.add(word_info.split("\t")[0])
                else:   #ひらがなの場合は非自立可能語であれば抽出
                    if len(a)>6:
                        if a[1]!="非自立可能":
                            word_set.add(a[10])
        
        return edit_sentence,word_set
    
    def complex_check(self,sentences):
        edit_sentences=[]
        for sentence in sentences:
            edit_sentence,words=self.word_extract(sentence)
            for word in words:
                if word in self.complex_dict.keys():    #辞書に存在する単語
                    if self.complex_dict[word] > 4:
                        edit_sentence=edit_sentence.replace(word,Fore.LIGHTYELLOW_EX+word+Fore.WHITE)
                else:   #辞書にない単語はBERTで予測する
                    self.model.to("cpu")
                    encoding = self.tokenizer(
                            word,
                            max_length=self.max_length, 
                            padding='max_length',
                            truncation=True)
                    pred=self.model(torch.tensor([encoding["input_ids"]]))
                    _, predicted = torch.max(pred[0], 1)
                    if predicted.item() > 4:
                        edit_sentence=edit_sentence.replace(word,Fore.LIGHTCYAN_EX+word+Fore.WHITE)
            edit_sentences.append(edit_sentence.replace(" ",""))
        return edit_sentences
