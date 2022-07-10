# ComplexWordChecker

## 概要
このプログラムは文中に現れる難しい単語を判別し、色付きで表示するプログラムになります。
難しい単語を判別する際、事前に日本語教育語彙表によって単語難易度推定タスクを学習済みのBERTとそのBERTによって構築された辞書を使います。
ここで、辞書だけでなくBERTを用いるのは辞書外の未知語に対しても難易度を付与できるようにするためです。

## 目的
この難しい単語を判別するプログラムは、日本語を母国語としない人に対し、文で説明する際に文中の単語難易度チェックに使えるようにするために作成しました。

## 動作確認済み環境
windows10\
python3.8.8\
pip list
```
torch==1.9.0
transformers==4.17.0
mecab-python3==1.0.3
```
## 使い方
まず、このリポジトリを保存します。\
そしてBERTの学習済みデータ(https://github.com/tsutsui6Electronics/ComplexWordChecker/releases/tag/BERT_complex_word_estimator_param_v1)
を```ComplexWordChecker/```に保存します。\
次に予測したい文を記述したテキストファイルを用意します。\
形式は以下になります。\
なお複数行になっていても大丈夫です。\
sample.txt\
```
地下の岩盤には様々な要因により力（ひずみ）がかかっており、急激な変形によってこれを解消する現象が地震である。地球の内部で起こる地質現象（地質活動）の一種。地震に対して、地殻が非常にゆっくりとずれ動く現象を地殻変動と呼ぶ。
```
\
最後に以下のコマンドを```ComplexWordChecker/```で実行します。\
```python main.py sample.txt```

## 実行結果例
難しいと判断された単語は赤で示されます。
![image](https://user-images.githubusercontent.com/55880071/178144939-0415e687-8635-46ea-9d0b-2917ae508bfb.png)


