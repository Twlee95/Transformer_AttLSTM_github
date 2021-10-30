# Deep learning for stock price prediction
![주식_이미지](https://user-images.githubusercontent.com/76574427/139482743-d017ba50-dacd-4642-8560-29f2af7169b4.jpg)

주가 예측은 힘들지만 금융산업에서 중요한 문제임.

최근 딥러닝을 이용한 주가예측 알고리즘이 뛰어난 성능을 보이고 있음.

아래에는 딥러닝을 이용한 방식 중 AttentionLSTM, TransformerEncoder를 이용한 예측방식을 소개함

## AttentionLSTM

## TransformerEncoder


## Experiment setting



## Datasets
```
from pandas_datareader import data as pdr
import yfinance as yfin

yfin.pdr_override()
self.data = pdr.get_data_yahoo(self.symbol, start=self.start, end=self.end)
```
pandas_datareader를 이용하여 야후 파이낸스에 있는 데이터셋을 위와같은 방법으로 불러올 수 있음.
yahoofinance에서 제공되는 정보(Open, Close, High, Low, Volume, AdjClose)를 불러옴.

## 예측 결과 예시
코스닥 주가 데이터를 이용한 방식
![예측 결과 임지](https://user-images.githubusercontent.com/76574427/139482798-87decde6-a9b9-458d-9e58-f43469498780.png)
