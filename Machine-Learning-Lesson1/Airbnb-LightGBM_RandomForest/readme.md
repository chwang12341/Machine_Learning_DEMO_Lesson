## Machine Learning - Airbnb Taiwan 數據分析與建模 - 視覺化 輕鬆看出在台灣Airbnb哪個(月/天)的房價平均最高  -  課程筆記與實作教學



這篇是我學習完課程後，整理觀念和上課筆記，並利用課程所學應用於我另外找的數據集上，確保自己真的了解與自己真的能夠實用所學



## 1. Airbnb數據集來源 - 台灣 台北



這份數據集是我從網路上取得的，它並不是Airbnb官方的網站，但它收集了許多國家的Airbnb資料: http://insideairbnb.com/get-the-data.html



## 2. 實作



### Step 1: 導入套件

```Python
## 導入所需的套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```



### Step 2: 導入calendar數據集

```Python
## 導入日期和價格的數據集
calendar = pd.read_csv('data/calendar.csv.gz')

## 顯示數據集
calendar
```

![image1](images\image1.PNG)





### Step 3: 數據集資訊

```Python
## 顯示數據集前十筆
calendar.head(10)
```

![image2](images\image2.PNG)





```Python
## 顯示數據大小
calendar.shape
```

**執行結果**

```
(1919789, 7)
```





**筆記:** 建議把數據的時間格式整理成 xxxx-xx-xx 這樣就能對時間做一些處理

```Python
## 筆記: 建議把數據的時間格式整理成 xxxx-xx-xx 這樣就能對時間做一些處理

## 最舊的數據時間和最新的數據時間
calendar.date.min(), calendar.date.max()
```

**執行結果**

```
('2020-12-31', '2022-01-02')
```







#### 補充: 如何將數據集中的日期轉換成Pandas可以操作的格式

```Python## 補充用法
## 將數據集中的日期轉換成Pandas可以操作的格式
calendar['date'] = pd.to_datetime(calendar['date'])

## 顯示數據集資訊
calendar.info()
```

**執行結果**

```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1919172 entries, 0 to 1919788
Data columns (total 7 columns):
 #   Column          Dtype         
---  ------          -----         
 0   listing_id      int64         
 1   date            datetime64[ns]
 2   available       object        
 3   price           object        
 4   adjusted_price  object        
 5   minimum_nights  float64       
 6   maximum_nights  float64       
dtypes: datetime64[ns](1), float64(2), int64(1), object(3)
memory usage: 117.1+ MB
```







### Step 4: 缺失值(NaN)操作

```Python
## 檢查每個特徵(列別)中有幾個缺失直(NaN)
calendar.isnull().sum()
```

**執行結果**

```
listing_id          0
date                0
available           0
price               0
adjusted_price      0
minimum_nights    617
maximum_nights    617
dtype: int64
```

+ 從結果可以看出"minimum_nights "和"maximum_nights "各有617個缺失值(NaN)



```Python
## 移除掉有缺失值的數據行
calendar = calendar.dropna()

## 檢查刪除缺失數據後的數據集大小
calendar.shape
```

**執行結果**

```
(1919172, 7)
```



### Step 5: 將價格欄位處理成可以計算的格式 - 將字符串轉換成浮點數



+ 查看數據集中特徵的數據類型

```Python
## 顯示數據集類型資訊
calendar.info()
```

**執行結果**

```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1919172 entries, 0 to 1919788
Data columns (total 7 columns):
 #   Column          Dtype         
---  ------          -----         
 0   listing_id      int64         
 1   date            datetime64[ns]
 2   available       object        
 3   price           object        
 4   adjusted_price  object        
 5   minimum_nights  float64       
 6   maximum_nights  float64       
dtypes: datetime64[ns](1), float64(2), int64(1), object(3)
memory usage: 117.1+ MB
```



結果中的price 欄位為object類型，也就是字符串類型，我們要將它轉換成浮點數類型



+ 將price中的符號拿掉

```Python
## 將price中的"$"、","符號替換成空的
calendar['price'] = calendar['price'].str.replace('$', '')
calendar['price'] = calendar['price'].str.replace(',', '')

## 顯示數據集
calendar.head()

```



![image3](images\image3.PNG)



+ 將"price"轉換成浮點數類型

```Python## 將"price"轉換成浮點數類型
calendar['price'] = calendar['price'].astype('float64')


## 顯示數據集資訊
calendar.info()

```



**執行結果**

```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1919172 entries, 0 to 1919788
Data columns (total 7 columns):
 #   Column          Dtype         
---  ------          -----         
 0   listing_id      int64         
 1   date            datetime64[ns]
 2   available       object        
 3   price           float64       
 4   adjusted_price  object        
 5   minimum_nights  float64       
 6   maximum_nights  float64       
dtypes: datetime64[ns](1), float64(3), int64(1), object(2)
memory usage: 117.1+ MB
```





### Step 6: 視覺化 - 月份跟價錢的關聯圖



+ 依據時間來分組，並提取我們指定的指標(ex. price)，進行平均計算

```Python
## 依據"date"並提取月份進行分組，然後將同個月份的價錢數據取平均當成新數據
mean_of_month = calendar.groupby(calendar['date'].dt.strftime('%B'))['price'].mean()

print('Type: ', type(mean_of_month))
## 顯示數據
mean_of_month 
```

**執行結果**: 返回一個Series

```
Type:  <class 'pandas.core.series.Series'>
```

```
date
April        2665.031882
August       2859.538522
December     2891.047534
February     2651.311152
January      2543.447503
July         2873.033074
June         2708.768898
March        2591.639609
May          2665.548945
November     2880.687663
October      2891.349636
September    2868.567611
Name: price, dtype: float64
```





+ 視覺化

```Python
## 視覺化
mean_of_month.plot(kind = 'bar', figsize = (12, 8))
```

![image6](images\image6.PNG)



**從結果中可以看出在台灣的Airbnb哪個月份的房價最高**





#### 補充: 時間格式符表

| 格式符 | 描述                                          |
| ------ | --------------------------------------------- |
| %a     | 星期的英文縮寫 ex. Mon                        |
| %A     | 星期的英文全寫 ex. Monday                     |
| %b     | 月份的英文縮寫 ex. Jan                        |
| %B     | 月份的英文全寫 ex. January                    |
| %c     | datetime的字符串表示 ex. 01/25/21 22:10:47    |
| %d     | 返回現在時間是這個月的第幾天                  |
| %f     | 用微秒呈現 範圍會介於0~999999                 |
| %H     | 以24小時格式表示目前的小時                    |
| %I     | 以12小時格式表示目前的小時                    |
| %j     | 返回今天是今年的第幾天 範圍介於001~366        |
| %m     | 返回月份 範圍介於0~12                         |
| %M     | 返回分鐘數 範圍介於0~59                       |
| %P     | 返回上午或是下午  ex. AM or PM                |
| %S     | 返回秒數 範圍介於0~61                         |
| %W     | 返回這週是今年的第幾週 以週一為第一天         |
| %U     | 返回這週是今年的第幾週 以週日為第一天         |
| %w     | 返回今天在這週的天數 範圍介於0~6，6表示星期日 |
| %x     | 日期的字符串表示 ex. 01/25/21                 |
| %X     | 時間的字符串表示 ex. 23:02:58                 |
| %y     | 用兩個數字來表示年份 ex. 21                   |
| %Y     | 用四個數字來表示年份 ex. 2021                 |



### Step 7: 視覺化 - 星期幾跟價錢的關聯圖



+ 構建一個新的列: 表示數據中的日期是一週中的位置 Monday是0 ~ Sunday是7

```Python
## 構建一個新的列: 表示數據中的日期是一週中的位置 Monday是0 ~ Sunday是7
calendar['dayofweek'] = calendar.date.dt.weekday
## 顯示數據集
calendar
```

![image7](images\image7.PNG)





+ 根據"dayofweek"計算其他指標的平均值

```Python
## 根據"dayofweek"計算其他指標的平均值
calendar.groupby(['dayofweek']).mean()
```

![image8](images\image8.PNG)





+ 更新一下dayofweek的數據名稱 -> 將數字轉成對應的星期

```Python
## 更新一下dayofweek的數據名稱 -> 將數字轉成對應的星期
calendar['dayofweek'] = calendar['dayofweek'].replace([0, 1, 2, 3, 4, 5, 6], ['Monday','Tueday','Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

## "dayofweek"計算其他指標的平均值
calendar.groupby(['dayofweek']).mean()

```



![image9](images\image9.PNG)



+ 重新索引成我們要的順序

```Python
## 重新索引成我們要的順序
order = ['Monday','Tueday','Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
## 將"dayofweek"變成索引列
price_week = calendar.groupby(['dayofweek']).mean().reindex(order)
price_week
```

![image10](images\image10.PNG)





+ 將price以外的列都移除掉

```Python
## 將price以外的列都移除掉
price_week = price_week.drop(['listing_id', 'minimum_nights', 'maximum_nights'], axis = 1)

price_week
```

![image11](images\image11.PNG)





+ 視覺化

```Python
## 視覺化
price_week.plot()
```

![image12](images\image12.PNG)

把前面的程式彙總在一起，一起執行

```Python
## 創建一個新的欄位'dayofweek'，使用dt.weekday將數據集中的'date'欄位轉換成星期幾
calendar['dayofweek'] = calendar.date.dt.weekday
## 創建要當x軸刻度的名稱串列
cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
## 只要'dayofweek'跟'price'欄位
price_week = calendar[['dayofweek', 'price']]
## 使用reindex就會把其他欄位的值設成NaN，所以我們使用plt後處理的方式將x軸的刻度名稱補上
price_week = calendar.groupby(['dayofweek']).mean()#.reindex(cats)
# ## 把'listing_id','minimum_nights', 'maximum_nights'欄位拿掉
price_week.drop(['listing_id', 'minimum_nights', 'maximum_nights'], axis = 1, inplace = True)
price_week.plot()
## 更改x軸的刻度
## 設定範圍
ticks = list(range(0, 7, 1))
x_labels = 'Mon Tues Wed Thurs Fri Sat Sun'.split(' ')
plt.xticks(ticks, x_labels)

plt.xlabel('Day Of Week')
plt.ylabel('Price')
```

**執行結果**

![image13](images\image13.PNG)



