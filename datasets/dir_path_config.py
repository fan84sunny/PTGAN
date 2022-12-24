import os
#用來參數化dir path。


@static method
def pwd():
    return os.getcwd()

def AIC21():
    return '/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AIC21'
    #return os.getcwd()+'/AIC21'

def datasets():
    return '/home/michuan.lh/datasets'
    #return os.getcwd()+'/datasets'
"""
問題
1. github上的 pip install忘記+ '-r'參數。
2. 為什麼datasets資料夾是分散的?
3. 可以列出train的花費時間嗎?
"""
