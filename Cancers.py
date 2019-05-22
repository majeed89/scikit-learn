'''
 زیاد کردنی ئەو کتێبخانانەی کە ئێمە بەکاریییان ئەهێنین 
پانداس بۆلۆدی داتا کە بەشێوەی فرەیم پێمان دەدات
وا نامپای بۆ چارەسەری داتای نادیار و بەشێوەی ئەرەی داتامان پێ دەدات 
'''
import pandas as pd 
import numpy as np
Cancer = pd.read_csv("Cancer.csv")#load and read data 


#-------------missing value-----------------------------------
'''
چارەسەری داتای نادیار لە ڕێکەی سکێڵ لێرنین سەرەتا پاکێجەکە بانگ دەکەین
وە لەڕێگەی میسۆدی missing_values وە داتای نادیار لە ناوی ئەری numpy دەدۆزینەوە
وە بەچەند ڕێگەیەک چارەسەری داتای نادیار دەکرێت یەکەم ئەڤەرەیجی داتاکە وەر دەگرین 
ئەمە بۆ ژمارە دەبێت یاخود کامیان زۆر تر هاتبوو یاخود دیلێتی دەکەین
بۆ داتایەک کە تێکستی تیا بێت ئەبێتmost_frequent بەکاربێنن  
'''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
imputer=imputer.fit(Cancer)
Cancer=imputer.transform(Cancer)
'''
لە چارەسەرکرنی داتای نادیار داتاکانمان لە فرەیمەوە بوون بە ئەرەی بۆیە لێرە دەیکەینەوە
بە ئەرەی 
'''
Cancer=  pd.DataFrame(Cancer)
#====================catagorais==========================================
'''
لیرە داتاکانمان دەکەینبە ژمارەی ئەگەر تێکستی تیا بوو سەرەتا پاکێچە کە بانگ دەکەین
دواتر چەند کۆڵۆمان هەبوو بۆ هەموو کۆڵۆمەکان داتاکە دەکینە ژمارەی و ناوی کۆڵۆمنی نوێ لی دەنێن
وە بۆیە کۆڵۆمنەکانمان لە ٠ و ١ یە چونکە لەکاتی چارەسەرکرنی داتای نادیار ناوی کۆڵۆمنەکانمان نەماوە تەنها
ئیندکسەکانی ماوە وە چەند کۆڵۆمن هەبێت ئەم کارە دووەبارە دەکەینەوە
'''

from sklearn.preprocessing import LabelEncoder 

label=LabelEncoder()
Cancer['A']=label.fit_transform(Cancer[0])
Cancer['out']=label.fit_transform(Cancer[1])
Cancer['C']=label.fit_transform(Cancer[2])
Cancer['D']=label.fit_transform(Cancer[3])
Cancer['E']=label.fit_transform(Cancer[4])
Cancer['F']=label.fit_transform(Cancer[5])
Cancer['G']=label.fit_transform(Cancer[6])
Cancer['H']=label.fit_transform(Cancer[7])
Cancer['I']=label.fit_transform(Cancer[8])
Cancer['J']=label.fit_transform(Cancer[9])
Cancer['K']=label.fit_transform(Cancer[10])
Cancer['L']=label.fit_transform(Cancer[11])
Cancer['M']=label.fit_transform(Cancer[12])
Cancer['N']=label.fit_transform(Cancer[13])
Cancer['o']=label.fit_transform(Cancer[14])
Cancer['p']=label.fit_transform(Cancer[15])
Cancer['q']=label.fit_transform(Cancer[16])
Cancer['r']=label.fit_transform(Cancer[17])
Cancer['s']=label.fit_transform(Cancer[18])
Cancer['t']=label.fit_transform(Cancer[19])
Cancer['v']=label.fit_transform(Cancer[20])
Cancer['w']=label.fit_transform(Cancer[21])
Cancer['x']=label.fit_transform(Cancer[22])
Cancer['y']=label.fit_transform(Cancer[23])
Cancer['z']=label.fit_transform(Cancer[24])
Cancer['a1']=label.fit_transform(Cancer[25])
Cancer['b1']=label.fit_transform(Cancer[26])
Cancer['c1']=label.fit_transform(Cancer[27])
Cancer['d1']=label.fit_transform(Cancer[28])
Cancer['e1']=label.fit_transform(Cancer[29])
Cancer['f1']=label.fit_transform(Cancer[30])
Cancer['g1']=label.fit_transform(Cancer[30])
Cancer['h1']=label.fit_transform(Cancer[31])
'''
دێین لەڕێگەی درۆپ کۆڵۆمنە کۆنەکانی خۆی لادەبەین و تەنها تازەکان ئە‌هێلینەوە واتە 
ئەوانەی کە کردووماننن بە ژمارەی 
'''
Cancer=Cancer.drop([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],axis='columns')

#========================normalization======================================================
'''
 داتاکانمان نۆرمەلایز دەکین بۆ بەینی سفرو یەک وە بۆ ئەم کردارە دوو ڕێکەمان هەیە کە 
 یەکەمیان ئەوەیە ئێمە سودمان لی وەرگرتووە   
 
'''
for i in Cancer:
    Cancer[i] = Cancer[i]/max(Cancer[i])
    
#=========================train and test split=================================
'''
دابەشکردنی داتا بۆ فێرکردن و تاقی کردنەوە
سەرەتا داتاکانمان دەکەینە هەردووگۆڕاوی a و b و دواتر بەشی input and output جیا دەکەینەوە 
بەجۆرێک لە a بەشی input وە ئەوانەی دەبنە output لە ئەی دەری دەکەین و ئەمان دەخەینە ناو x
بۆ b دەکەینە بەشی output و کە یەک کۆڵۆمنەو وە بەشی input لێدەردەکەین 
دواتر لە سەدا ٧٠ دەکەینە بەشی فێرکرن و لە سەدا ٣٠ دەکەینە بەشی تاقی کردنەوە

'''        
a=Cancer
b=Cancer
from sklearn.model_selection import train_test_split
x=a.drop(['out'],axis='columns')
y=b.drop(['A','C','D','E','F','G','H','I','J','K','L','M','N','o','p','q','r','s','t','v','w','x','y','z','a1','b1','c1','d1','e1','f1','g1','h1'],axis='columns')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

 
#=====================creat model classfication================
'''
سەرەتا مۆدێلێک دروست دەکەین لەو جۆرەی کە پێوستمانەو داتاکانما لەو جۆرەیە کە ئێمە دووجۆرمان بەکار هێناوە
classfication and Regression 
وە ئەگەرclassfication بوو ئەوا لەڕێگەی entropy وە ڕووتەکەی دیاری بکات 
دواتر مۆدێلەکەمان فێر دەکەین لە بەشی فێرکردنی بەم کۆدە 
model = model.fit(X_train, y_train)
دواتر دێین مۆدێلەکەمان تێست دەکەین بەم کۆدە بزانین جەندی جواب داوەتوە وەک خۆی
result = model.predict(X_test)
'''
from sklearn import tree 
model = tree.DecisionTreeClassifier(criterion="entropy")
model = model.fit(X_train, y_train)
result = model.predict(X_test)

#accuracy
'''
پێوانە کردنی ئەو خەمڵاندنەی کە مۆدێلەکە بۆی کردوین لەڕێگەی accuracy وە سەیری دەکەین
کە ئەو ئەنجامەی بەدەستمان هاتوو لەگەڵ وەڵامەکانی خۆی بەراورد دەکەین بزانیین جەنی ڕاستە
Accuracy=TP+TN/TP+TN+FP+FN
Precision=TP/TP+FP
Recall=TP/TP+FN

سەرەتا کتێبخانەی هەر یەکێکیان بانگ دەکەین دواتر ئەو نرخەی بەدەستمان هاوردووە 
لەگەڵ ئەوی کە هەمانە سەیری دەکەین بزانین چەندێکی تەواوە 
وە هەتا ڕێژەکە زیاتر بێت ئەوە داتاکانمان باشترن 
'''
#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, result)
print("Accuracy is : {}".format(accuracy))

#precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, result)
print("Precision is : {}".format(precision))

#recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, result)
print("Recall is : {}".format(recall))

#auc 
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, result)
print("AUC is : {}".format(auc))
#============cross validation ===================================
'''
ئێمە لە کرۆس ڤالێدەیشنا دێین بەشی فێڕکاریمان دابەش دەکەین بۆ چەند بەشێک و
هەرچارەی بەشێک لەو بەشانە دەکەین بە بەشی تاقی کرندوە  وە بەشەکانی تر دەکەین بە بەشی فێرکاری 
بەمەش ئەنجامی وردترو دەقیق ترمان دەست دەکەوێت 
کە ئێستا هاتووین بۆ فۆلد ٣ و فۆڵد ٥ وەرمان گرتووە 
'''
 
from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(model, x, y, cv=3)
accuracy = accuracy_score(y_pred, y)
print("Accuracy validation 3 fold : {}".format(accuracy))

precision = precision_score(y_pred, y)
print("Precision validation 3 fold : {}".format(precision))

recall = recall_score(y_pred, y)
print("Recall validation 3 flod: {}".format(recall))

auc = roc_auc_score(y_pred, y)
print("AUC validation 3 fold: {}".format(auc))
#----------------------------
y_pred = cross_val_predict(model, Cancer, y, cv=5)
accuracy = accuracy_score(y_pred, y)
print("Accuracy validation 5 fold : {}".format(accuracy))

precision = precision_score(y_pred, y)
print("Precision validation 5 fold : {}".format(precision))

recall = recall_score(y_pred, y)
print("Recall validation 5 flod: {}".format(recall))

auc = roc_auc_score(y_pred, y)
print("AUC validation 5 fold: {}".format(auc))
#--------------------------------------------------------------
#7
'''
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
'''
 