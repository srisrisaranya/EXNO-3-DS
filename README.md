## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
# FEATURE ENCODING
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/7985936b-520d-449b-91fb-22f9769f6bcd)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/92ad8d9f-1f8c-4222-94ef-4be3d6da6fb4)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/9911a769-34e5-4dd1-9093-e540557299cd)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/9e177426-afdd-4b1a-b465-81e4f16baffb)
```
from sklearn.preprocessing import OneHotEncoder
Ohe = OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(Ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/51c9ab82-5a89-436f-817d-26337353a546)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/8b6d1a72-adce-4149-a40a-8a42f3539e44)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/1d30a6aa-bfc4-40fb-be69-cdca5e9ce7aa)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/888afc2b-f094-41d5-be05-4ed5d558ed32)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb1
```
![image](https://github.com/user-attachments/assets/1cc2ca22-2fab-40c4-8cd3-5818023bb779)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/f6992904-a78f-4951-9413-045a62c69c55)

# FEATURE TRANSFORMATION

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/ec658895-b12d-4e55-9c94-d74c573b992e)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/9204ab9e-66d1-474b-bdca-fb55a9e9737a)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/e1f76bdf-463e-421a-9ba6-6d72a54dc8dd)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/7b622e62-8bb1-412c-aaf1-2041bdd4e140)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/2a37ff1e-0c59-4690-badc-54f72289bbe0)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/69749f8b-2339-4fac-8de7-10aa5751fd92)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/de8b6de1-4227-47aa-87e6-5c39d9f2a62a)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/961432a9-eb04-45c4-9d2a-a81f2f3d0bf0)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/4e927a4b-afc4-41f4-9b4b-8705ea2c57a1)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution="normal")
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/3f6d3cbf-5a05-4e5d-9313-5cbea116acfa)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/4b1cb148-399e-41ed-a363-32ba9fd186a4)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/983a9ae6-d958-4535-8765-a78a8cdf4afe)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution="normal",n_quantiles=891)
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/b3f3d73e-6968-4832-9493-826b6a24dfd9)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/3b61d716-e8b2-4785-a060-c3c994d1847f)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/e2edb0a0-822f-455c-b3b1-5f70870ad7ee)
```
dt=pd.read_csv("/content/titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution="normal",n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt["Age"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/3c1dc84f-e748-4169-8e7e-0183656500a1)
```
sm.qqplot(dt["Age_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/9a6b0430-20cf-4c52-ad23-c24a833e8ec3)



# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
