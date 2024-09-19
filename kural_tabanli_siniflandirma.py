# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Görev 1: Aşağıdaki Soruları Yanıtlayınız.

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("datasets/persona.csv")
print(df.shape)
print(df.info())
print(df.head())

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?

print(df["SOURCE"].unique())
print(df["SOURCE"].nunique())
print(df["SOURCE"].value_counts())

# Soru 3: Kaç unique PRICE vardır?

print(df["PRICE"].nunique())

# Soru 4: Hangi PRICE'dankaçar tane satış gerçekleşmiş?

print(df["PRICE"].value_counts())

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

print(df["COUNTRY"].value_counts())
print(df.groupby("COUNTRY")["PRICE"].count())

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

print(df.groupby("COUNTRY")["PRICE"].agg("sum"))

# Soru 7: SOURCE türlerine göre satış sayıları nedir?

print(df["SOURCE"].value_counts())
print(df.groupby("SOURCE")["PRICE"].sum())

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

print(df.groupby("COUNTRY")["PRICE"].agg("mean"))

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

print(df.groupby("SOURCE")["PRICE"].agg("mean"))

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

print(df.groupby(by=["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"}))


# Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

print(df.groupby(['COUNTRY', 'SOURCE', 'SEX','AGE']).agg({'PRICE': 'mean'}))


# Görev 3: Çıktıyı PRICE’a göre sıralayınız.

agg_df = df.groupby(['COUNTRY', 'SOURCE', 'SEX','AGE']).agg({'PRICE': 'mean'}).sort_values("PRICE", ascending = False)
print(agg_df.head())


# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.

agg_df = agg_df.reset_index()
print(agg_df.head())
print(agg_df.columns)


# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’eekleyiniz.

age_bins = [0, 18, 23, 30, 40, 70]
age_labels = ['0_18', '19_23', '24_30', '31_40', '41_70']
agg_df["AGE_CATEGORY"] = pd.cut(agg_df["AGE"], bins = age_bins, labels = age_labels)
agg_df.head()


# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.

agg_df["customers_level_based"] = (agg_df["COUNTRY"].astype(str) + "_" + agg_df["SOURCE"].astype(str)+"_" + agg_df["SEX"].astype(str)+"_" + agg_df["AGE_CAT"].astype(str)).str.upper()
agg_df.groupby(["customers_level_based"])["PRICE"].mean()
type(agg_df["customers_level_based"])
print(agg_df.head())


# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4 , labels=["D", "C", "B", "A"])
print(agg_df.head(30))
print(agg_df.groupby("SEGMENT").agg({"PRICE": ["mean","max","sum"]}))


# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini  tahmin ediniz.

# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]
agg_df[agg_df["customers_level_based"] == "TUR_ANDROID_FEMALE_31_40"]
agg_df[agg_df["customers_level_based"] == "TUR_ANDROID_FEMALE_31_40"]["PRICE"].mean()

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
agg_df[agg_df["customers_level_based"] == "FRA_IOS_FEMALE_31_40"]
agg_df[agg_df["customers_level_based"] == "FRA_IOS_FEMALE_31_40"]["PRICE"].mean()