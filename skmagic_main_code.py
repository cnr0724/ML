#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

path11='C:/Users/Nuree/Desktop/산학/어린이집.xls'
path12='C:/Users/Nuree/Desktop/산학/유치원(공공데이터).csv'
path13='C:/Users/Nuree/Desktop/산학/유치원(유치원알리미).xlsx'
path2='C:/Users/Nuree/Desktop/산학/중복제거_이름우편.csv'

totalKinder1=pd.read_excel(path11, encoding='CP949')
totalKinder2=pd.read_csv(path12, encoding='CP949')
totalKinder3=pd.read_excel(path13, encoding='CP949')
contractKinder=pd.read_csv(path2,encoding = 'CP949')


# In[4]:


totalKinder=totalKinder3


# In[3]:


print(totalKinder2.columns)


# In[5]:


contractKinder_drop=pd.DataFrame(columns=contractKinder.columns)

for i in range(len(contractKinder)):
    if "유치원" in contractKinder.iloc[i,0]:
        contractKinder_drop.loc[len(contractKinder_drop)]=contractKinder.iloc[i]


# In[35]:


contractKinder_drop.to_excel("C:/Users/Nuree/Desktop/산학/유치원.xls", encoding='CP949')


# In[37]:


cK_pre=contractKinder_drop


# In[38]:


for i in range(len(cK_pre)):
    if len(str(cK_pre.iloc[i,1]))==7:
        cK_pre.iloc[i,1]='0'+str(cK_pre.iloc[i,1])[0:4]
    elif len(str(cK_pre.iloc[i,1]))==8:
        cK_pre.iloc[i,1]=str(cK_pre.iloc[i,1])[0:5]


# In[39]:


cK_pre.to_excel("C:/Users/Nuree/Desktop/산학/유치원우편.xls", encoding='CP949')


# In[6]:


cK_pre=pd.read_excel("C:/Users/Nuree/Desktop/산학/유치원우편.xls",encoding='CP949')
cK_pre["주소"]=""


# In[60]:


for i in range(len(cK_pre)):
    address=cK_pre.iloc[i,2]
    address=address.split()
    if address[0] in ['강원도', '광주광역시', '대구광역시', '대전광역시', 
                      '부산광역시', '서울특별시','울산광역시', '인천광역시', '전라남도',
                      '제주특별자치도']:
        address=" ".join(address[0:4])
    elif address[0] in ['세종특별자치시']:
        address=" ".join(address[0:3])
    elif address[0] in ['경기도', '경상남도', '경상북도', '전라북도', '충청남도', '충청북도']:
        if address[1] in ['고양시', '성남시', '수원시', '안산시',
                          '안양시', '용인시', '창원시', '포항시', '전주시', '천안시', '청주시']:
            address=" ".join(address[0:5])
        else:
            address=" ".join(address[0:4])
    
    cK_pre.iloc[i,12]=address


# In[61]:


cK_pre.to_excel("C:/Users/Nuree/Desktop/산학/유치원_p.xls", encoding='CP949')


# In[9]:


cK_pre=pd.read_excel("C:/Users/Nuree/Desktop/산학/유치원_p.xls", encoding="CP949")


# In[10]:


cK_pre.head()


# In[63]:


joinKinder=pd.DataFrame(columns=contractKinder.columns)


# In[69]:


count=0

for i in range(2, len(totalKinder)):
    name=totalKinder.iloc[i,2]
    add=totalKinder.iloc[i,8]
    for j in range(len(cK_pre)):
        if cK_pre.iloc[j,0]==name:
            if cK_pre.iloc[j,12]==add:
                joinKinder[count]=cK_pre.iloc[j]
                count+=1


# In[81]:


joinKinder=pd.read_excel("C:/Users/Nuree/Desktop/산학/1차 매칭.xlsx", encoding='CP949')


# In[83]:


joinKinder=joinKinder.T


# In[84]:


joinKinder.columns=joinKinder.iloc[0]
joinKinder.drop(['Unnamed: 0'])


# In[121]:


count=0
contract_seoul=pd.DataFrame()
for i in range(len(cK_pre)):
    address=cK_pre.iloc[i,2]
    address=address.split()
    if address[0]=='서울특별시':
        contract_seoul[count]=cK_pre.iloc[i]
        count+=1


# In[113]:


count=0
total_seoul=pd.DataFrame()
for i in range(2, len(totalKinder)):
    address=totalKinder.iloc[i,8]
    address=address.split()
    if address[0]=='서울특별시':
        total_seoul[count]=totalKinder.iloc[i]
        count+=1


# In[122]:


contract_seoul=contract_seoul.T
contract_seoul.to_excel("C:/Users/Nuree/Desktop/산학/계약서울.xls", encoding='CP949')


# In[116]:


total_seoul=total_seoul.T


# In[126]:


total_seoul.columns=totalKinder.iloc[1]


# In[127]:


total_seoul.to_excel("C:/Users/Nuree/Desktop/산학/전체서울.xls", encoding='CP949')


# In[4]:


import konlpy
from konlpy.tag import Okt

okt=Okt()


# In[16]:


word2index={}
skMagicAdd="서울특별시 중구 통일로 10. 14층"
skMagicAdd2="서울특별시 중구 통일로 통일로 10. 14층"
skMagicRental="서울특별시 중구 서소문로 88 서소문 타워"
print("1. ",skMagicAdd)
print("2. ",skMagicAdd2)
print("3. ", skMagicRental)
o=okt.morphs(skMagicAdd)
o2=okt.morphs(skMagicRental)
o3=okt.morphs(skMagicAdd2)
for voca in o:
    if voca not in word2index.keys():
        word2index[voca]=len(word2index)
for voca in o2:
    if voca not in word2index.keys():
        word2index[voca]=len(word2index)
for voca in o3:
    if voca not in word2index.keys():
        word2index[voca]=len(word2index)

vector1=[0 for _ in range(len(word2index))]
for voca in o:
    index=word2index.get(voca)
    vector1[index]=vector1[index]+1

vector2=[0 for _ in range(len(word2index))]
for voca in o2:
    index=word2index.get(voca)
    if vector2[index]==0:
        vector2[index]+=1

vector3=[0 for _ in range(len(word2index))]
for voca in o3:
    index=word2index.get(voca)
    if vector3[index]==0:
        vector3[index]+=1

print(skMagicAdd,"->",vector1)
print(skMagicAdd2,"->",vector3)
print(skMagicRental,"->",vector2)
print()

print(word2index)


# In[18]:


print("첫번째 주소와 두번째 주소의 코사인 유사도: ", cos_sim(vector1, vector3))
print("첫번째 주소와 세번째 주소의 코사인 유사도:  ",cos_sim(vector2,vector1))


# In[146]:


word2index={}
total_vec=[]
for i in range(len(total_seoul)):
    o=okt.morphs(total_seoul.iloc[i,8])
    for voca in o:
        if voca not in word2index.keys():
            word2index[voca]=len(word2index)

for i in range(len(contract_seoul)):
    o=okt.morphs(contract_seoul.iloc[i,2])
    for voca in o:
        if voca not in word2index.keys():
            word2index[voca]=len(word2index)
    
for  i in range(len(total_seoul)):
    o=okt.morphs(total_seoul.iloc[i,8])
    bow=[0 for _ in range(len(word2index))]
    for voca in o:
        index=word2index.get(voca)
        bow[index]=bow[index]+1
    total_vec.append(bow)


# In[17]:


from numpy import dot
from numpy.linalg import norm
import numpy as np
def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))


# In[163]:


contract_vec=[]
    
for  i in range(len(contract_seoul)):
    o=okt.morphs(contract_seoul.iloc[i,12])
    bow=[0 for _ in range(len(word2index))]
    for voca in o:
        index=word2index.get(voca)
        bow[index]=bow[index]+1
    contract_vec.append(bow)


# In[164]:


print(len(total_vec), len(contract_vec))


# In[168]:


langLstTot=pd.DataFrame()
langLstCon=pd.DataFrame()

count=0
for i in range(len(total_vec)):
    for j in range(len(contract_vec)):
        result=cos_sim(total_vec[i],contract_vec[j])
        if result>=0.9:
            print(total_seoul.iloc[i, 2],': ',total_seoul.iloc[i,8])
            print(contract_seoul.iloc[j, 0],': ',contract_seoul.iloc[j,12])
            print(result)
            langLstTot[count]=total_seoul.iloc[i]
            langLstCon[count]=contract_seoul.iloc[j]
            count+=1

print(count)


# In[188]:


langLstTot.to_excel("C:/Users/Nuree/Desktop/산학/매칭전체서울.xls", encoding='CP949')


# In[190]:


langLstCon=langLstCon.T


# In[191]:


langLstCon.to_excel("C:/Users/Nuree/Desktop/산학/매칭계약서울.xls", encoding='CP949')


# In[192]:


matchSeoul=pd.read_excel("C:/Users/Nuree/Desktop/산학/매칭전체서울.xls", encoding='CP949')


# In[195]:


matchSeoul.columns


# In[196]:


matchSeoul['구']=""


# In[198]:


for i in range(len(matchSeoul)):
    add=matchSeoul.iloc[i,8]
    matchSeoul.iloc[i,21]=add.split()[1]


# In[200]:


matchSeoul.to_excel("C:/Users/Nuree/Desktop/산학/매칭결과최종.xls", encoding='CP949')


# In[201]:


totalSeoul=pd.read_excel("C:/Users/Nuree/Desktop/산학/전체서울.xls", encoding='CP949')


# In[202]:


totalSeoul['구']=""


# In[203]:


for i in range(len(totalSeoul)):
    add=totalSeoul.iloc[i,8]
    totalSeoul.iloc[i,21]=add.split()[1]


# In[204]:


totalSeoul.to_excel("C:/Users/Nuree/Desktop/산학/전체서울최종.xls", encoding='CP949')


# In[207]:


word2index={}
total_vec=[]
for i in range(2, len(totalKinder)):
    o=okt.morphs(totalKinder.iloc[i,8])
    for voca in o:
        if voca not in word2index.keys():
            word2index[voca]=len(word2index)

for i in range(len(cK_pre)):
    o=okt.morphs(cK_pre.iloc[i,2])
    for voca in o:
        if voca not in word2index.keys():
            word2index[voca]=len(word2index)
    
for  i in range(2, len(totalKinder)):
    o=okt.morphs(totalKinder.iloc[i,8])
    bow=[0 for _ in range(len(word2index))]
    for voca in o:
        index=word2index.get(voca)
        bow[index]=bow[index]+1
    total_vec.append(bow)


# In[208]:


contract_vec=[]
    
for  i in range(len(contract_seoul)):
    o=okt.morphs(contract_seoul.iloc[i,12])
    bow=[0 for _ in range(len(word2index))]
    for voca in o:
        index=word2index.get(voca)
        bow[index]=bow[index]+1
    contract_vec.append(bow)


# In[ ]:


langLstTot=pd.DataFrame()
langLstCon=pd.DataFrame()

count=0
for i in range(len(total_vec)):
    for j in range(len(contract_vec)):
        result=cos_sim(total_vec[i],contract_vec[j])
        if result>=0.7:
            print(total_seoul.iloc[i, 2],': ',total_seoul.iloc[i,8])
            print(contract_seoul.iloc[j, 0],': ',contract_seoul.iloc[j,12])
            print(result)
            langLstTot[count]=total_seoul.iloc[i]
            langLstCon[count]=contract_seoul.iloc[j]
            count+=1

print(count)


# In[35]:


def sepaRegion(contractData, totalData, region):
    count=0
    regionDataContract=pd.DataFrame()
    for i in range(len(contractData)):
        address=contractData.iloc[i,2]
        address=address.split()
        if address[0]==region:
            regionDataContract[count]=contractData.iloc[i]
            count+=1
            
    count=0
    regionDataTotal=pd.DataFrame()
    for i in range(2, len(totalData)):
        address=totalData.iloc[i,8]
        address=address.split()
        if address[0]==region:
            regionDataTotal[count]=totalData.iloc[i]
            count+=1
            
    regionDataContract=regionDataContract.T
    regionDataContract.to_excel("C:/Users/Nuree/Desktop/산학/매칭내역파일/지역별계약"+region+".xlsx", encoding='CP949')
    regionDataTotal=regionDataTotal.T
    regionDataTotal.to_excel("C:/Users/Nuree/Desktop/산학/매칭내역파일/지역별전체"+region+".xlsx", encoding='CP949')
    return regionDataContract, regionDataTotal


# In[12]:


#강원도, 광주광역시, 대구광역시, 대전광역시, 부산광역시,울산광역시
#인천광역시, 전라남도, 제주특별자치도, 세종특별자치시, 경기도, 경상남도, 경상북도,
#전라북도, 충청남도, 충청북도


# In[13]:


regionLst=['강원도', '광주광역시', '대구광역시', '대전광역시', '부산광역시',
          '울산광역시', '인천광역시', '전라남도', '제주특별자치도', '세종특별자치시',
           '경기도', '경상남도', '경상북도', '전라북도', '충청남도', '충청북도']
dataLst=[cK_pre, totalKinder]
dataLstString=['계약', '전체']


# In[31]:


for j in range(len(regionLst)):
    contractData, totalData=sepaRegion(cK_pre, totalKinder, regionLst[j])
    
    word2index={}
    total_vec=[]
    for i in range(len(totalData)):
        o=okt.morphs(totalData.iloc[i,8])
        for voca in o:
            if voca not in word2index.keys():
                word2index[voca]=len(word2index)

    for i in range(len(contractData)):
        o=okt.morphs(contractData.iloc[i,2])
        for voca in o:
            if voca not in word2index.keys():
                word2index[voca]=len(word2index)
    
    for  i in range(len(totalData)):
        o=okt.morphs(totalData.iloc[i,8])
        bow=[0 for _ in range(len(word2index))]
        for voca in o:
            index=word2index.get(voca)
            bow[index]=bow[index]+1
        total_vec.append(bow)

    contract_vec=[]
    for  i in range(len(contractData)):
        o=okt.morphs(contractData.iloc[i,12])
        bow=[0 for _ in range(len(word2index))]
        for voca in o:
            index=word2index.get(voca)
            bow[index]=bow[index]+1
        contract_vec.append(bow)
    
    langLstTot=pd.DataFrame()
    langLstCon=pd.DataFrame()

    count=0
    for i in range(len(total_vec)):
        for k in range(len(contract_vec)):
            result=cos_sim(total_vec[i],contract_vec[k])
            if result>=0.9:
                langLstTot[count]=totalData.iloc[i]
                langLstCon[count]=contractData.iloc[k]
                count+=1
    
    langLstTot=langLstTot.T
    langLstTot.to_excel("C:/Users/Nuree/Desktop/산학/매칭전체"+regionLst[j]+".xlsx", encoding='CP949')
    langLstCon=langLstCon.T
    langLstCon.to_excel("C:/Users/Nuree/Desktop/산학/매칭계약"+regionLst[j]+".xlsx", encoding='CP949')
    print(regionLst[j]+"완료")


# In[36]:


for j in range(len(regionLst)):
    contractData, totalData=sepaRegion(cK_pre, totalKinder, regionLst[j])


# In[ ]:


for j in range(len(contract_seoul)):
    o=okt.morphs(contract_seoul.iloc[j,2])
    
    
    # word2index={}
# bow1=[]
# for voca in o1:
#     if voca not in word2index1.keys():
#         word2index1[voca]=len(word2index1)
#         bow1.insert(len(word2index1)-1,1)
#     else:
#         index=word2index1.get(voca)
#         bow1[index]=bow1[index]+1

# word2index2={}
# bow2=[]
# for voca in o2:
#     if voca not in word2index2.keys():
#         word2index2[voca]=len(word2index2)
#         bow2.insert(len(word2index2)-1,1)
#     else:
#         index=word2index2.get(voca)
#         bow2[index]=bow2[index]+1

# print(word2index1)
# print(word2index2)
# print(bow1)
# print(bow2)

# from numpy import dot

# # for i in range(len(contractKinder)):
# #     o=okt.morphs(contractKinder.iloc[i,6])
# #     contractKinder.iloc[i,17]


# In[6]:


# fstSetTot=list(set(totalKinder['1st 주소']))
# seclstTot=list(set(totalKinder['2nd 주소']))
# nByFstTot=totalKinder.groupby('1st 주소')['1st 주소'].count()
# print(nByFstTot)
# nBySecTot=totalKinder.groupby('2nd 주소')['2nd 주소'].count()
# print(nBySecTot)

