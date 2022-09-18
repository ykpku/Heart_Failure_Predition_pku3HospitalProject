import numpy as np
import pandas as pd
from sklearn import preprocessing


def feature_engineering(file_path, label_space=None):
    if label_space is None:
        label_space = ['此次发病至离院过程中是否合并AKI', '本次就诊后7天是否存活', '本次就诊后30天是否存活', '本次就诊后30天内是否再入急诊']
    data = pd.read_csv(file_path, header=0)
    print(data[:5])

    data['年龄'].fillna(data['年龄'].mean(), inplace=True)
    data['身高'].fillna(data['身高'].mean(), inplace=True)
    data['体重'].fillna(data['体重'].mean(), inplace=True)
    data['BMI'].fillna(data['BMI'].mean(), inplace=True)

    col = '高血压病,高血脂症,糖尿病,冠心病,先心病,风湿性心脏病,退行性心脏瓣膜病,起搏器植入,心肌病,心脏外科手术,短暂性脑缺血发作,脑梗塞,脑出血,慢性阻塞性肺病,肺源性心脏病,肺栓塞,其他呼吸系统疾病,贫血,风湿性免疫性疾病,甲状腺功能亢进,甲状腺功能减退,慢性肾功能不全,血液透析,腹膜透析,其他疾病'
    col = col.split(',')
    for item in col:
        data[item].fillna(2, inplace=True)

    data['3.1类型'].fillna(3, inplace=True)
    data['3.2类型'].fillna(4, inplace=True)
    data['急性心衰症状出现至就诊时间日'].fillna(data['急性心衰症状出现至就诊时间日'].mean(), inplace=True)

    data['来诊多长时间明确病因'].fillna(6, inplace=True)
    data['本次AHF是否有明确诱因'].fillna(2, inplace=True)
    data['来诊多长时间明确诱因'].fillna(6, inplace=True)
    col1 = '急性冠脉综合征,急性血压升高,缓慢型心律失常,快速型心律失常,急性机械原因,诱因肺栓塞,感染,肾功能减退,不依从盐水摄入或药物的医嘱,有毒物质,诱因心肌病,慢性阻塞性肺病加重,手术和围手术期并发症,交感神经活性增强,代谢激素紊乱,脑血管损害,药物,其他'
    col1 = col1.split(',')
    for item in col1:
        data[item].fillna(2, inplace=True)

    col2 = '体温,心率,呼吸,血压收缩压,血压舒张压,平均动脉压,脉氧饱和度'
    col2 = col2.split(',')
    for item in col2:
        data[item].fillna(data[item].mean(), inplace=True)

    col3 = '端坐呼吸,夜间阵发性呼吸困难,肺部湿罗音,下肢/踝部水肿,颈静脉充盈/怒张,肝淤血,肝颈静脉回流征,纳差/腹胀/腹水,头晕,神志模糊,少尿,四肢湿冷,脉压减小'
    col3 = col3.split(',')
    for item in col3:
        data[item].fillna(2, inplace=True)

    data['心衰类型'].fillna(5, inplace=True)
    data['咳嗽'].fillna(2, inplace=True)
    data['呼吸困难likert评分'].fillna(data['呼吸困难likert评分'].mean(), inplace=True)
    data['纽约心功能分级'].fillna(5, inplace=True)
    data['急性心肌梗死分级'].fillna(5, inplace=True)

    data['经过院前急救来诊'].fillna(2, inplace=True)
    data['PH'].fillna(7.40, inplace=True)
    data['PaCO2'].fillna(40, inplace=True)
    data['Pa02'].fillna(90, inplace=True)
    data['SaO2'].fillna(95, inplace=True)
    data['Lac'].fillna(1, inplace=True)
    data['BE'].fillna(0.0, inplace=True)
    data['HCO3'].fillna(24, inplace=True)
    data['血气标本是否在吸氧状态下获取'].fillna(2, inplace=True)
    data['鼻导管'].fillna(2, inplace=True)
    data['文丘里面罩'].fillna(2, inplace=True)
    data['储氧面罩'].fillna(2, inplace=True)
    data['WBC'].fillna(7, inplace=True)
    data['Neut'].fillna(57.5, inplace=True)
    data['Hb'].fillna(132.5, inplace=True)
    data['HCT'].fillna(40.0, inplace=True)
    data['PLT'].fillna(332, inplace=True)
    data['RDW'].fillna(13, inplace=True)
    data['ALB'].fillna(45, inplace=True)
    data['ALT'].fillna(22.5, inplace=True)
    data['AST'].fillna(24, inplace=True)
    data['Tbil'].fillna(13.35, inplace=True)
    data['GGT'].fillna(10, inplace=True)
    data['Cr'].fillna(90, inplace=True)
    data['ALP'].fillna(85, inplace=True)
    data['BUN'].fillna(5, inplace=True)
    data['K'].fillna(4.5, inplace=True)
    data['Na'].fillna(140, inplace=True)
    data['Ca'].fillna(2.35, inplace=True)
    data['Glu'].fillna(4.85, inplace=True)
    data['CK'].fillna(100, inplace=True)
    data['CKMB'].fillna(12, inplace=True)
    data['LDH'].fillna(152, inplace=True)
    data['HbA1c'].fillna(3.05, inplace=True)
    data['cTNI'].fillna(0.0165, inplace=True)
    data['cTNT'].fillna(0.11, inplace=True)
    data['NTproBNP'].fillna(375, inplace=True)
    data['BNP'].fillna(50, inplace=True)
    data['PCT'].fillna(0.5, inplace=True)
    data['FT3'].fillna(3.25, inplace=True)
    data['FT4'].fillna(1.35, inplace=True)
    data['TSH'].fillna(2.665, inplace=True)
    data['DDimer'].fillna(0.15, inplace=True)
    data.loc[data['DDimer'] == '.', 'DDimer'] = 0.15
    data.loc[data['DDimer'] == '#VALUE!', 'DDimer'] = 0.15

    data['检查心率'].fillna(data['检查心率'].mean(), inplace=True)
    col4 = '窦律,病窦,房扑/房颤/房速,室上速,室速,起搏心律,交界区逸搏,室性逸博,心电图其他,正常,左室高电压,低电压,右束支阻滞,左束支阻滞,I度AVB,II度AVB,Ⅲ度AVB,室早,ST段抬高,ST段压低，T波改变'
    col4 = col4.split(',')
    for item in col4:
        data[item].fillna(2, inplace=True)
    data['肺'].fillna(4, inplace=True)
    data['胸腔积液'].fillna(2, inplace=True)
    data['心影大'].fillna(2, inplace=True)
    data['LA前后'].fillna(data['LA前后'].mean(), inplace=True)
    data['LVDD'].fillna(data['LVDD'].mean(), inplace=True)
    data['RA'].fillna(data['RA'].mean(), inplace=True)
    data['RV'].fillna(data['RV'].mean(), inplace=True)
    data['PASP'].fillna(data['PASP'].mean(), inplace=True)
    data['LAP'].fillna(data['LAP'].mean(), inplace=True)
    data['LVEF'].fillna(data['LVEF'].mean(), inplace=True)
    col5 = '二尖瓣狭窄,二尖瓣反流,三尖瓣狭窄,三尖瓣反流,主动脉瓣狭窄,主动脉瓣反流,肺动脉瓣狭窄,肺动脉瓣反流'
    col5 = col5.split(',')
    for item in col5:
        data[item].fillna(5, inplace=True)
    data['心包填塞'].fillna(2, inplace=True)
    data['PICCO'].fillna(2, inplace=True)
    data['肺动脉导管'].fillna(2, inplace=True)
    data['动脉血压监测'].fillna(2, inplace=True)
    data['中心静脉压监测'].fillna(2, inplace=True)

    data['强心剂'].fillna(2, inplace=True)
    data['扩血管药'].fillna(2, inplace=True)
    data['利尿药'].fillna(2, inplace=True)
    data['呋塞米起始剂量'].fillna(data['呋塞米起始剂量'].mean(), inplace=True)
    data['托拉塞米起始剂量'].fillna(data['托拉塞米起始剂量'].mean(), inplace=True)
    data['布美他尼起始剂量'].fillna(data['布美他尼起始剂量'].mean(), inplace=True)
    data['抗心律失常药'].fillna(2, inplace=True)
    data['抗凝药'].fillna(2, inplace=True)
    data['镇静剂'].fillna(2, inplace=True)
    data['溶栓剂'].fillna(2, inplace=True)
    data['造影剂'].fillna(2, inplace=True)
    data['ACEI/ARB'].fillna(2, inplace=True)
    data['甘露醇'].fillna(2, inplace=True)
    data['万古霉素'].fillna(2, inplace=True)
    data['氨基甙类'].fillna(2, inplace=True)
    data['喹诺酮类'].fillna(2, inplace=True)
    data['NSAIDS类'].fillna(2, inplace=True)

    col6 = '电转复,起搏器,食道调博,冠脉介入治疗,肾脏替代治疗,是否有机械通气,氧疗,氧疗方式鼻导管,氧疗方式文丘里面罩,氧疗方式储氧面罩,IABP主动脉内球囊反搏,ECMO体外膜肺氧合'
    col6 = col6.split(',')
    for item in col6:
        data[item].fillna(2, inplace=True)

    data['是否住院治疗'].fillna(2, inplace=True)
    data['收治重症加护病房'].fillna(2, inplace=True)
    data['收治普通病房'].fillna(2, inplace=True)
    data['住院距离来诊时间'].fillna(6, inplace=True)

    col7 = 'BUN即刻,BNP即刻,NT-proBNP即刻,BUN2,BNP2,NT-proBNP2,BUN3,BNP3,NT-proBNP3,BUN4,BNP4,NT-proBNP4,BUN5,BNP5,NT-proBNP5,BUN6,BNP6,NT-proBNP6,BUN7,BNP7,NT-proBNP7'
    col7 = col7.split(',')
    for item in col7:
        data[item].fillna(data[item].mean(), inplace=True)
    col8 = '尿量即刻,尿量2,尿量3,尿量4,尿量5,尿量6,尿量7'
    col8 = col8.split(',')
    for item in col8:
        data[item].fillna(0, inplace=True)

    data['CR指标'].fillna(data['CR指标'].mean(), inplace=True)
    data['离院转归'].fillna(0, inplace=True)
    data['此次发病至离院过程中是否合并AKI'].fillna(1, inplace=True)
    data['心肾综合征的类型'].fillna(6, inplace=True)

    data['心衰AKI'].fillna(0, inplace=True)
    data['本次就诊后7天是否存活'].fillna(0, inplace=True)
    data['本次就诊后30天是否存活'].fillna(0, inplace=True)
    data['本次就诊后30天内是否再入急诊'].fillna(0, inplace=True)

    names = [column for column in data]
    print(len(names))

    categorical_features_str = "性别,受教育程度,高血压病,高血脂症,糖尿病,冠心病,先心病,风湿性心脏病,退行性心脏瓣膜病,起搏器植入,心肌病,心脏外科手术," \
                               "短暂性脑缺血发作,脑梗塞,脑出血,慢性阻塞性肺病,肺源性心脏病,肺栓塞,其他呼吸系统疾病,贫血,风湿性免疫性疾病," \
                               "甲状腺功能亢进,甲状腺功能减退,慢性肾功能不全,血液透析,腹膜透析,其他疾病,3.1类型,3.2类型,来诊多长时间明确病因," \
                               "本次AHF是否有明确诱因,来诊多长时间明确诱因,急性冠脉综合征,急性血压升高,缓慢型心律失常,快速型心律失常," \
                               "急性机械原因,诱因肺栓塞,感染,肾功能减退,不依从盐水摄入或药物的医嘱,有毒物质,诱因心肌病,慢性阻塞性肺病加重," \
                               "手术和围手术期并发症,交感神经活性增强,代谢激素紊乱,脑血管损害,药物,其他,端坐呼吸,夜间阵发性呼吸困难," \
                               "肺部湿罗音,下肢/踝部水肿,颈静脉充盈/怒张,肝淤血,肝颈静脉回流征,纳差/腹胀/腹水,头晕,神志模糊,少尿,四肢湿冷," \
                               "脉压减小,心衰类型,咳嗽,纽约心功能分级,急性心肌梗死分级,经过院前急救来诊,血气标本是否在吸氧状态下获取,鼻导管," \
                               "文丘里面罩,储氧面罩,窦律,病窦,房扑/房颤/房速,室上速,室速,起搏心律,交界区逸搏,室性逸博,心电图其他,正常," \
                               "左室高电压,低电压,右束支阻滞,左束支阻滞,I度AVB,II度AVB,Ⅲ度AVB,室早,ST段抬高,ST段压低，T波改变," \
                               "肺,胸腔积液,心影大,二尖瓣狭窄,二尖瓣反流,三尖瓣狭窄,三尖瓣反流,主动脉瓣狭窄,主动脉瓣反流,肺动脉瓣狭窄," \
                               "肺动脉瓣反流,心包填塞,PICCO,肺动脉导管,动脉血压监测,中心静脉压监测,强心剂,扩血管药,利尿药,抗心律失常药," \
                               "抗凝药,镇静剂,溶栓剂,造影剂,ACEI/ARB,甘露醇,万古霉素,氨基甙类,喹诺酮类,NSAIDS类,电转复,起搏器,食道调博," \
                               "冠脉介入治疗,肾脏替代治疗,氧疗,氧疗方式鼻导管,氧疗方式文丘里面罩,氧疗方式储氧面罩,IABP主动脉内球囊反搏," \
                               "ECMO体外膜肺氧合,是否有机械通气,是否住院治疗,收治重症加护病房,收治普通病房,住院距离来诊时间,尿量即刻,尿量2," \
                               "尿量3,尿量4,尿量5,尿量6,尿量7"
    value_features_str = "年龄,身高,体重,BMI,急性心衰症状出现至就诊时间日,体温,心率,呼吸,血压收缩压,血压舒张压,平均动脉压,脉氧饱和度," \
                         "呼吸困难likert评分,PH,PaCO2,Pa02,SaO2,Lac,BE,HCO3,WBC,Neut,Hb,HCT,PLT,RDW,ALB,ALT,AST,Tbil," \
                         "GGT,Cr,ALP,BUN,K,Na,Ca,Glu,CK,CKMB,LDH,HbA1c,cTNI,cTNT,NTproBNP,BNP,PCT,FT3,FT4,TSH,DDimer," \
                         "检查心率,LA前后,LVDD,RA,RV,PASP,LAP,LVEF,呋塞米起始剂量,托拉塞米起始剂量,布美他尼起始剂量,BUN即刻,BNP即刻," \
                         "NT-proBNP即刻,BUN2,BNP2,NT-proBNP2,BUN3,BNP3,NT-proBNP3,BUN4,BNP4,NT-proBNP4,BUN5,BNP5," \
                         "NT-proBNP5,BUN6,BNP6,NT-proBNP6,BUN7,BNP7,NT-proBNP7,CR指标"
    categorical_features = categorical_features_str.split(',')
    value_features = value_features_str.split(',')
    print(len(categorical_features), len(value_features))
    print(set(names) - set(categorical_features) - set(value_features))
    print(data[categorical_features])
    cat_onehot = preprocessing.OneHotEncoder(drop='if_binary').fit(data[categorical_features])
    sum = 0
    for i in range(len(categorical_features)):
        # print(categorical_features[i])
        # print(cat_onehot.categories_[i])
        if len(cat_onehot.categories_[i]) > 2:
            sum += len(cat_onehot.categories_[i])
        else:
            sum += 1
    n_hots = cat_onehot.transform(data[categorical_features]).toarray()
    print(n_hots.shape, sum)

    n_vals = data[value_features].values
    print(n_vals.shape)

    features = np.concatenate((n_hots, n_vals), axis=1)
    print(features.shape)

    # pd.DataFrame(features).to_csv("out1.csv")

    Y1 = data['离院转归']  # [0. 1. 2. 3. 4.]
    Y2 = data['心肾综合征的类型']  # [1. 2. 3. 4. 5. 6.]
    Y3 = data['心衰AKI']  # [0. 1.]
    Y4 = data['此次发病至离院过程中是否合并AKI']  # [1. 2.]
    Y5 = data['本次就诊后7天是否存活']  # [0. 1.]
    Y6 = data['本次就诊后30天是否存活']  # [0. 1.]
    Y7 = data['本次就诊后30天内是否再入急诊']  # [0. 1.]
    y_dict = {'离院转归': Y1, '心肾综合征的类型': Y2, '心衰AKI': Y3, '此次发病至离院过程中是否合并AKI': Y4,
              '本次就诊后7天是否存活': Y5, '本次就诊后30天是否存活': Y6, '本次就诊后30天内是否再入急诊': Y7}
    Y4.loc[Y4 == 1.0] = 0
    Y4.loc[Y4 == 2.0] = 1
    Y1.loc[Y1 != 1.0] = 0
    Y1.loc[Y1 == 1.0] = 1

    print(Y1)
    print(np.sum(list(Y1)))
    return features, categorical_features, value_features, cat_onehot.categories, [y_dict[i] for i in label_space], \
           data[categorical_features + value_features]


def new_feature_engineering(file_path, label_space=None, split_further=False):
    if label_space is None:
        label_space = ['此次发病至离院过程中是否合并AKI', '本次就诊后7天是否存活', '本次就诊后30天是否存活', '本次就诊后30天内是否再入急诊']

    data = pd.read_csv(file_path, header=0)
    print(data[:5])

    col = '高血压病,高血脂症,糖尿病,冠心病,先心病,风湿性心脏病,退行性心脏瓣膜病,起搏器植入,心肌病,心脏外科手术,短暂性脑缺血发作,脑梗塞,脑出血,' \
          '慢性阻塞性肺病,肺源性心脏病,肺栓塞,其他呼吸系统疾病,贫血,风湿性免疫性疾病,甲状腺功能亢进,甲状腺功能减退,慢性肾功能不全,血液透析,' \
          '腹膜透析,其他疾病,急性冠脉综合征,急性血压升高,缓慢型心律失常,快速型心律失常,急性机械原因,诱因肺栓塞,感染,肾功能减退,' \
          '不依从盐水摄入或药物的医嘱,有毒物质,诱因心肌病,慢性阻塞性肺病加重,手术和围手术期并发症,交感神经活性增强,代谢激素紊乱,脑血管损害,' \
          '药物,其他,本次AHF是否有明确诱因,端坐呼吸,夜间阵发性呼吸困难,肺部湿罗音,下肢/踝部水肿,颈静脉充盈/怒张,肝淤血,肝颈静脉回流征,' \
          '纳差/腹胀/腹水,头晕,神志模糊,少尿,四肢湿冷,脉压减小,咳嗽,经过院前急救来诊,血气标本是否在吸氧状态下获取,鼻导管,文丘里面罩,储氧面罩,' \
          '窦律,病窦,房扑/房颤/房速,室上速,室速,起搏心律,交界区逸搏,室性逸博,心电图其他,正常,左室高电压,低电压,右束支阻滞,左束支阻滞,' \
          'I度AVB,II度AVB,Ⅲ度AVB,室早,ST段抬高,ST段压低，T波改变,胸腔积液,心影大,心包填塞,PICCO,肺动脉导管,动脉血压监测,中心静脉压监测,' \
          '强心剂,扩血管药,利尿药,抗心律失常药,抗凝药,镇静剂,溶栓剂,造影剂,ACEI/ARB,甘露醇,万古霉素,氨基甙类,喹诺酮类,NSAIDS类,电转复,' \
          '起搏器,食道调博,冠脉介入治疗,肾脏替代治疗,是否有机械通气,氧疗,氧疗方式鼻导管,氧疗方式文丘里面罩,氧疗方式储氧面罩,' \
          'IABP主动脉内球囊反搏,ECMO体外膜肺氧合,是否住院治疗,收治重症加护病房,收治普通病房'
    col = col.split(',')
    print("col1 num: ", len(col))
    for item in col:
        data[item].fillna(2, inplace=True)
    col5 = '心衰类型,纽约心功能分级,急性心肌梗死分级'
    col5 = col5.split(',')
    print("col5 num: ", len(col5))
    for item in col5:
        data[item].fillna(5, inplace=True)
    # 补全数值由5更改为1
    col6 = '二尖瓣狭窄,二尖瓣反流,三尖瓣狭窄,三尖瓣反流,主动脉瓣狭窄,主动脉瓣反流,肺动脉瓣狭窄,肺动脉瓣反流'
    col6 = col6.split(',')
    print("col6 num: ", len(col6))
    for item in col6:
        data[item].fillna(1, inplace=True)
    col8 = '尿量即刻,尿量2,尿量3,尿量4,尿量5,尿量6,尿量7'
    col8 = col8.split(',')
    print("col8 num: ", len(col8))
    for item in col8:
        data[item].fillna(0, inplace=True)
    col2 = '年龄,身高,体重,BMI,急性心衰症状出现至就诊时间日,体温,心率,呼吸,血压收缩压,血压舒张压,平均动脉压,脉氧饱和度,呼吸困难likert评分,' \
           '检查心率,LA前后,LVDD,RA,RV,PASP,LAP,LVEF,呋塞米起始剂量,托拉塞米起始剂量,布美他尼起始剂量,BUN即刻,BNP即刻,NT-proBNP即刻,' \
           'BUN2,BNP2,NT-proBNP2,BUN3,BNP3,NT-proBNP3,BUN4,BNP4,NT-proBNP4,BUN5,BNP5,NT-proBNP5,BUN6,BNP6,NT-proBNP6,' \
           'BUN7,BNP7,NT-proBNP7,CR指标'
    col2 = col2.split(',')
    print("col2 num: ", len(col2))
    for item in col2:
        data[item].fillna(data[item].mean(), inplace=True)
    data['3.1类型'].fillna(3, inplace=True)
    data['3.2类型'].fillna(4, inplace=True)
    data['来诊多长时间明确病因'].fillna(6, inplace=True)
    data['来诊多长时间明确诱因'].fillna(6, inplace=True)
    data['PH'].fillna(7.40, inplace=True)
    data['PaCO2'].fillna(40, inplace=True)
    data['Pa02'].fillna(90, inplace=True)
    data['SaO2'].fillna(95, inplace=True)
    data['Lac'].fillna(1, inplace=True)
    data['BE'].fillna(0.0, inplace=True)
    data['HCO3'].fillna(24, inplace=True)
    data['WBC'].fillna(7, inplace=True)
    data['Neut'].fillna(57.5, inplace=True)
    data['Hb'].fillna(132.5, inplace=True)
    data['HCT'].fillna(40.0, inplace=True)
    data['PLT'].fillna(332, inplace=True)
    data['RDW'].fillna(13, inplace=True)
    data['ALB'].fillna(45, inplace=True)
    data['ALT'].fillna(22.5, inplace=True)
    data['AST'].fillna(24, inplace=True)
    data['Tbil'].fillna(13.35, inplace=True)
    data['GGT'].fillna(10, inplace=True)
    data['Cr'].fillna(90, inplace=True)
    data['ALP'].fillna(85, inplace=True)
    data['BUN'].fillna(5, inplace=True)
    data['K'].fillna(4.5, inplace=True)
    data['Na'].fillna(140, inplace=True)
    data['Ca'].fillna(2.35, inplace=True)
    data['Glu'].fillna(4.85, inplace=True)
    data['CK'].fillna(100, inplace=True)
    data['CKMB'].fillna(12, inplace=True)
    data['LDH'].fillna(152, inplace=True)
    data['HbA1c'].fillna(3.05, inplace=True)
    data['cTNI'].fillna(0.0165, inplace=True)
    data['cTNT'].fillna(0.11, inplace=True)
    data['NTproBNP'].fillna(375, inplace=True)
    data['BNP'].fillna(50, inplace=True)
    data['PCT'].fillna(0.5, inplace=True)
    data['FT3'].fillna(3.25, inplace=True)
    data['FT4'].fillna(1.35, inplace=True)
    data['TSH'].fillna(2.665, inplace=True)
    data['DDimer'].fillna(0.15, inplace=True)
    # 补全数值由4更改为1
    data['肺'].fillna(1, inplace=True)
    data['住院距离来诊时间'].fillna(6, inplace=True)
    data.loc[data['DDimer'] == '.', 'DDimer'] = 0.15
    data.loc[data['DDimer'] == '#VALUE!', 'DDimer'] = 0.15
    print("special col num : 46")

    # 结果列补全
    data['离院转归'].fillna(0, inplace=True)
    data['此次发病至离院过程中是否合并AKI'].fillna(1, inplace=True)
    data['心肾综合征的类型'].fillna(6, inplace=True)
    data['心衰AKI'].fillna(0, inplace=True)
    data['本次就诊后7天是否存活'].fillna(0, inplace=True)
    data['本次就诊后30天是否存活'].fillna(0, inplace=True)
    data['本次就诊后30天内是否再入急诊'].fillna(0, inplace=True)
    print("label num : 7")

    names = [column for column in data]
    print("all col num: ", len(names))

    # delete features: "食道调博", "利尿药", "交感神经活性增强"
    categorical_features_str = "性别,受教育程度,高血压病,高血脂症,糖尿病,冠心病,先心病,风湿性心脏病,退行性心脏瓣膜病,起搏器植入,心肌病," \
                               "心脏外科手术,短暂性脑缺血发作,脑梗塞,脑出血,慢性阻塞性肺病,肺源性心脏病,肺栓塞,其他呼吸系统疾病,贫血," \
                               "风湿性免疫性疾病,甲状腺功能亢进,甲状腺功能减退,慢性肾功能不全,血液透析,腹膜透析,其他疾病,3.1类型,3.2类型," \
                               "来诊多长时间明确病因,本次AHF是否有明确诱因,来诊多长时间明确诱因,急性冠脉综合征,急性血压升高,缓慢型心律失常," \
                               "快速型心律失常,急性机械原因,诱因肺栓塞,感染,肾功能减退,不依从盐水摄入或药物的医嘱,有毒物质,诱因心肌病," \
                               "慢性阻塞性肺病加重,手术和围手术期并发症,代谢激素紊乱,脑血管损害,药物,其他,端坐呼吸," \
                               "夜间阵发性呼吸困难,肺部湿罗音,下肢/踝部水肿,颈静脉充盈/怒张,肝淤血,肝颈静脉回流征,纳差/腹胀/腹水,头晕," \
                               "神志模糊,少尿,四肢湿冷,脉压减小,心衰类型,咳嗽,纽约心功能分级,急性心肌梗死分级,经过院前急救来诊," \
                               "血气标本是否在吸氧状态下获取,鼻导管,文丘里面罩,储氧面罩,窦律,病窦,房扑/房颤/房速,室上速,室速,起搏心律," \
                               "交界区逸搏,室性逸博,心电图其他,正常,左室高电压,低电压,右束支阻滞,左束支阻滞,I度AVB,II度AVB,Ⅲ度AVB," \
                               "室早,ST段抬高,ST段压低，T波改变,肺,胸腔积液,心影大,二尖瓣狭窄,二尖瓣反流,三尖瓣狭窄,三尖瓣反流," \
                               "主动脉瓣狭窄,主动脉瓣反流,肺动脉瓣狭窄,肺动脉瓣反流,心包填塞,PICCO,肺动脉导管,动脉血压监测,中心静脉压监测," \
                               "强心剂,扩血管药,抗心律失常药,抗凝药,镇静剂,溶栓剂,造影剂,ACEI/ARB,甘露醇,万古霉素,氨基甙类," \
                               "喹诺酮类,NSAIDS类,电转复,起搏器,冠脉介入治疗,肾脏替代治疗,氧疗,氧疗方式鼻导管,氧疗方式文丘里面罩," \
                               "氧疗方式储氧面罩,IABP主动脉内球囊反搏,ECMO体外膜肺氧合,是否有机械通气,是否住院治疗,收治重症加护病房," \
                               "收治普通病房,住院距离来诊时间,尿量即刻,尿量2,尿量3,尿量4,尿量5,尿量6,尿量7"
    # delete duplicate feature: "BUN即刻", "NT-proBNP即刻", "BNP即刻"
    value_features_str = "年龄,身高,体重,BMI,急性心衰症状出现至就诊时间日,体温,心率,呼吸,血压收缩压,血压舒张压,平均动脉压,脉氧饱和度," \
                         "呼吸困难likert评分,PH,PaCO2,Pa02,SaO2,Lac,BE,HCO3,WBC,Neut,Hb,HCT,PLT,RDW,ALB,ALT,AST,Tbil," \
                         "GGT,Cr,ALP,BUN,K,Na,Ca,Glu,CK,CKMB,LDH,HbA1c,cTNI,cTNT,NTproBNP,BNP,PCT,FT3,FT4,TSH,DDimer," \
                         "检查心率,LA前后,LVDD,RA,RV,PASP,LAP,LVEF,呋塞米起始剂量,托拉塞米起始剂量,布美他尼起始剂量," \
                         "BUN2,BNP2,NT-proBNP2,BUN3,BNP3,NT-proBNP3,BUN4,BNP4,NT-proBNP4,BUN5,BNP5," \
                         "NT-proBNP5,BUN6,BNP6,NT-proBNP6,BUN7,BNP7,NT-proBNP7,CR指标"
    categorical_features = categorical_features_str.split(',')
    value_features = value_features_str.split(',')
    print("categorical feature num: ", len(categorical_features), "value feature num:", len(value_features))

    split_val2cat = {'Neut': {50, 70},
                     'Ca': {2.2, 2.7},
                     'K': {3.5, 5.5},
                     'HCO3': {22, 27},
                     'PaCO2': {35, 45},
                     'BMI': {18.5, 24},
                     'PH': {7.35, 7.45},
                     '呼吸': {16, 20},
                     'FT3': {2.3, 4.2}}


    cat_onehot = preprocessing.OneHotEncoder(drop='if_binary').fit(data[categorical_features])
    n_hots = cat_onehot.transform(data[categorical_features]).toarray()
    print("categorical to n_hot features: ", n_hots.shape)

    n_vals = data[value_features].values
    print("value features: ", n_vals.shape)

    features = np.concatenate((n_hots, n_vals), axis=1)
    print("overall features: ", features.shape)

    # pd.DataFrame(features).to_csv("out1.csv")

    Y1 = data['离院转归']  # [0. 1. 2. 3. 4.]
    Y2 = data['心肾综合征的类型']  # [1. 2. 3. 4. 5. 6.]
    Y3 = data['心衰AKI']  # [0. 1.]
    Y4 = data['此次发病至离院过程中是否合并AKI']  # [1. 2.]
    Y5 = data['本次就诊后7天是否存活']  # [0. 1.]
    Y6 = data['本次就诊后30天是否存活']  # [0. 1.]
    Y7 = data['本次就诊后30天内是否再入急诊']  # [0. 1.]
    res_Y4 = Y4.copy()
    res_Y4.loc[Y4 == 1.0] = 0
    res_Y4.loc[Y4 == 2.0] = 1
    res_Y1 = Y1.copy()
    res_Y1.loc[Y1 != 1.0] = 0
    res_Y1.loc[Y1 == 1.0] = 1
    res_Y7 = Y7.copy()
    res_Y7.loc[Y7 != 1.0] = 0
    res_Y7.loc[Y7 == 1.0] = 1

    y_dict = {'离院转归': res_Y1, '心肾综合征的类型': Y2, '心衰AKI': Y3, '此次发病至离院过程中是否合并AKI': res_Y4,
              '本次就诊后7天是否存活': Y5, '本次就诊后30天是否存活': Y6, '本次就诊后30天内是否再入急诊': res_Y7}

    return features, categorical_features, value_features, cat_onehot, [y_dict[i] for i in label_space], data[categorical_features + value_features]


def __split_val_cat(data, cat_f, val_f, sp_dict):
    data1 = pd.DataFrame(np.array([[1, 2, 3], [2, 4, 6]]))
    print(data1)