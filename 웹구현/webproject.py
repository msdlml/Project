from flask import Flask, render_template, request, jsonify
from folium.plugins import MarkerCluster
import folium
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import collections
from collections import Counter
from PIL import Image

app = Flask(__name__)
#워드 클라우드 데이터프레임 불러오기
wordcloud_final=pd.read_csv('wordcloud_final.csv')
wordcloud_child = wordcloud_final[wordcloud_final['AGRDE_FLAG_NM'] == '10대']
wordcloud_teen = wordcloud_final[wordcloud_final['AGRDE_FLAG_NM'] == '10대']
wordcloud_adult = wordcloud_final[wordcloud_final['AGRDE_FLAG_NM'].str.contains('20대|30대|40대|50대')]
wordcloud_elder = wordcloud_final[wordcloud_final['AGRDE_FLAG_NM'].str.contains('60대|70대 이상')]
# 여기에서 sport 데이터프레임을 불러오거나 정의해야 합니다.
public_sport = pd.read_csv('new_public.csv')
sport = pd.read_csv('final_sport.csv')
sport_child = sport[sport['ITEM_CL_NM'].str.contains('13세이하부')]
sport_teen = sport[sport['ITEM_CL_NM'].str.contains('16세이하부') |
                   sport['ITEM_CL_NM'].str.contains('19세이하부')]
sport_adult = sport[sport['ITEM_CL_NM'].str.contains('대학부') |
                   sport['ITEM_CL_NM'].str.match('실업(일반)') |
                   sport['ITEM_CL_NM'].str.contains('20세이상부')]
sport_senior = sport[sport['ITEM_CL_NM'].str.contains('20세이상부') |
                   sport['ITEM_CL_NM'].str.match('실업(일반)') ]

# 레이더 차트 데이터프레임 불러오기
data_child_criteria = pd.read_excel('data_child2.xlsx')
data_child_criteria['등급'] = data_child_criteria['등급'].map({ '3등급':3, '2등급':2, '1등급':1})
data_teen_criteria = pd.read_excel('data_teen2.xlsx')
data_teen_criteria['등급'] = data_teen_criteria['등급'].map({ '3등급':3, '2등급':2, '1등급':1})
data_adult_criteria = pd.read_excel('data_adult.xlsx')
data_adult_criteria['등급'] = data_adult_criteria['등급'].map({ '3등급':3, '2등급':2, '1등급':1})
data_elder_criteria = pd.read_excel('data_elder2.xlsx')
data_elder_criteria['등급'] = data_elder_criteria['등급'].map({ '3등급':3, '2등급':2, '1등급':1})

def child_load_model():
    model_path = 'child_rf_model.pickle'
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

child_model = child_load_model()

def teen_load_model():
    model_path = 'teen_rf_model.pickle'
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

teen_model = teen_load_model()

def adult_load_model():
    model_path = 'adult_rf_model.pickle'
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

adult_model = adult_load_model()

def elder_load_model():
    model_path = 'elder_rf_model.pickle'
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

elder_model = elder_load_model()

mask_image_path = 'wordcloud_icon.png'
#초기화면
@app.route('/')
def index():
    return render_template('healthy.html')
#개인 정보를 입력하면 나이에 따라 리다이렉션
@app.route('/process_form', methods=['POST'])
def process_form():
    global age, gender,name,city
    try:
        name = request.form['name']
        age = int(request.form['age'])
        gender = request.form['gender']
        residence = request.form['residence']
        city = request.form['city']
        if gender == '남':
            gender = 0
        else:
            gender = 1
        
    # 나이에 따라 리디렉션
        if age <= 12:
            return render_template('child_page.html')
        elif 13 <= age <= 19:
            return render_template('teen_page.html')
        elif 20 <= age <= 64:
            return render_template('adult_page.html')
        else:
            return render_template('elder_page.html')
    except Exception as e:
        # 예외가 발생한 경우 로그에 출력하고 적절한 응답을 반환
        print(f"Error processing form: {e}")
        return "Error processing form", 400  # Internal Server Error
#머신러닝
# 유소년 : 나이,성별, 키, 몸무게, 유연성, bmi(자동계산), 셔플런, 제자리 멀리뛰기
# 청소년 : 나이,성별, 키, 몸무게, 유연성, bmi(자동계산), 셔플런, 제자리 멀리뛰기
# 성인 : 나이,성별, 키, 몸무게, 유연성, bmi(자동계산),교차 윗몸 일으키기, 셔플런, 10m4회 왕복달리기, 제자리 멀리뛰기
# 노인 : 나이, 성별, 키, 몸무게, 유연성, bmi,의자에 앉았다 일어나기, 2분제자리걷기, 8자보행
@app.route('/predict_child',methods = ['POST'])
def predict_child():
    #global standingBroadJump, situp,shuttleRun,sit_and_reach
    try:
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        sit_and_reach = int(request.form['sitAndReach'])
        shuttleRun = int(request.form['shuttleRun'])
        standingBroadJump = int(request.form['standingBroadJump'])
        situp = int(request.form['situp'])

        input_data = [[ age,gender,height, weight, sit_and_reach, 
                      weight / (height*height/10000),shuttleRun,standingBroadJump]]
        
        prediction = child_model.predict(input_data)
        
        radar_img_path=child_health_estimate(age,gender,shuttleRun,standingBroadJump,standingBroadJump,sit_and_reach,situp)
        wordcloud_path = generate_wordcloud_with_mask(wordcloud_child,prediction)
        return render_template('result(승주님).html',name = name,prediction = prediction,radar_img_path=radar_img_path,wordcloud_path=wordcloud_path)
    except Exception as e:
        # 예외 처리
        print(f"Error predicting rank: {e}")
        return "Error predicting rank", 400  # Bad Request

@app.route('/predict_teen',methods = ['POST'])
def predict_teen():
    #global shuttleRun,standingBroadJump,ilinoi,situp,eyehand,sit_and_reach
    try:
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        sit_and_reach = int(request.form['sitAndReach'])
        shuttleRun = int(request.form['shuttleRun'])
        standingBroadJump = int(request.form['standingBroadJump'])
        situp = int(request.form['situp'])
        ilinoi = int(request.form['ilinoi'])
        eyehand = int(request.form['eyehand'])
        input_data = [[ age,gender,height, weight, sit_and_reach, 
                      weight / (height*height/10000),shuttleRun,standingBroadJump]]
        
        prediction = teen_model.predict(input_data)

        radar_img_path = teen_health_estimate(age,gender,shuttleRun,standingBroadJump,sit_and_reach,situp,ilinoi,eyehand)
        wordcloud_path = generate_wordcloud_with_mask(wordcloud_teen,prediction)
        return render_template('result(승주님).html',name = name,prediction = prediction,radar_img_path=radar_img_path,wordcloud_path=wordcloud_path)
    except Exception as e:
        # 예외 처리
        print(f"Error predicting rank: {e}")
        return "Error predicting rank", 400  # Bad Request
    
@app.route('/predict_adult',methods = ['POST'])
def predict_adult():
    try:
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        sit_and_reach = int(request.form['sitAndReach'])
        cross = int(request.form['cross'])
        shuttleRun = int(request.form['shuttleRun'])
        _10run = int(request.form['10run'])
        standingBroadJump = int(request.form['standingBroadJump'])
        
        input_data = [[ age,gender,height, weight, sit_and_reach,weight / (height*height/10000)
                       , cross, shuttleRun,_10run, standingBroadJump]]
        
        prediction = adult_model.predict(input_data)

        radar_img_path = adult_health_estimate(age,gender, shuttleRun,standingBroadJump,cross,_10run,sit_and_reach,standingBroadJump)
        wordcloud_path = generate_wordcloud_with_mask(wordcloud_adult,prediction)
        return render_template('result(승주님).html',name = name,prediction = prediction, radar_img_path = radar_img_path,wordcloud_path=wordcloud_path)
    except Exception as e:
        # 예외 처리
        print(f"Error predicting rank: {e}")
        return "Error predicting rank", 400  # Bad Request

@app.route('/predict_elder',methods = ['POST'])
def predict_elder():
    try:
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        sit_and_reach = int(request.form['sitAndReach'])
        sit_and_up = int(request.form['sitAndup'])
        walking = int(request.form['walking'])
        _8step = int(request.form['8step'])
        target3m = int(request.form['target3m'])
        #노인 : 나이, 성별, 키, 몸무게, 유연성, bmi,의자에 앉았다 일어나기, 2분제자리걷기, 8자보행
        input_data = [[age,gender,height, weight, sit_and_reach, weight / (height*height/10000), sit_and_up, walking, _8step,None]]
        
        prediction = elder_model.predict(input_data)

        radar_img_path = elder_health_estimate(age,gender,walking,_8step,sit_and_up,_8step,sit_and_reach,target3m)
        wordcloud_path = generate_wordcloud_with_mask(wordcloud_elder,prediction)
        return render_template('result(승주님).html',name = name, prediction = prediction, radar_img_path = radar_img_path,wordcloud_path=wordcloud_path)
    except Exception as e:
        # 예외 처리
        print(f"Error predicting rank: {e}")
        return "Error predicting rank", 400  # Bad Request
@app.route('/explain',methods = ['POST'])
def explain():
    return render_template('measurement.html')
@app.route('/sport_search',methods=['POST'])
def sport_search():
    return render_template('sport_search.html')
#지도에서 스포츠클럽확인
@app.route('/map', methods=['POST'])
def map():
    sport_name = request.form['sport_name']
    sport_age = int(request.form['sport_age'])
    location = request.form['location']

    # age에 따른 데이터프레임 선택 로직
    if sport_age <= 13:
        df = sport_child
    elif (sport_age < 20) & (sport_age > 13):
        df = sport_teen
    elif (sport_age >= 20) & (sport_age < 65):
        df = sport_adult
    else:
        df = sport_senior
   
    # 여기에 마커 생성 및 클러스터 추가 로직 추가
    kor_map = folium.Map(location=[36, 128], zoom_start=6.5)
    mc = MarkerCluster().add_to(kor_map)

    for i in df[(df['ITEM_NM'].str.match(sport_name)) & (df['SIGNGU_NM'].str.match(location))].index:
        club_name = df["CLUB_NM"][i].replace('\n', '<br>') #줄바꿈 변경
        popup_content = f'<div style="display: flex; flex-direction: row;"><p><b>상호: {club_name}</b></p></div>'
        marker1 = folium.Marker([df['Latitude'][i],df['Longitude'][i]], popup = folium.Popup(popup_content, max_width=200, min_width=200)
                                ,icon = folium.Icon(color = 'red',icon='star'))
        marker1.add_to(mc)

    folium_map = kor_map._repr_html_()  # templates 폴더에 map.html 파일 저장
    return render_template('map.html',folium_map=folium_map)

@app.route('/public_map', methods=['POST'])
def public_map():
    
    kor_map2 = folium.Map(location=[36, 128], zoom_start=6.5)
    mc1 = MarkerCluster().add_to(kor_map2)

    for i in public_sport[public_sport['ROAD_NM_SIGNGU_NM'].str.match(city)].index:
        club_name = public_sport["FCLTY_NM"][i].replace('\n', '<br>') #줄바꿈 변경
        popup_content = f'<div style="display: flex; flex-direction: row;"><p><b>상호: {club_name}</b></p></div>'
        marker2 = folium.Marker([public_sport['FCLTY_LA'][i],public_sport['FCLTY_LO'][i]], popup = folium.Popup(popup_content, max_width=200, min_width=200)
                                ,tooltip=public_sport['INDUTY_NM'][i],icon = folium.Icon(color = 'blue',icon='cloud'))
        marker2.add_to(mc1)
    folium_map2 = kor_map2._repr_html_()  # templates 폴더에 map.html 파일 저장
    return render_template('public_map.html',folium_map=folium_map2)


def child_health_estimate(age, gender, value1, value2, value3, value4, *value5):
    # 기준표 성별 범위,나이 범위 좁히기
        positive_rating = ('MESURE_IEM_009_VALUE','MESURE_IEM_012_VALUE','MESURE_IEM_019_VALUE','MESURE_IEM_020_VALUE','MESURE_IEM_022_VALUE','MESURE_IEM_022_VALUE.1','MESURE_IEM_023_VALUE','MESURE_IEM_025_VALUE')
        negative_rating = ('MESURE_IEM_013_VALUE','MESURE_IEM_015_VALUE','MESURE_IEM_021_VALUE','MESURE_IEM_026_VALUE','MESURE_IEM_027_VALUE','MESURE_IEM_027_VALUE.1')

        if gender == 0:
            gender = '남'
        else:
            gender = '여'
        data_list = []
        if age<=12:
            data_child_gender = data_child_criteria[data_child_criteria['성별']==gender]
            data_age = data_child_gender[data_child_gender['나이']==age]
       
        columns = data_age.columns.tolist()
        if value5:
            data_list = [value1, value2, value3, value4]
            for value in value5:
                data_list.append(value)
        else:
            data_list = [value1, value2, value3, value4]
    
    #행 하나씩 내려가게
        grade_li = []
    
        for i in range(0,len(data_list)):  #열에 적용
            grade_cnt = 0
            grade=1
            for index, row in data_age.iterrows(): # 행에 적용
                if columns[i+3] in positive_rating:
                    if row[columns[i+3]] <= data_list[i]:
                        if grade <= 5-int(row['등급']):
                            grade = 5-int(row['등급'])
                            grade_li.append(grade)
                        break
                    else:
                        grade_cnt = grade_cnt +1
                    if grade_cnt==3:
                        grade = 1
                        grade_li.append(grade)
                    
                elif columns[i+3] in negative_rating:
                    if row[columns[i+3]] >= data_list[i]:
                        if grade <= 5-int(row['등급']):
                            grade=5-int(row['등급'])
                            grade_li.append(grade)
                        break
                    else:
                        grade_cnt = grade_cnt +1
                    if grade_cnt ==3:
                        grade = 1
                        grade_li.append(grade)
        return making_childradar_chart(grade_li)

def making_childradar_chart(grade_li):
    # 데이터 준비
        values=[]
        num_vars = 0
        categories = []
        if len(grade_li) ==4:
            categories = ['심폐지구력', '순발력', '평형성', '유연성']
            num_vars = len(categories)
            for i in range(len(grade_li)):
                values.append(grade_li[i])


        elif len(grade_li) ==5:
            categories = ['심폐지구력', '순발력',  '평형성', '유연성','근지구력']
            num_vars = len(categories)
            for i in range(len(grade_li)):
                values.append(grade_li[i])
        # 각 카테고리의 각도 계산
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        # 첫 번째 요소를 뒤로 복사하여 폐곡선을 만듭니다.
        values += values[:1]
        angles += angles[:1]

        # 레이더 차트 그리기
        plt.rc('font', family='Malgun Gothic')
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='green', alpha=0.25)

        # 각 카테고리에 레이블 추가
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
    
        # y축 눈금을 1 단위로 설정
        ax.set_yticks(np.arange(1, 5, 1))
        ax.set_yticklabels(['노력필요', '낮음', '보통','좋음'])

        # 차트 제목
        plt.rc('font', family='Malgun Gothic')
        plt.title('체력 항목 레이더 차트',pad=20)

        image_path = "static/radar_chart.png"
        plt.savefig(image_path, format='png')
        plt.close()
        image = 'radar_chart.png'
        return image

def teen_health_estimate(age,gender,value1, value2, value3, *value4):
    # 기준표 성별 범위,나이 범위 좁히기
    positive_rating = ('MESURE_IEM_009_VALUE','MESURE_IEM_012_VALUE','MESURE_IEM_019_VALUE','MESURE_IEM_020_VALUE','MESURE_IEM_022_VALUE','MESURE_IEM_022_VALUE.1','MESURE_IEM_023_VALUE','MESURE_IEM_025_VALUE')
    negative_rating = ('MESURE_IEM_013_VALUE','MESURE_IEM_015_VALUE','MESURE_IEM_021_VALUE','MESURE_IEM_026_VALUE','MESURE_IEM_027_VALUE','MESURE_IEM_027_VALUE.1')
    if gender == 0:
        gender = '남'
    else:
        gender = '여'
    data_list=[]
    if age>=13:
        data_teen_gender = data_teen_criteria[data_teen_criteria['성별']==gender]
        data_age = data_teen_gender[data_teen_gender['나이']==age]
    columns = data_age.columns.tolist()
    if value4:
        data_list = [value1, value2, value3]
        for arg in value4:
            data_list.append(arg)
    
    else:
        data_list = [value1, value2, value3]
    
    #행 하나씩 내려가게
    grade_li = []
    for i in range(0,len(data_list)):  # 열 도는거
        grade_cnt = 0
        grade=1
        for index, row in data_age.iterrows(): # 행 도는거
            if columns[i+3] in positive_rating:
                if data_list[i]==0:
                    grade = 0
                    grade_li.append(grade)
                    
                    break
                else:
                    if row[columns[i+3]] <= data_list[i]:
                        if grade <= 5-int(row['등급']):
                            grade = 5-int(row['등급'])
                            
                            grade_li.append(grade)
                        break
                    else:
                        grade_cnt = grade_cnt +1
                    if grade_cnt==3:
                        grade = 1
                        
                        grade_li.append(grade)
                
            elif columns[i+3] in negative_rating:
                if data_list[i]==0:
                    grade = 0
                    grade_li.append(grade)
                    break
                else:
                    if row[columns[i+3]] >= data_list[i]:
                        if grade <= 5-int(row['등급']):
                            grade=5-int(row['등급'])
                            #print(grade)
                            grade_li.append(grade)
                        break
                    else:
                        grade_cnt = grade_cnt +1
                    if grade_cnt ==3:
                        grade = 1
                        grade_li.append(grade)
    
    return making_teenradar_chart(grade_li)

def making_teenradar_chart(grade_li):
    # 데이터 준비
    values=[]
    num_vars = 0
    
    categories = ['심폐지구력', '순발력','유연성', '근지구력', '민첩성', '평형성' ]
    filtered_categories=[]
    if len(grade_li) ==6:
        
        zero_indices = [index for index, value in enumerate(grade_li) if value == 0] #값이 0인 항목의 인덱스
        
        filtered_categories = [value for index, value in enumerate(categories) if index not in zero_indices] #해당 항목 카테고리만 남겨놓음
        
        filtered_value = [value for index, value in enumerate(grade_li) if value != 0] # 0인 값을 제외한 value
        
        num_vars = len(filtered_categories) 
          
        for i in range(len(filtered_categories)):
            values.append(filtered_value[i])
    
    # 각 카테고리의 각도 계산
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # 첫 번째 요소를 뒤로 복사하여 폐곡선을 만듭니다.
    values += values[:1]
    angles += angles[:1]

    # 레이더 차트 그리기
    plt.rc('font', family='Malgun Gothic')
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='green', alpha=0.25)

    # 각 카테고리에 레이블 추가
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(filtered_categories)
    
    # y축 눈금을 1 단위로 설정
    ax.set_yticks(np.arange(1, 5, 1))
    ax.set_yticklabels(['노력필요', '낮음', '보통','좋음'])

    # 차트 제목
    plt.rc('font', family='Malgun Gothic')
    plt.title('체력 항목 레이더 차트',pad=20)
    image_path = "static/radar_chart.png"
    plt.savefig(image_path, format='png')
    plt.close()
    image = 'radar_chart.png'
    return image

def adult_health_estimate(age,gender, value1,value2,value3,value4,value5,value6):
    positive_rating = ('MESURE_IEM_009_VALUE','MESURE_IEM_012_VALUE','MESURE_IEM_019_VALUE','MESURE_IEM_020_VALUE','MESURE_IEM_022_VALUE','MESURE_IEM_022_VALUE.1','MESURE_IEM_023_VALUE','MESURE_IEM_025_VALUE')
    negative_rating = ('MESURE_IEM_013_VALUE','MESURE_IEM_015_VALUE','MESURE_IEM_021_VALUE','MESURE_IEM_026_VALUE','MESURE_IEM_027_VALUE','MESURE_IEM_027_VALUE.1')

    if gender == 0:
        gender = '남'
    else:
        gender = '여'

    data_list = []
    # 기준표 성별,나이 범위 좁히기    
    if age<=64 and age>=19:
        data_adult_gender = data_adult_criteria[data_adult_criteria['성별']==gender]
        ranges = [(i, i + 4) for i in range(25, 65, 5)]
        ranges.append((19, 24))
        for start, end in ranges:
            if start <= age <= end:
                data_age = data_adult_gender[data_adult_gender['나이']==f'{start}~{end}'] 
            
    columns = data_age.columns.tolist()   
    
    data_list = [value1, value2, value3, value4, value5, value6]

    #행 하나씩 내려가게
    grade_li = []
    for i in range(0, len(data_list)):
        grade_cnt = 0
        grade = 1
        for index, row in data_age.iterrows():
            #print(data_age.columns[i])
            #print(row)
            if columns[i+3] in positive_rating:
                if row[data_age.columns.tolist()[i+3]] <= data_list[i]:
                    #print(row[data_age.columns.tolist()[i]], "입력값=>",data_list[i-3])
                    if grade<= 5-int(row['등급']):
                        grade = 5- int(row['등급'])
                        #print(grade,grade_cnt)
                        grade_li.append(grade)
                        #print(grade_li)
                    break                        

                else:
                    grade_cnt = grade_cnt +1
                if grade_cnt == 3:
                    grade = 1
                    grade_li.append(grade)
                    #print(grade_li)

            elif columns[i+3] in negative_rating:
                if row[columns[i+3]] >= data_list[i]:
                    #print(row[data_age.columns.tolist()[i]], "입력값=>",data_list[i-3])
                    if grade<= 5-int(row['등급']):
                        grade = 5-int(row['등급'])
                        grade_li.append(grade)
                        #print(grade_li)
                    break
                else:
                    grade_cnt = grade_cnt +1
                if grade_cnt ==3:
                    grade = 1
                    grade_li.append(grade)
                    #print(grade_li)
    
    return making_adultradar_chart(grade_li)


def making_adultradar_chart(grade_li):
    # 데이터 준비
    values=[]
    num_vars = 0
    categories = []
    if len(grade_li) ==5:
        categories = ['심폐지구력', '순발력', '근지구력', '민첩성', '유연성']
        num_vars = len(categories)
        for i in range(len(grade_li)):
            values.append(grade_li[i])


    elif len(grade_li) ==6:
        categories = ['심폐지구력', '순발력', '근지구력', '민첩성', '유연성','평형성']
        num_vars = len(categories)
        for i in range(len(grade_li)):
            values.append(grade_li[i])
    # 각 카테고리의 각도 계산
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # 첫 번째 요소를 뒤로 복사하여 폐곡선을 만듭니다.
    values += values[:1]
    angles += angles[:1]

    # 레이더 차트 그리기
    plt.rc('font', family='Malgun Gothic')
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='green', alpha=0.25)

    # 각 카테고리에 레이블 추가
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # y축 눈금을 1 단위로 설정
    ax.set_yticks(np.arange(1, 5, 1))
    ax.set_yticklabels(['노력필요', '낮음', '보통','좋음'])

    # 차트 제목
    plt.rc('font', family='Malgun Gothic')
    plt.title('체력 항목 레이더 차트',pad=20)

    image_path = "static/radar_chart.png"
    plt.savefig(image_path, format='png')
    plt.close()
    image = 'radar_chart.png'
    return image  

def elder_health_estimate(age,gender,value1, value2,value3,value4, value5,*value6):
    positive_rating = ('MESURE_IEM_009_VALUE','MESURE_IEM_012_VALUE','MESURE_IEM_019_VALUE','MESURE_IEM_020_VALUE','MESURE_IEM_022_VALUE','MESURE_IEM_022_VALUE.1','MESURE_IEM_023_VALUE','MESURE_IEM_025_VALUE')
    negative_rating = ('MESURE_IEM_013_VALUE','MESURE_IEM_015_VALUE','MESURE_IEM_021_VALUE','MESURE_IEM_026_VALUE','MESURE_IEM_027_VALUE','MESURE_IEM_027_VALUE.1')

    if gender == 0:
        gender = '남'
    else:
        gender = '여'

    data_list = []
    # 기준표 성별,나이 범위 좁히기
    if age>=65:
        data_elder_gender = data_elder_criteria[data_elder_criteria['성별']==gender]
        if age>=85:
            data_age = data_elder_gender[data_elder_gender['나이']=='85~']
        else:
            ranges = [(i, i + 4) for i in range(65, 85, 5)]
            for start, end in ranges:
                #print(start,end)
                if start <= age <= end:
                    #print(start,end)
                    data_age = data_elder_gender[data_elder_gender['나이']==f'{start}~{end}'] 
    
    columns = data_age.columns.tolist()   
    if value6:
        data_list = [value1, value2, value3, value4,value5]
        for value in value6:
            data_list.append(value)
    else:
        data_list = [value1, value2, value3, value4, value5]
    
    
    #행 하나씩 내려가게
    grade_li = []
    for i in range(0, len(data_list)):
        grade_cnt = 0
        grade = 1
        for index, row in data_age.iterrows():
            if columns[i+3] in positive_rating:
                if row[columns[i+3]] <= data_list[i]:
                    if grade<= 5-int(row['등급']):
                        grade = 5- int(row['등급'])
                        grade_li.append(grade)
                    break                        

                else:
                    grade_cnt = grade_cnt +1
                if grade_cnt == 3:
                    grade = 1
                    grade_li.append(grade)
                    #print(grade_li)

            elif columns[i+3] in negative_rating:
                if row[columns[i+3]] >= data_list[i]:
                    if grade<= 5-int(row['등급']):
                        grade = 5-int(row['등급'])
                        grade_li.append(grade)
                        #print(grade_li)
                    break
                else:
                    grade_cnt = grade_cnt +1
                if grade_cnt ==3:
                    grade = 1
                    grade_li.append(grade)
    return making_elderradar_chart(grade_li)

def making_elderradar_chart(grade_li):
    # 데이터 준비
    values=[]
    num_vars = 0
    categories = []

    if len(grade_li) == 5:
        categories = ['심폐지구력', '순발력', '근지구력', '민첩성', '유연성']
        num_vars = len(grade_li)
        for i in range(len(grade_li)):
            values.append(grade_li[i])
    elif len(grade_li) ==6:
        categories = ['심폐지구력', '순발력', '근지구력', '민첩성', '유연성', '평형성']
        num_vars = len(grade_li)
        for i in range(len(grade_li)):
            values.append(grade_li[i])
    # 각 카테고리의 각도 계산
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # 첫 번째 요소를 뒤로 복사하여 폐곡선을 만듭니다.
    values += values[:1]
    angles += angles[:1]

    # 레이더 차트 그리기
    plt.rc('font', family='Malgun Gothic')
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='green', alpha=0.25)

    # 각 카테고리에 레이블 추가
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # y축 눈금을 1 단위로 설정
    ax.set_yticks(np.arange(1, 5, 1))
    ax.set_yticklabels(['노력필요', '낮음', '보통','좋음'])

    # 차트 제목
    plt.rc('font', family='Malgun Gothic')
    plt.title('체력 항목 레이더 차트',pad=20)

    image_path = "static/radar_chart.png"
    plt.savefig(image_path, format='png')
    plt.close()
    image = 'radar_chart.png'
    return image

def generate_wordcloud_with_mask(data, prediction,font_path='malgun', background_color='white', max_font_size=60):
    # 마스크 이미지 불러오기
    icon = Image.open('wordcloud_icon.png')
    mask = Image.new("RGB", icon.size, (255, 255, 255))
    mask.paste(icon, icon)
    mask = np.array(mask)
    
    # 등급별로 반복하여 워드클라우드 생성
    category_value=prediction
    if category_value == prediction:
        # 단어 빈도수 계산
        counts = collections.Counter(data['RECOMEND_MVM_NM'][data['COAW_FLAG_NM'] == category_value[0]])
        tag = counts.most_common(250)  # 가장 빈도수가 많은 순서대로 n개 추출

        # WordCloud 생성
        wc = WordCloud(font_path=font_path, background_color=background_color, max_font_size=max_font_size, width=800, height=600, mask=mask)
        cloud = wc.generate_from_frequencies(dict(tag))

        # 시각화 (dpi 설정 추가)
        plt.figure(figsize=(10, 10), dpi=300)  # 액자사이즈설정
        plt.title(f"{category_value}등급 추천운동", fontsize=25)           
        plt.axis('off')  # 테두리 선 없애기
        plt.imshow(cloud, interpolation="bilinear")
        plt.savefig('static/wordcloud_image.png',format = 'png')
        plt.close()
        wordcloud_final='wordcloud_image.png'
    return wordcloud_final


if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"에러 발생: {e}")
       