from flask import Flask, request, jsonify, render_template
import pandas as pd
import chardet

app = Flask(__name__)

# CSV 파일 경로
file_path = 'game_data.csv'

# 파일 인코딩 감지 및 데이터 읽기
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())

# 데이터 읽기
data = pd.read_csv(file_path, encoding=result['encoding'])

# 필요한 컬럼만 추출
data = data[['level2', 'level3']]
data.columns = ['genre', 'game_name']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input'].lower()
    
    # 장르를 포함하는 게임 필터링
    filtered_games = data[data['genre'].str.contains(user_input, case=False, na=False)]
    
    if not filtered_games.empty:
        # 무작위로 하나의 게임을 선택
        selected_game = filtered_games.sample(1)['game_name'].values[0]
    else:
        # 장르가 포함되지 않는 경우 임의의 게임 추천
        selected_game = data.sample(1)['game_name'].values[0]
    
    prediction = f"AI가 추천한 게임: {selected_game}."
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
