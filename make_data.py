import json
import csv

# 입력 JSON 파일 경로와 출력 CSV 파일 경로 정의
json_file_path = "game_list.json"
csv_file_path = "filtered_games.csv"  # 더 많은 공간이 있는 경로로 변경

# JSON 파일에서 데이터 읽기
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# gm_status에 "개발중" 또는 "서비스종료"가 없는 항목 필터링
filtered_data = [
    {
        "gm_name": item["gm_name"],
        "gm_platform_no": item["gm_platform_no"],
        "gm_genre": item["gm_genre"]
    }
    for item in data["list"]
    if "개발중" not in item["gm_status"] and "서비스종료" not in item["gm_status"]
]

# 필터링된 데이터를 CSV 파일로 쓰기
with open(csv_file_path, mode="w", newline='', encoding='utf-8-sig') as file:
    writer = csv.DictWriter(file, fieldnames=["gm_name", "gm_platform_no", "gm_genre"])
    writer.writeheader()
    writer.writerows(filtered_data)

print(f"Filtered data has been written to {csv_file_path}")
