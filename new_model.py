from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import chardet

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

# 프롬프트 구성 함수
def construct_game_prompts(data):
    total_prompts = []
    for idx in range(len(data)):
        genre = data['genre'][idx].lower()
        game_name = data['game_name'][idx]
        prompt = f"### Human: I am looking for a game that fits the genre of {genre}. Can you recommend one? ### Assistant: {game_name}"
        total_prompts.append(prompt)
    return total_prompts

prompts = construct_game_prompts(data)

# 데이터셋을 Huggingface 포맷으로 변환
dataset = Dataset.from_dict({"text": prompts})

# 모델과 토크나이저 초기화
model_name = 'gpt2'  # 원하는 모델 이름으로 변경
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 새로운 pad_token 추가
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 모델의 임베딩 사이즈 확장
model.resize_token_embeddings(len(tokenizer))

# 토큰화 함수
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# 데이터셋 토큰화
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 학습 파라미터 설정
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 트레이너 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # 평가 데이터셋을 별도로 준비하는 것이 좋습니다.
)

# 모델 학습
trainer.train()

# 모델 저장
model.save_pretrained('trained_model')
tokenizer.save_pretrained('trained_model')
