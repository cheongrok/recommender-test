import os

# 환경 변수 설정
RUN_NAME = os.environ['EXP_NAME']
EXP_NAME = os.environ["PLACE_NAME"]
MLFLOW_PATH = "/home/jovyan/work/" # MLflow tracking URI 경로

# 하이퍼파라미터 설정
EMBEDDING_DIM = 8 # 임베딩 차원
LEARNING_RATE = 0.01 # 학습률
NUM_EPOCHS = 2 # 에포크 수
BATCH_SIZE = 1024 # 배치 사이즈
LAYERS = [16, 64, 32, 16, 8] # 레이어 구조
TOP_K = 100 # 추천상위 k개
