## GRIP recommendation labs


그립 앱 내의 섹션 별로 추천 알고리즘을 실험하고 결과를 서빙하는 공간입니다.    
- docker-compose
  - 이미지: recommender-image-test
- mlflow
  - 이미지: mlflow-image
  - 모니터링: gpu1:15555


### clone
```bash
git clone git@github.com:cheongrok/recommender-test.git
```

### mlflow UI
```bash
cd recommender-test
docker compose up -d mlflow
```


### train
recommender-test/recommender/{PLACE_NAME}/{EXP_NAME} 경로에서 train.sh를 실행합니다.   
 실험은 섹션별 실험별로 폴더단위로 관리하고, 실험 폴더 내부에 config.py 파일로 파라미터를 조정할 수 있습니다. PLACE_NAME은 고정된 섹션명으로 사용하고, EXP_NAME의 경우는 모델의 피처나 하이퍼파라미터의 특징으로 자유롭게 정해서 사용합니다.
- EXP_NAME: 실험명 
- PLACE_NAME: 섹션명
```bash
cd recommender-test/recommender/{PLACE_NAME}/{EXP_NAME}
./train.sh -e {EXP_NAME} -p {PLACE_NAME}
```

### serve(진행중)
가장 성능이 좋은 모델로 추천 결과를 생성하여 전달합니다.
```bash
docker compose up -d {serving_container}
```

### 프로세스 모니터링
학습 컨테이너 이름은 training-${PLACE_NAME}-${EXP_NAME} 이고, mlflow의 컨테이너 이름은 mlflow_container 입니다.
```bash
docker compose logs -f {컨테이너이름}
```
---

### 폴더 구조
```bash
recommender-test
├── docker-compose.yml
├── recommender
│   └── home_whole
│         ├── __init__.py
│         └── NCF
│             ├── config.py
│             ├── __init__.py
│             ├── NCF.py
│             └── train.sh
├── mlruns
│   ├── 633984049566377248
│   │   ├── ef5c9b72790a44e8942c70dd936f1adb
│   │   │   ├── artifacts
│   │   │   ├── meta.yaml
│   │   │   ├── metrics
│   │   │   │   ├── accuracy
│   │   │   │   ├── f1_score
│   │   │   │   ├── precision
│   │   │   │   ├── recall
│   │   │   │   ├── test_loss
│   │   │   │   └── train_loss
│   │   │   ├── params
│   │   │   └── tags
│   │   │       ├── mlflow.runName
│   │   │       ├── mlflow.source.name
│   │   │       ├── mlflow.source.type
│   │   │       └── mlflow.user
│   │   └── meta.yaml
│   └── models
├── __init__.py
├── common
├── grip
│   └── common
├── install-dependency.sh
├── poetry.lock
├── pyproject.toml
├── install-package.sh
└── README.md


```
