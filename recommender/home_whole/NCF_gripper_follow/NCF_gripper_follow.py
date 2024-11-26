from grip.common.datasource.clickhouse import Clickhouse

from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
import pandas as pd
import numpy as np
import random
import faiss
import pytz
import os

from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.init as init
import torch.optim as optim
import torch.nn as nn
import torch

import mlflow
import mlflow.pytorch

import config


def load_data():
    df = Clickhouse.run_sql("""
        SELECT 
            userSeq
            , contentUserSeq
            , `@timestamp` AS timestamp
        FROM default.elasticsearch e
        WHERE
            e.userType = 2
            AND e.viewTimeMillis > 300000
            AND `@timestamp` >= NOW() - INTERVAL 1 DAY
        ORDER BY userSeq ASC, contentUserSeq ASC
    """)
    return df


def prepDataset(df):
    # 2회 이상 시청한 사용자
    df = df.drop_duplicates(subset=['userSeq', 'contentUserSeq'])
    df_tmp = df[['userSeq', 'contentUserSeq']].groupby(['userSeq']).count()
    df_tmp = df_tmp.rename(columns={'contentUserSeq': 'count'})
    df_tmp = df_tmp[df_tmp['count'] > 1].reset_index(drop=False)
    df = pd.merge(df_tmp, df, on='userSeq', how='left').drop(columns=['count'])

    # 사용자 reindex
    user_id = df[['userSeq']].drop_duplicates().reset_index(drop=True)
    user_id['userId'] = np.arange(len(user_id))
    
    # 원래의 userSeq와 새 userId 간의 매핑 사전 생성
    user_mapping = dict(zip(user_id['userId'], user_id['userSeq']))
    
    # 데이터프레임에 새 userId 추가
    df = pd.merge(df, user_id, on='userSeq', how='left')

    # 그리퍼 reindex
    item_id = df[['contentUserSeq']].drop_duplicates().reset_index(drop=True)
    item_id['itemId'] = np.arange(len(item_id))
    
    # 원래의 contentUserSeq와 새 itemId 간의 매핑 사전 생성
    item_mapping = dict(zip(item_id['itemId'], item_id['contentUserSeq']))
    
    # 데이터프레임에 새 itemId 추가
    df = pd.merge(df, item_id, on='contentUserSeq', how='left')
    
    user_pool = set(df['userId'].unique())
    item_pool = set(df['itemId'].unique())
    
    df['implicit'] = 1

    return df, user_pool, item_pool, user_mapping, item_mapping


def num_count(user_pool, item_pool):
    num_users, num_items = len(user_pool), len(item_pool)
    return num_users, num_items


def sample_negative(df, item_pool, num_neg=30):
    """return all negative items & 100 sampled negative items"""
    interact_status = df.groupby('userId')['itemId'].apply(set).reset_index().rename(
        columns={'itemId': 'interacted_items'})
    interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: item_pool - x)
    interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), num_neg))
    
    return interact_status[['userId', 'negative_items', 'negative_samples']]


def splitDataset(df):
    df['rank_latest'] = df.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
    test = df[df['rank_latest'] == 1]
    train = df[df['rank_latest'] > 1]
    assert set(train['userId']).issubset(set(test['userId']))
    
    return train[['userId', 'itemId', 'implicit']], test[['userId', 'itemId', 'implicit']]


def add_negative_samples(train_df, test_df, negatives):

    # negatives를 DataFrame으로 변환
    negative_samples = []
    
    for idx, row in negatives.iterrows():
        for negative_item in row['negative_samples']:
            negative_samples.append({
                'userId': row['userId'],
                'itemId': negative_item,
                'implicit': 0  # negative 샘플의 implicit은 0
            })
    negative_df = pd.DataFrame(negative_samples)

    # negative_df를 활용해서 test에는 1개 train에는 나머지행을 가져가도록 만듦.
    test_neg_df = negative_df.groupby('userId').apply(lambda x: x.sample(1)).reset_index(drop=True)
    train_neg_df = negative_df[~negative_df.index.isin(test_neg_df.index)]
    
    # train_df와 negative_df 합치기
    combined_train_df = pd.concat([train_df, train_neg_df], ignore_index=True)
    combined_test_df = pd.concat([test_df, test_neg_df], ignore_index=True)
    
    return combined_train_df, combined_test_df


class GripNCFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, layers):
        super(GripNCFModel, self).__init__() # nn.Module로 초기화.
        self.user_embedding = nn.Embedding(num_users+1, embedding_dim)
        self.item_embedding = nn.Embedding(num_items+1, embedding_dim)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.affine_output = nn.Linear(in_features=layers[-1], out_features=1) # 행렬곱을수행
        self.logistic = nn.Sigmoid()
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for sm in self.modules():
            if isinstance(sm, (nn.Embedding, nn.Linear)):
                print(sm)
                nn.init.normal_(sm.weight.data, 0.0, 0.01)
        
    def forward(self, user_ids, item_ids):
        user_embedded = self.user_embedding(user_ids)
        item_embedded = self.item_embedding(item_ids)
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        logits = self.affine_output(vector)
        pred = self.logistic(logits)
        
        return pred
    
    
class GripDataset(Dataset):
    def __init__(self, user_ids, item_ids, labels):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]
    
    
    
def GripDataLoader(train_df, test_df, batch_size, shuffle=True, num_workers=4):
    # 데이터셋 텐서화 및 데이터로더 준비
    train_users = torch.LongTensor(train_df['userId'].values)
    train_items = torch.LongTensor(train_df['itemId'].values)
    train_labels = torch.LongTensor(train_df['implicit'].values)
    test_users = torch.LongTensor(test_df['userId'].values)
    test_items = torch.LongTensor(test_df['itemId'].values)
    test_labels = torch.LongTensor(test_df['implicit'].values)

    # 사용자 지정 데이터셋 인스턴스 생성
    train_dataset = GripDataset(train_users, train_items, train_labels)
    test_dataset = GripDataset(test_users, test_items, test_labels)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_loader, test_loader


def make_time():
    korea_tz = pytz.timezone('Asia/Seoul')
    korea_time = datetime.now(korea_tz)
    korea_time = korea_time.strftime("%Y-%m-%d %H:%M:%S")
    return korea_time


def train_model(train_loader, test_loader):
    # 실험 이름 설정
    # os.chdir(config.MLFLOW_PATH)
    mlflow.set_experiment(config.EXP_NAME)
    mlflow.set_tracking_uri(f"{config.MLFLOW_PATH}/mlruns")
    best_test_loss = float('inf')
    best_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 초기화
    model = GripNCFModel(num_users, num_items, config.EMBEDDING_DIM, config.LAYERS)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCELoss()

    # MLflow 실험 시작
    with mlflow.start_run(run_name=f"{config.RUN_NAME}_{make_time()}"):
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            train_loss = 0
            for user, item, label in train_loader:
                user, item, label = user.to(device), item.to(device), label.float().to(device)

                optimizer.zero_grad()
                output = model(user, item)
                loss = criterion(output.squeeze(), label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            train_loss /= len(train_loader)

            # 평가
            model.eval()
            test_loss = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for user, item, label in test_loader:
                    user, item, label = user.to(device), item.to(device), label.float().to(device)
                    output = model(user, item)
                    loss = criterion(output.squeeze(), label)
                    test_loss += loss.item()
                    preds = (output > 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())

            test_loss /= len(test_loader)

            # 성능 지표 계산
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            accuracy = np.mean(all_preds == all_labels)
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)

            # 메트릭 로그
            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
            mlflow.log_metric("test_loss", test_loss, step=epoch + 1)
            mlflow.log_metric("accuracy", accuracy, step=epoch + 1)
            mlflow.log_metric("precision", precision, step=epoch + 1)
            mlflow.log_metric("recall", recall, step=epoch + 1)
            mlflow.log_metric("f1_score", f1, step=epoch + 1)

            print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}, Accuracy: {accuracy:.4f}')

            # 최상의 모델 저장
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = model
    
    return best_model


def extract_embedding(best_model):
    user_embeddings = best_model.user_embedding.weight.data  # 사용자 임베딩
    item_embeddings = best_model.item_embedding.weight.data  # 아이템 임베딩
    return user_embeddings, item_embeddings


def get_all_recommendations(user_embeddings, item_embeddings, user_mapping, item_mapping):
    
    user_embeddings, item_embeddings = user_embeddings[1:], item_embeddings[1:]
    k=item_embeddings.shape[0]
    # 아이템 임베딩을 NumPy 배열로 변환
    item_embeddings_np = item_embeddings.cpu().detach().numpy()  # GPU에서 CPU로 이동하고 NumPy로 변환

    # FAISS Index 생성 (L2 거리 사용)
    index = faiss.IndexFlatL2(item_embeddings_np.shape[1])  # 아이템 임베딩의 차원 수
    index.add(item_embeddings_np)  # 아이템 임베딩을 인덱스에 추가

    all_recommendations = {}

    # 모든 사용자에 대해 추천 계산
    for user_id in range(user_embeddings.shape[0]): # padding 부분 제거
        # 사용자 임베딩 추출
        user_embedding_np = user_embeddings[user_id].cpu().detach().numpy().reshape(1, -1)  # 특정 사용자 추출

        # k개의 가장 가까운 아이템 찾기
        distances, indices = index.search(user_embedding_np, k)  # k가 가장 유사한 아이템의 개수

        # 추천 결과를 딕셔너리에 저장
        recommendations = []
        for i in range(k):
            recommended_item_index = indices[0][i]  # 가장 유사한 아이템 인덱스
            original_item_id = item_mapping[recommended_item_index]  # 원래 아이템 ID 찾기
            recommendations.append((original_item_id, distances[0][i]))  # (아이템 ID, 유사도) 튜플 추가

        # 유사도에 따라 내림차순 정렬
        recommendations.sort(key=lambda x: x[1])  # 유사도 기준으로 정렬
        recommendations = recommendations[::-1]  # 내림차순으로 되돌리기

        # 사용자 실제 ID를 키로 하고, 추천된 아이템과 유사도를 값으로 사용
        all_recommendations[user_mapping[user_id]] = recommendations

    return all_recommendations


def calculate_hit_rate_ndcg(all_recommendations, test_df, k=10):

    hit_count = 0
    ndcg_scores = []

    for user_id in test_df['userId'].unique():
        # Get ground truth items with positive interactions for the user
        relevant_items = set(test_df[(test_df['userId'] == user_id) & (test_df['implicit'] == 1)]['itemId'])
        recommended_items = [item for item, _ in all_recommendations.get(user_id, [])[:k]]

        hits = any(item in relevant_items for item in recommended_items)
        if hits:
            hit_count += 1

        # Calculate NDCG
        dcg = 0.0
        idcg = 0.0
        for rank, item in enumerate(recommended_items):
            if item in relevant_items:
                dcg += 1 / np.log2(rank + 2)  # DCG component for a hit at this rank

        # Ideal DCG (IDCG) for NDCG normalization, assumes all relevant items are in top positions
        for rank in range(min(len(relevant_items), k)):
            idcg += 1 / np.log2(rank + 2)  # Ideal gain per rank position

        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

    # Calculate overall hit rate and average NDCG
    hit_rate = hit_count / len(test_df['userId'].unique())
    ndcg_score = np.mean(ndcg_scores)

    print(f"hit_rate, ndcg_score : {hit_rate}, {ndcg_score}")
    return hit_rate, ndcg_score


if __name__=="__main__":
    
    df = load_data()
    df, user_pool, item_pool, user_mapping, item_mapping = prepDataset(df) # 데이터셋 전처리
    num_users, num_items = num_count(user_pool, item_pool)
    negatives = sample_negative(df, item_pool) # 네거티브 샘플링
    train_df, test_df = splitDataset(df) # train/test 데이터셋 분리
    train_df, test_df = add_negative_samples(train_df, test_df, negatives) # 네거티브 샘플 병합
    train_loader, test_loader = GripDataLoader(train_df, test_df, config.BATCH_SIZE) # 데이터로더
    best_model = train_model(train_loader, test_loader)
    user_embeddings, item_embeddings = extract_embedding(best_model)
    all_recommendations = get_all_recommendations(user_embeddings, item_embeddings, user_mapping, item_mapping)
    hit_rate, ndcg_score = calculate_hit_rate_ndcg(all_recommendations, test_df, k=config.TOP_K)
