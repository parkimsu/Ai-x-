# Ai-x-딥러닝
# Table of Contents
[I. Title](#Title)

[II. Members](#Members)

[III. Proposal](#Proposal)

[IV. Datasets](#Datasets)

[V. Methodology](#Methodology)

[VI. Evaluation & Analysis](#Evaluation_&_Analysis)

[VII. Related Work](#Related_Work)
# Title
### LSTM을 이용한 축구경기 결과 예측
# Members
송호영 | 전기공학 전공 | 2020064584 | eriklamela11@hanyang.ac.kr 

박준호 | 데이터사이언스 전공 | 2023076508 | dpsxlwnsgh@naver.com

하영현 | 전기공학 전공 | 2025062179 | voltha123@hanyang.ac.kr
# Proposal (Option A)
### Motivation:
최근 손흥민 선수가 활약하고 있는 토트넘 홋스퍼가 무려 17년 만에 공식 대회 우승에 도전한다는 소식이 큰 화제가 되고 있습니다. 

이 같은 뉴스는 축구 팬들뿐만 아니라 스포츠 데이터 분석가들에게도 큰 관심을 불러일으키고 있습니다. 

저희 팀은 이 기사를 접하며 다가오는 유로파리그 결승전의 승패가 어떻게 결정될지에 대해 자연스럽게 궁금증이 생겨 승/패 예측 모델을 만들어보기로 하였습니다.
### Our Goal:
축구에서 팀의 승/패, 득점력, 수비력은 직전 경기들에 영향을 많이 받습니다.

최근 5경기에서 연패를 했냐 연승을 했냐에 따라 컨디션의 흐름이 많이 달라집니다. 

LSTM은 데이터를 시간순으로 넣으면 과거 경기들의 영향력을 자동 학습하고, 내부적으로 시간의 흐름에 따라 기억/망각을 조절합니다. 또한 최근 경기에서
어떤 선수가 부진하거나, 새로운 전술을 시도 할 경우 LSTM은 이 정보를 기억했다가 중요할때 활용합니다.

저희는 이 LSTM을 이용하여 홈팀의 승, 원정팀의 승, 무승부 등을 예측하는 모델을 만들 것입니다.
# Datasets
# Methodology
본 프로젝트에서는 EPL 경기 결과 예측을 위한 시계열 분류 모델로 LSTM(Long Short-Term Memory) 기반 딥러닝 모델을 구성하였으며, 클래스 불균형 문제를 완화하기 위해 Focal Loss를 손실 함수로 채택하였습니다.
### What is LSTM?
LSTM은 RNN의 한 종류로, 기존 RNN이 가진 장기 의존성 문제(long-term dependency)를 해결하기 위해 고안된 모델입니다.
이는 단순히 바로 이전 정보뿐만 아니라, 더 과거의 중요한 정보를 장기적으로 기억하고 반영하여 미래 값을 예측할 수 있도록 설계되었습니다.

![LSTM구조](https://github.com/user-attachments/assets/5d643bf1-a4fe-445f-8b2f-69bf304d6ed3)

위의 그림과 같이 모든 RNN 은 Neural Network 모듈을 반복시키는 체인과 같은 형태를 하고 있습니다. 기본적으로 RNN에서는 이렇게 반복되는 간단한 구조를 가지고 있습니다.

![딥러닝2](https://github.com/user-attachments/assets/dea5472f-cea7-4461-93bf-c16be1082586)

위의 그림은 LSTM 네트워크의 내부 구조를 시각화한 것입니다.
LSTM도 똑같이 체인 구조를 가지고 있지만, 4개의 Layer가 특별한 방식으로 서로 정보를 주고 받도록 되어있습니다.

![딥러닝3](https://github.com/user-attachments/assets/8e43ee6c-bb60-4a28-a87a-d77d54ad7901)

LSTM은 위의 그림과 같이 총 6개의 파라미터와 4개의 게이트로 이루어져 있습니다.

Cell state는 LSTM의 기억 저장소 역할을 하며, 수평선처럼 전체 시퀀스를 따라 흐르는 경로입니다. 이는 컨베이어 벨트처럼 작동하여, 작은 선형 변화만 적용하며 정보를 전달합니다. 덕분에 정보 손실이 적고, 시간이 오래 지나도 Gradient가 잘 전파됩니다.

Forget Gate는 과거 정보를 얼마나 유지할지 결정하는 역할을 합니다. 결과는 0과 1 사이의 값으로, 1에 가까울수록 정보를 보존하고 0에 가까울수록 정보를 제거합니다.

Input Gate는 현재 정보를 기억하기 위한 게이트 입니다. 현재의 Cell state 값에 얼마나 더할지 말지를 정하는 역할을 합니다.

Update는 Forget Gate와 Input Gate의 출력을 활용해 이루어지며, Forget Gate는 이전 기억에서 얼마나 버릴지, Input Gate는 새로운 정보를 얼마나 더할지 결정합니다.

Output Gate는 어떤 출력값을 출력할지 결정하는 과정으로 최종적으로 얻어진 Cell State 값을 얼마나 빼낼지 결정하는 역할을 해줍니다.

### 데이터 전처리 및 시계열 입력 구성
각 경기는 하나의 샘플이 아니라 과거 경기 기록들을 기반으로 한 시계열(sequence)로 처리됩니다. 구체적으로, 각 팀에 대해 과거 SEQ_LEN = 10개의 경기를 기반으로 피처 벡터를 만들고, 해당 팀이 이후에 치른 경기에 대한 실제 결과(FTR_encoded)를 레이블로 사용합니다.

입력 피처는 경기 기록 지표 (FTHG, FTAG, HST, AST, HY, AY 등)로 구성된 16차원의 수치 벡터입니다.

각 팀에 대해 홈 경기/원정 경기 모두를 고려해 순서대로 데이터를 정렬한 후, 슬라이딩 윈도우 방식으로 시퀀스를 생성합니다. 시퀀스 길이는 10으로 고정됩니다. 추가적으로 팀 ID를 Embedding 처리하기 위해 home_id, away_id를 LabelEncoder를 통해 정수 인코딩한 후, 이를 시퀀스 정체 길이에 맞게 복제하여 LSTM의 입력으로 사용합니다.

![딥러닝5](https://github.com/user-attachments/assets/7b439006-1fd3-48e5-9779-b4509f67497f)

### 모델 입력 구조
모델은 총 세 개의 입력을 받습니다.

num_input: 홈/원정 경기의 수치형 시계열 피처.shape = (batch_size, 10, 16)

home_id_input: 홈팀 ID 시퀀스.shape = (batch_size, 10)

away_id_input: 원정팀 ID 시퀀스.shape = (batch_size, 10)

팀 ID는 고정된 벡터가 아닌 시퀀스 형태로 넣고, Embedding을 거쳐 수치피처와 동일한 시계열 공간에서 처리될 수 있도록 구성합니다.

### Embedding 및 Concatenation
팀 ID(home_id, away_id)는 Embedding(N_TEAMS, 8)을 거쳐 8차원 벡터로 변환됩니다.

그 후, TimeDistributed(Dense(8))층을 통해 각 시간 단계마다 추가적인 표현력을 갖도록 변환됩니다. 즉 단순한 팀 벡터가 아니라, 경기에 따라 달라질 수 있는 벡터로 보완됩니다. 

이렇게 변환된 임베딩 벡터는 수치형 경기 기록 피처, 홈팀 임베딩 벡터, 원정팀 임베딩 벡터로 결합됩니다.

세 가지를 하나의 시퀀스 피처로 합치면 각 시점의 입력 벡터는 최종적으로 32차원이 됩니다.

즉, 최종 입력 데이터는 shape = (batch_size, 10, 32)로 구성되며, 이는 10경기의 각 시점마다 32개의 특성을 갖는 시계열 데이터라고 할 수 있습니다.

### LSTM 기반 시계열 처리
첫번쨰 계층: Bidirectional(LSTM(256, return, sequences=True)) -> 시계열 정보를 양방향으로 학습하며, 다음 LSTM 계층 입력을 위해 시퀀스 형태를 유지합니다.

중간 정규화: SpatialDropout1D(0.3)을 삽입하여 overfitting 방지 및 시퀀스 feature 간 독립성을 확보합니다.

두번째 계층: Bidirectional(LSTM(128)) -> 시퀀스 정보를 압축하여 단일 벡터로 변환합니다.

이후 Dense(128) -> BatchNormalization() -> Dropout(0.4) 계층을 거쳐 모델의 일반화 성능을 향상시킵니다.

### 출력 및 손실 함수
출력 계층은 Dense(3, activation='softmax')로 구성되며, H, D, A 세 가지 클래스에 대한 확률값을 예측합니다.

손실 함수는 Focal Loss를 사용합니다. 이는 Cross Entropy에 비해 어렵게 예측되는 샘플에 더 큰 가중치를 부여함으로써 불균형 데이터셋에서 모델의 집중력을 높입니다. 하이퍼파라미터로는 감쇠 계수 y = 2.0, 클래스별 가중치 α = [1.2, 1.5, 0.8]을 사용하여 무승부에 높은 중요도를 부여하였습니다.

### 학습 전략
Adam 옵티마이저(learning rate = 0.0003)을 사용하여 학습 안정성과 수렴 속도를 균형 있게 확보하였고, 클래스 불균형 문제를 완화하기 위해 Focal Loss를 도입하였습니다. 또한 compute, class_weight 함수를 활용하여 데이터 내 클래스 비율에 따라 자동으로 가중치를 계산하고, 학습 시 반영함으로써 불균형 분포를 보완하였습니다.

검증 손실이 8 epoch 이상 개선되지 않을 경우 학습을 종료하고, 가장 성능이 좋았던 모델 가중치를 복원하도록 설정하여 과적합을 방지하였습니다. 배치 크기는 64로 설정하였으며, 최대 80 epoch까지 학습하도록 하였습니다. 훈련 데이터 중 10%를 검증에 사용하였습니다. 






# Evaluation & Analysis
# Related Work
