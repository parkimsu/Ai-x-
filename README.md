# Ai-x-딥러닝
# Table of Contents
[I. Title](#Title)

[II. Members](#Members)

[III. Proposal](#Proposal)

[IV. Datasets](#Datasets)

[V. Methodology](#Methodology)

[VI. Evaluation & Analysis](#Evaluation_&_Analysis)

[VII. Related Work](#Related_Work)

[VIII. Related Work](#Related_Work)
# Title
### LSTM을 이용한 축구경기 결과 예측
# Members
송호영 | 전기공학 전공 | 2020064584 | eriklamela11@hanyang.ac.kr 

박준호 | 데이터사이언스 전공 | 2023076508 | dpsxlwnsgh@naver.com

하영현 | 전기공학 전공 | 2025062179 | voltha123@hanyang.ac.kr

| 이름   | 프로젝트 내 역할 |
|--------|------------------|
| 송호영 | 코드 구현, 데이터 수집, 동영상 촬영, Evaluation & Analysis 작성|
| 박준호 | 코드 구현, 데이터 수집, Mehtodology 작성, 블로그 작성|
| 하영현 | 코드 구현, 데이터 수집, Datasets 작성|


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
본 프로젝트에서는 English Premier League(EPL)의 1993/94부터 2020/21까지의 경기 데이터와 2000/01부터 2024/25까지의 선수 데이터를 사용하였습니다.

### 불필요한 열 및 결측값 처리
데이터셋에서 불필요한 열(Unnamed로 시작하는 열, HomeTeam, AwayTeam, FTR, DateTime)과 경기결과(FTR)가 없는 행은 제거한 뒤, 남은 결측값은 0으로 채웠습니다.
<pre><code>df = df.dropna(axis=1, how='all')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(subset=['FTR'])
df['FTR'] = df['FTR'].astype(str)

team_encoder = LabelEncoder()
df['home_id'] = team_encoder.fit_transform(df['HomeTeam'].astype(str))
df['away_id'] = team_encoder.transform(df['AwayTeam'].astype(str))

le = LabelEncoder()
df['FTR_encoded'] = le.fit_transform(df['FTR'])

df = df.drop(columns=['HomeTeam', 'AwayTeam', 'FTR', 'DateTime'], errors='ignore')
df = df.fillna(0)</code></pre>

### 범주형 변수 인코딩
팀 이름(HomeTeam, AwayTeam)과 경기결과(FTR)는 모델이 이해할 수 있도록 정수로 변환하였습니다.
<pre><code>team_encoder = LabelEncoder()
df['home_id'] = team_encoder.fit_transform(df['HomeTeam'].astype(str))
df['away_id'] = team_encoder.transform(df['AwayTeam'].astype(str))

le = LabelEncoder()
df['FTR_encoded'] = le.fit_transform(df['FTR'])</code></pre>

### 입력 특성 선택
home_id, away_id, FTR_encoded를 제외한 수치형 변수는 입력 특성으로 사용합니다.
<pre><code>numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in ['home_id', 'away_id', 'FTR_encoded']]</code></pre>

### 최종 데이터 구조
홈팀과 원정팀의 최근 10경기가 시퀀스로 만들어지며, 타깃값은 해당 경기의 FTR입니다.

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

<pre><code>df = pd.read_csv('epl_combined2.csv', encoding='cp949')
df = df.dropna(subset=['FTR']).fillna(0)
df['home_id'] = LabelEncoder().fit_transform(df['HomeTeam'])
df['away_id'] = LabelEncoder().fit_transform(df['AwayTeam'])
df['FTR_encoded'] = LabelEncoder().fit_transform(df['FTR'])

home_X, away_X, y, home_ids, away_ids = make_home_away_sequences(df)
home_X_ids = np.repeat(home_ids[:, None], 10, axis=1)
away_X_ids = np.repeat(away_ids[:, None], 10, axis=1)
</code></pre>

### 모델 입력 구조
모델은 총 세 개의 입력을 받습니다.

num_input: 홈/원정 경기의 수치형 시계열 피처.shape = (batch_size, 10, 16)

home_id_input: 홈팀 ID 시퀀스.shape = (batch_size, 10)

away_id_input: 원정팀 ID 시퀀스.shape = (batch_size, 10)

팀 ID는 고정된 벡터가 아닌 시퀀스 형태로 넣고, Embedding을 거쳐 수치피처와 동일한 시계열 공간에서 처리될 수 있도록 구성합니다.

<pre><code>num_in = layers.Input(shape=(n_step, n_feat), name='num_input')
home_id_in = layers.Input(shape=(n_step,), name='home_id_input')
away_id_in = layers.Input(shape=(n_step,), name='away_id_input')
</code></pre>

### Embedding 및 Concatenation
팀 ID(home_id, away_id)는 Embedding(N_TEAMS, 8)을 거쳐 8차원 벡터로 변환됩니다.

그 후, TimeDistributed(Dense(8))층을 통해 각 시간 단계마다 추가적인 표현력을 갖도록 변환됩니다. 즉 단순한 팀 벡터가 아니라, 경기에 따라 달라질 수 있는 벡터로 보완됩니다. 

이렇게 변환된 임베딩 벡터는 수치형 경기 기록 피처, 홈팀 임베딩 벡터, 원정팀 임베딩 벡터로 결합됩니다.

세 가지를 하나의 시퀀스 피처로 합치면 각 시점의 입력 벡터는 최종적으로 32차원이 됩니다.

즉, 최종 입력 데이터는 shape = (batch_size, 10, 32)로 구성되며, 이는 10경기의 각 시점마다 32개의 특성을 갖는 시계열 데이터라고 할 수 있습니다.

<pre><code>num_in = layers.Input(shape=(n_step, n_feat), name='num_input')
home_id_in = layers.Input(shape=(n_step,), name='home_id_input')
away_id_in = layers.Input(shape=(n_step,), name='away_id_input')

home_emb = layers.Embedding(input_dim=N_TEAMS, output_dim=EMB_DIM)(home_id_in)
away_emb = layers.Embedding(input_dim=N_TEAMS, output_dim=EMB_DIM)(away_id_in)

home_emb = layers.TimeDistributed(layers.Dense(8))(home_emb)
away_emb = layers.TimeDistributed(layers.Dense(8))(away_emb)

x = layers.Concatenate()([num_in, home_emb, away_emb])
</code></pre>

### LSTM 기반 시계열 처리
첫번쨰 계층: Bidirectional(LSTM(256, return, sequences=True)) -> 시계열 정보를 양방향으로 학습하며, 다음 LSTM 계층 입력을 위해 시퀀스 형태를 유지합니다.

중간 정규화: SpatialDropout1D(0.3)을 삽입하여 overfitting 방지 및 시퀀스 feature 간 독립성을 확보합니다.

두번째 계층: Bidirectional(LSTM(128)) -> 시퀀스 정보를 압축하여 단일 벡터로 변환합니다.

이후 Dense(128) -> BatchNormalization() -> Dropout(0.4) 계층을 거쳐 모델의 일반화 성능을 향상시킵니다.

<pre><code>x = layers.Concatenate()([num_in, home_emb, away_emb])
x = layers.LayerNormalization()(x)
x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
x = layers.SpatialDropout1D(0.3)(x)
x = layers.Bidirectional(layers.LSTM(128))(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
</code></pre>

### 출력 및 손실 함수
출력 계층은 Dense(3, activation='softmax')로 구성되며, H, D, A 세 가지 클래스에 대한 확률값을 예측합니다.

손실 함수는 Focal Loss를 사용합니다. 이는 Cross Entropy에 비해 어렵게 예측되는 샘플에 더 큰 가중치를 부여함으로써 불균형 데이터셋에서 모델의 집중력을 높입니다. 하이퍼파라미터로는 감쇠 계수 y = 2.0, 클래스별 가중치 α = [1.2, 1.5, 0.8]을 사용하여 무승부에 높은 중요도를 부여하였습니다.

<pre><code>out = layers.Dense(3, activation='softmax')(x)

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2., alpha=[1.2, 1.5, 0.8], **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        ce = -y_true * tf.keras.backend.log(y_pred)
        weight = self.alpha * tf.keras.backend.pow(1 - y_pred, self.gamma)
        return tf.keras.backend.sum(weight * ce, axis=1)
</code></pre>
### 학습 전략
Adam 옵티마이저(learning rate = 0.0003)을 사용하여 학습 안정성과 수렴 속도를 균형 있게 확보하였고, 클래스 불균형 문제를 완화하기 위해 Focal Loss를 도입하였습니다. 또한 compute, class_weight 함수를 활용하여 데이터 내 클래스 비율에 따라 자동으로 가중치를 계산하고, 학습 시 반영함으로써 불균형 분포를 보완하였습니다.

검증 손실이 8 epoch 이상 개선되지 않을 경우 학습을 종료하고, 가장 성능이 좋았던 모델 가중치를 복원하도록 설정하여 과적합을 방지하였습니다. 배치 크기는 64로 설정하였으며, 최대 80 epoch까지 학습하도록 하였습니다. 훈련 데이터 중 10%를 검증에 사용하였습니다. 

<pre><code>model = models.Model([num_in, home_id_in, away_id_in], out)
model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
              loss=FocalLoss(),
              metrics=['accuracy'])

cb = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

history = model.fit([home_tr, home_ids_tr, away_ids_tr], y_tr_cat,
                    validation_split=0.1,
                    epochs=80,
                    batch_size=64,
                    class_weight=CLASS_WEIGHT,
                    callbacks=[cb],
                    verbose=2)
</code></pre>


# Evaluation & Analysis
본 프로젝트에서는 LSTM 모델을 사용하여 EPL 경기 결과(홈 승리, 무승부, 원정 승리)를 예측습니다. 모델의 전체 정확도는 42.04%, loss(손실 함수 값)는 0.5108로 나타났습니다.
정확도(accuracy)는 전체 예측 중 맞춘 비율이고,
loss는 예측 확률이 정답에 얼마나 가까웠는지를 나타내는 수치로,
0에 가까울수록 좋은 모델임을 의미합니다.
따라서 이번 결과는 무작위 예측보다 좋은 수준의 성능을 보여줬다고 볼 수 있습니다.

![딥러닝8](https://github.com/user-attachments/assets/51acc895-df91-4705-bb5e-14273fdadceb)

모델의 예측 성능을 더 자세히 이해하기 위해
precision, recall, f1-score라는 세 가지 지표로 평가하였습니다.

![딥러닝12](https://github.com/user-attachments/assets/053b1c39-217f-424e-aeec-02029d2328ed)


### 각 지표 설명
Precision :
모델이 A/D/H라고 예측한 것 중 실제로 맞은 비율로, D의 precision이 0.28이면, 무승부라고 예측한 경기 중 28%만 진짜 무승부였다는 의미입니다.

Recall :
실제로 A/D/H였던 경기 중에서 모델이 정확히 예측한 비율로 D의 recall이 0.64라면, 실제 무승부 경기의 64%를 맞췄다는 의미입니다.

F1-score:
precision과 recall의 조화 평균으로 두 지표가 모두 높아야 F1도 높아진다. D는 둘의 균형이 깨져 0.3951로 낮은 편입니다.

Support:
각 클래스가 데이터셋에 얼마나 등장했는지를 나타냅니다. 홈 승리가 가장 많고, 무승부는 상대적으로 적습니다.

Classification Report 하단에는 모델의 전반적인 성능을 요약하는 세 가지 지표가 있습니다.

accuracy는 전체 예측 중 정답을 맞힌 비율로, 가장 직관적인 성능 지표입니다.

이번 모델의 accuracy는 0.42, 즉 전체 경기 중 약 42%를 정확히 예측했습니다.

macro avg는 각 클래스(A, D, H)의 precision, recall, f1-score를 동등한 가중치로 평균 낸 값입니다.

클래스가 균형을 이뤘을 때 의미 있지만, 클래스 비율이 크게 차이 나는 경우에는 성능을 과대 또는 과소평가할 수 있습니다.

weighted avg는 각 클래스의 등장 횟수(support)를 가중치로 반영해 평균을 낸 값입니다.

실제 데이터에서 홈 승리(H)가 가장 많기 때문에, H 클래스의 성능이 상대적으로 더 많이 반영된다.

클래스 불균형이 존재하는 상황에서는 weighted avg가 모델 전체 성능을 보다 현실적으로 평가할 수 있는 기준이 됩니다.

이 세 가지 수치를 함께 참고하면, 단순 정확도만 볼 때보다 훨씬 더 세밀한 모델 해석이 가능합니다.

### Confusion Matrix 분석
Confusion Matrix는 모델이 어떤 클래스(A, D, H)를 얼마나 잘 예측했는지를 구체적으로 보여준다.

![딥러닝11](https://github.com/user-attachments/assets/b73d5e68-deee-4f96-8e6d-01d61a34fd36)


표의 일부를 살펴보면 다음과 같습니다.

509: 실제 A였고 예측도 A → 정확한 예측 

640: 실제 A였는데 D로 예측 → 무승부로 잘못 예측 

720: 실제 D였고 예측도 D → 무승부 예측 성공

1,159: 실제 H였지만 D로 예측 → 무승부 예측에 편향된 경향

특히 눈에 띄는 부분은 무승부(D)로 예측한 경우가 전체적으로 많다는 점입니다. 실제 D뿐 아니라 A와 H도 상당수 D로 예측되었습니다. 이는 모델이 무승부를 예측하는 데 있어 높은 recall을 기록했지만 precision은 낮았고, 홈이나 원정 승리를 예측하는 데 있어 높은 precision을 기록했지만 recall이 낮았다는 Classification Report의 내용과 정확히 일치합니다.

### 결론 및 개선점

이번 결과에서는 모델이 무승부로 예측하는 비율이 지나치게 높은 경향을 보였습니다. 홈 원정 승리 모두 정확히 예측한 것보다도 무승부로 예측한 수가 더 많았습니다.

이는 경기의 승패를 예측하는 데이터가 부족하여 승과 패를 나누지 못하고 무승부로 예측하는 경우가 많았다고 해석할 수 있습니다.


# Related Work

### 데이터 수집
https://www.kaggle.com/datasets/irkaal/english-premier-league-results

https://fbref.com/en/comps/9/stats/Premier-League-Stats#site_menu_link

### 참고 사이트
https://wikidocs.net/152773

