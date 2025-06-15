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


# Evaluation & Analysis
# Related Work
