# Ai-x-딥러닝
# Table of Contents
[I. Title](#Title)

[I. Members](#Members)

[I. Proposal](#Proposal)

[I. Datasets](#Datasets)

[I. Methodology](#Methodology)

[I. Evaluation & Analysis](#Evaluation_&_Analysis)

[I. Related Work](#Related_Work)
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
# Evaluation & Analysis
# Related Work
