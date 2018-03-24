# DQN (MountainCar)

##How to run 

CPU
 
`python main.py` 

GPU 

`python main.py -g gpu_number` 

##Example 

`python main.py -c "ConstrainedDQN_pw2_re1_f100_sd0" -dp result -m regularize -pw 2 -re 1.0 -sd 0` 

"ConstrainedDQN_pw2_re1_f100_sd0"というフォルダ名で、"result"というディレクトリにregularizeモード(constrainedモード)、penalty weight = 2、rmsprop epsilon = 1.0、seed = 0の結果を保存。 

##Requirement 
- Python 2.7.12 
- chainer 2.0.2 
- gym 0.9.2 

##Result
- log: エピソードごとの報酬などのデータ 
- evaluation: 一定ステップごとに評価を行なった結果
- loss: 一定ステップごとのlossとpenalty(制約項)の値の記録
- network: 一定ステップごとのネットワーク
- replay_memory: 最終的なリプレイメモリ内のデータ

##How to make graph 

`python make_multi_graph_2.py` 

##Example 

`python make_multi_graph_2.py -m test -dr result -k DQN_re0.01 -o DQN_re0.01 -a 1 -g 1` 

テストモード(evaluationの結果を)resultディレクトリの"DQN_re0.01"という　キーワードを含む全てのフォルダについてグラフを描し、"DQN_re0.01.png"というファイル名で出力。また、各エピソードごとの勾配の絶対値の平均値の図も出力。




