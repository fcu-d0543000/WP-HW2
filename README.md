# 簡介
- 所有程式以程式語言**Python**編寫。
- 請使用**jupyter notebook**執行**ipynb**檔案，接著從頭執行到尾就行了!
- 程式執行步驟
	- 依照檔名 {數字}_ 順序執行，優先度小>大。
	- 0_XXX，可直接執行。 舉例：0_linearSVR.ipynb, 0_rnn_不分組.ipynb ......。
    - 1_XXX，不須先執行0_XXX程式，可直接執行**程式** 或 **資料夾內程式**。
	- 2_XXX，請先執行過**1_XXX程式** 或 **1_XXX資料夾內的程式**。
    - 以此類推。
    
## 分組簡介
1. **label_AUM12_sd**：使用AUM *1~12*月的標準差的四分位數分組，得到*4*組後，再將最後一組以四分位數再分組一次，得到*7*組，再將最後一組再分組一次，共分得*10*組。
1. **graph_cluster_aum9_1_150000**：以*SERIAL_NUM*照順序分為*3*組，分別為*1 ~ 50000*、*50001 ~ 100000* 與 *100001 ~ 150000*，再將各組以客戶的AUM *1~9*月的相似度矩陣各分為*4*群，共得*12*群。
1. **graph_cluster_aum9_STD_Q0123**：使用AUM *1~12*月的標準差的四分位數分組，得到*4*組後，再將各組以客戶的AUM *1~9*月的相似度矩陣各分為*4*群，共得*16*群。
1. **graph_cluster_aum9_STD_Q0123_Q0123**：將分組*graph_cluster_aum9_STD_Q0123*的*16*群以AUM 第12個月的四分位數再各分為4組，共得*64*組。

## 程式簡介
1. **0_similarity_background.ipynb**：用於計算客戶背景相似度，並以Graph方式呈現相似度接近而連接起來的客戶網絡。
1. **0_linearSVR.ipynb**：使用基礎的LinearSVR模型作為BaseLine，並觀察模型的Coefficient。
1. **0_rnn_不分組.ipynb**：使用Recurrent neural network(RNN) base 的LSTM、GRU模型，在此檔案中不做分組訓練。
1. **0_rnn_總資產STD的分位數分成10組.ipynb**：使用Recurrent neural network(RNN) base 的LSTM、GRU模型，分組使用*label_AUM12_sd*。
1. **0_xgboost_不分組_觀察特徵重要性.ipynb**：使用XGBoost模型，在此檔案中不做分組訓練，並重複挑選重要度Top*k*的特徵重新訓練模型，觀察特徵的重要性。
1. **0_xgboost_總資產STD的分位數分成10組.ipynb**：使用XGBoost模型，訓練完XGBoost模型後，再訓練Tree Embedding Model，分組使用*label_AUM12_sd*。
1. **1_graph_cluster_aum9_1_150000**：此資料夾為計算分組*graph_cluster_aum9_1_150000*的程式，請依照順序執行。
1. **1_graph_cluster_aum9_STD_Q0123**：此資料夾為計算分組*graph_cluster_aum9_STD_Q0123*與*graph_cluster_aum9_STD_Q0123_Q0123*的程式，請依照順序執行。
1. **2_node2vec_cluster.ipynb**：使用Graph neural network(GNN) base 的node2vec模型，從客戶的AUM 1~12月的相似度矩陣訓練出node(user) embedding，再以*K*-means演算法分群，目前只有抽樣3000人來計算，尚未使用再這次的進度中。
1. **2_rnn_圖(相似度)分群.ipynb**：使用Recurrent neural network(RNN) base 的LSTM、GRU模型，分組可挑選*graph_cluster_aum9_1_150000*、*graph_cluster_aum9_STD_Q0123* 或 *graph_cluster_aum9_STD_Q0123_Q0123*。
1. **2_xgboost_圖(相似度)分群.ipynb**：使用XGBoost模型，訓練完XGBoost模型後，再訓練Tree Embedding Model，分組可挑選*graph_cluster_aum9_1_150000*、*graph_cluster_aum9_STD_Q0123* 或 *graph_cluster_aum9_STD_Q0123_Q0123*。
1. **tem.py**：Tree Embedding Model 的 Function 設定，不需要執行。
1. **xgb_utils.py**：XGBoost Model 的 Function 設定，不需要執行。

## 模型參數
- RNN：在cell設定要使用LSTM 或 GRU模型，與其他可調的超參數。
``` python
params = {
    'cell': 'LSTM', # LSTM or GRU
    'embedding_size': uid_train.max()+1,
    'embedding_dim': 16,
    'input_dim': x_train.shape[-1],
    'hidden_dim': 16, 
    'n_layers': 2,
    'learning_rate': 2e-2,
    'batch_size': 2048,
    'epoch': 20,
    'bidirectional':False,
    'early_stop':5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}
```
- XGBoost：
``` python
xgb_params = {"objective":"reg:squarederror", 'learning_rate': 0.05, 'max_depth': 6, 
              'subsample':0.7, 'min_child_weight':1, 'eval_metric':'rmse', 'n_jobs':40}
```
- Tree Embedding Model：設定use_attention來決定是否使用Attention，與其他可調的超參數。
``` python
tem_params = {
    'use_attention': True, # True or False
    'user_num': df.SERIAL_NUM.max()+1,
    'node_num': 0,
    'embed_dim': 32,
    'hidden_dim': 32,
    'learning_rate': 5e-3,
    'max_grad_norm': 1.5,
    'batch_size': 1024,
    'epoch': 400,
    'early_stop': 5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}
```

