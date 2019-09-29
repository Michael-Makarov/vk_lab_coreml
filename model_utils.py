import torch
import torch.utils.data
import pandas as pd
import numpy as np
import scipy.sparse
import implicit
from gensim.models import Word2Vec


class WRMFEmbedded(torch.nn.Module):
  def __init__(self, user_mapping, mapping, reverse_mapping, embedding, most_similar, size=100, alpha=0.1, lambd=0.01, cuda=True):
    self.device = torch.device('cuda') if cuda and torch.cuda.is_available() else torch.device('cpu')
    
    self.user_mapping = user_mapping
    self.mapping = mapping
    self.reverse_mapping = reverse_mapping
    embedding = torch.tensor(embedding, device=self.device)
    self.most_similar = torch.tensor(most_similar, device=self.device)
    self.size = size
    self.alpha = alpha
    self.lambd = lambd

    self.global_mean = torch.rand((), requires_grad=True, device=self.device)
    self.user_mean = torch.rand((len(user_mapping)), requires_grad=True, device=self.device)
    self.item_mean = torch.rand((len(mapping)), requires_grad=True, device=self.device)
    self.user_implicit = torch.rand((len(user_mapping), size), requires_grad=True, device=self.device)
    self.item_implicit = torch.rand((len(mapping), size), requires_grad=True, device=self.device)

    self.sim_values = torch.zeros(self.most_similar.size(), device=self.device)
    for i, sims in enumerate(self.most_similar):
      self.sim_values[i] = (embedding[i] * embedding[sims]).sum(dim=1)

  def prepare_data(self, data):
    items = torch.tensor([self.mapping[x] for x in data['movieId'].values], device=self.device)
    users = torch.tensor([self.user_mapping[x] for x in data['userId'].values], device=self.device)
    ratings = torch.tensor(data['rating'].values, dtype=torch.float32, device=self.device)
    dataset = torch.utils.data.TensorDataset(items, users, ratings)
    return dataset

  def calc_rating(self, user_mapped, item_mapped):
    values = (self.item_implicit[item_mapped] * self.user_implicit[user_mapped]).sum(dim=1)
    return self.global_mean + self.item_mean[item_mapped] + self.user_mean[user_mapped] + values

  def rank_items(self, user, items):
    with torch.no_grad():
      user = self.user_mapping[user]
      res = []
      mapped_items = [self.mapping[x] for x in items]
      user_t = torch.tensor([user])
      item_t = torch.tensor(mapped_items)
      rating = self.calc_rating(user_t, item_t).to(device=torch.device('cpu')).numpy()
      for true_item, ra in zip(items, rating):
        res.append((true_item, ra))
      return res

  def move_to_device(self, to_cpu):
    dev = torch.device('cpu') if to_cpu else self.device
    self.global_mean = self.global_mean.to(device=dev)
    self.user_mean = self.user_mean.to(device=dev)
    self.item_mean = self.item_mean.to(device=dev)
    self.user_implicit = self.user_implicit.to(device=dev)
    self.item_implicit = self.item_implicit.to(device=dev)

  def fit(self, prepared_data, epochs=3, verbose=0, batch_size=100, lr=0.1):
      params = [
        self.global_mean,
        self.user_mean,
        self.item_mean,
        self.user_implicit,
        self.item_implicit,
      ]
      optimizer = torch.optim.Adam(params, lr=lr)
      for epoch in range(epochs):
        if verbose > 1:
          print("Epoch {}".format(epoch))
        for param in params:
          if param.grad is not None:
            param.grad.zero_()

        loader = torch.utils.data.DataLoader(prepared_data, batch_size=batch_size)
        loss = 0
        for items, users, ratings in loader:
          value = 0.5 * torch.sum((self.calc_rating(users, items) - ratings).pow(2))
          value.backward()
          loss += value.item()

        if verbose > 1:
          print("Found prediction loss")
        
        sim_err = 0
        for i, sims in enumerate(self.most_similar):
          value = 0.5 * self.alpha * torch.pow(self.sim_values[i] - (self.item_implicit[i] * self.item_implicit[sims]).sum(dim=1), 2).sum()
          value.backward()
          sim_err += value.item()
        
        if verbose > 1:
          print("Found similarity loss")

        value = 0.5 * self.lambd * (torch.sum(self.item_implicit.pow(2)) + torch.sum(self.user_implicit.pow(2)))
        value.backward()
        regul = value.item()

        if verbose > 1:
          print("Found regularization loss")

        loss = loss + sim_err + regul

        optimizer.step()
        if verbose > 0:
          print("Epoch {} ended, loss {}".format(epoch, loss)) 

def split_data(data, test_part=0.2, min_test=10):
  grouped = data.groupby('userId')
  train = []
  test = []
  for name, group in grouped:
    entries = group.sort_values('timestamp')
    test_cnt = max(min_test, int(len(entries) * test_part))
    train.append(entries[:-test_cnt])
    test.append(entries[-test_cnt:])

  data_train = pd.concat(train)
  data_test = pd.concat(test)
  return data_train, data_test

def dataframe_to_csr(df, row, column, value, shape):
  return scipy.sparse.coo_matrix((df[value].values, (df[row].values, df[column].values)), shape=shape).tocsr()

def calc_dcg(y_true, y_pred, k):
  order = np.argsort(y_pred)[:-k - 1:-1]
  return np.sum((np.power(2, y_true[order]) - 1) / np.log2(np.arange(2, k + 2)))

def calc_ndcg(y_true, y_pred, k):
  return calc_dcg(y_true, y_pred, k) / calc_dcg(y_true, y_true, k)

def calc_mean_ndcg_als(model, csr_train, data, k):
  grouped = data.groupby('userId')
  res = []
  for user_id, group in grouped:
    ranked = model.rank_items(user_id, csr_train, group['movieId'].values)
    ranked.sort(key=lambda x: x[0])
    y_pred = np.array([x[1] for x in ranked])
    y_true = group.sort_values('movieId')['rating'].values
    ndcg = calc_ndcg(y_true, y_pred, k)
    res.append(ndcg)
  return np.mean(res)

def calc_mean_ndcg_wrmf(model, data, k):
  grouped = data.groupby('userId')
  res = []
  for user_id, group in grouped:
    ranked = WRMFEmbedded.rank_items(model, user_id, group['movieId'].values)
    y_pred = np.array([x[1] for x in ranked])
    y_true = group['rating'].values
    ndcg = calc_ndcg(y_true, y_pred, k)
    res.append(ndcg)
  return np.mean(res)


def get_sequences(data):
  grouped = data.groupby('userId')
  res = []
  users = []  
  for user_id, group in grouped:
    seq = group.sort_values('timestamp')['movieId'].values
    res.append(list(seq.astype(str)))
    users.append(user_id)
  return np.array(users), res

def get_embedding_data(users, sequences, k=2):
  res = []
  for user_id, seq in zip(users, sequences):
    for i in range(len(seq)):
      for dx in range(-k, k + 1):
        if dx != 0 and i + dx >= 0 and i + dx < len(seq):
          res.append(np.array([seq[i], seq[i + dx]]))
  return np.array(res)

def get_embedding(model, data, similar_cnt=5):
  model.init_sims()
  reverse_mapping = np.array(data['movieId'].unique())
  mapping = dict()
  embed = np.zeros((len(model.wv.vocab), model.vector_size))
  for i, num in enumerate(reverse_mapping):
    mapping[num] = i
    embed[i] = model.wv.word_vec(str(num), use_norm=True)
  similar = np.zeros((len(model.wv.vocab), similar_cnt), dtype=np.int)
  for i, num in enumerate(reverse_mapping):
    similar[i] = np.array([mapping[int(x[0])] for x in model.wv.most_similar(str(num), topn=similar_cnt)])
  return mapping, reverse_mapping, embed, similar

def get_user_mapping(data):
    mapping = dict()
    for i, num in enumerate(data['userId'].unique()):
      mapping[num] = i
    return mapping

def als_gridsearch(csr_train, csr_train_t, data_test, k, extra_k=[1], factors_grid = [32, 64, 128], regularization_grid = [0.001, 0.01, 0.1]):
  mean_values = []
  other_values = []
  args = []
  for factor in factors_grid:
    for reg in regularization_grid:
      als = implicit.als.AlternatingLeastSquares(factors=factor, regularization=reg, use_cg=True, calculate_training_loss=True, iterations=30, use_gpu=True)
      als.fit(csr_train, show_progress=True)
      mean = calc_mean_ndcg_als(als, csr_train_t, data_test, k)
      print("factors: {}, reg: {}, ndcg@{}: {}".format(factor, reg, k, mean))
      mean_values.append(mean)
      args.append((factor, reg))
      vals = []
      for ek in extra_k:
        val = calc_mean_ndcg_als(als, csr_train_t, data_test, ek)
        print("ndcg@{}: {}".format(ek, val))
        vals.append(val)
      other_values.append(vals)
  pos = np.argmax(mean_values)
  print("best ndcg: {}, args: {}".format(mean_values[pos], args[pos]))
  return mean_values[pos], args[pos], other_values[pos]

def get_als_values(all_data, data_train, data_test, use_gridsearch):
    csr_train = dataframe_to_csr(data_train, 'movieId', 'userId', 'rating', (all_data['movieId'].max() + 1, all_data['userId'].max() + 1))
    csr_train_t = csr_train.transpose().tocsr()
    if use_gridsearch:
        val, _, vals = als_gridsearch(csr_train, csr_train_t, data_test, 10, extra_k=[1])
    else:
        val, _, vals = als_gridsearch(csr_train, csr_train_t, data_test, 10, extra_k=[1], factors_grid=[32], regularization_grid=[0.01])
    return val, vals[0]
