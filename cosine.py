from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


clusters = np.load('resnet_res_final.npy', allow_pickle=True)

url_map = dict()
ct = 0

id_to_res = dict()

for k in range(5):
  with open('resnet_correspondences' + str(k) + '.txt','r') as f:
    urls = f.read().split(' ')
  
    for j in range(len(urls)-1):
      if j == 0:
        url_map[urls[j]] = [ct]
        ct += 1
      else:
        idx = urls[j].split('h')[0]
        cur_url = urls[j][len(urls[j].split('h')[0]):]
        if cur_url in url_map:
          url_map[cur_url].append(ct)
          id_to_res[int(idx)] = ct
          ct += 1
        else:
          url_map[cur_url] = [ct]
          ct += 1
          id_to_res[int(idx)] = ct

print(ct)
print(len(url_map))
inv_url_map = {v[0] : k for k,v in url_map.items()}
#ct = 0
#for k,v in url_map.items():
#  # k is url, v is number
#  print(v)
#  print(k)
#  v.sort()
#  ct += 1
#  inv_url_map[v[0]] = k
    
#inv url map is a mapping for idx to url
print(len(inv_url_map))
cols = ['URL', 'CAPTION']
caps = pd.read_csv('../TGIF-Release-master/data/tgif-v1.0.tsv', names=cols, sep='\t')

def get_cluster_urls(cluster):
  ids = []
  for i in range(clusters.shape[0]):
    if clusters[i] == cluster:
      ids.append(i)
  urls = [inv_url_map[x] for x in ids]
  c = []
  for u in urls:
    if len(caps[caps['URL'] == u]['CAPTION']) != 0:
      c.append(caps[caps['URL'] == u]['CAPTION'].tolist()[0])
  return c
 
def compute_cosine(inp, cluster):
  captions = get_cluster_urls(cluster)
  vocab = dict()
  for w in inp.split(' '):
    if w not in vocab:
      vocab[w] = len(vocab)
  for c in captions:
    for w in c.split(' '):
      if w not in vocab:
        vocab[w] = len(vocab)
  
  inp_bow = np.zeros(len(vocab))
  cluster_bow = np.zeros(len(vocab))
  for w in inp.split(' '):
    inp_bow[vocab[w]] += 1
  for c in captions:
    for w in c.split(' '):
      cluster_bow[vocab[w]] += 1
  # Normalize
  inp_bow = inp_bow/np.linalg.norm(inp_bow)
  cluster_bow = cluster_bow/np.linalg.norm(cluster_bow)

  return np.dot(inp_bow, cluster_bow)
ct = 0
with open('res_training_val.tsv') as f:
    lns = f.read().split('\n')
    res = []
    print(len(lns))
    for l in lns:
      print(ct) 
      cap = l.split('[SEP]')[0].replace(".", "")
      tk = l.split('[SEP]')[1]
      tk = tk[2:]
      if len(tk) > 0:
        res.append(compute_cosine(cap, int(tk)))
      ct += 1
    print(np.mean(np.array(res)))
