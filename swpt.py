import numpy as np
import seaborn
import pywt
import re
import matplotlib.pyplot as plt
plt.style.use(['seaborn'])
from pywt._thresholding import hard, soft


class SWPT(object):

  def __init__(self, wavelet='db4', max_level=3, start_level=0):
    self._wavelet = wavelet
    self._start_level = 0
    self._max_level = max_level - start_level
    self._coeff_dict = {}
    self._entropy_dict = {}


  def _pre_decompose(self, signal, a):
    if a > 0:
      coeff = pywt.swt(np.squeeze(signal),
                       wavelet=self._wavelet,
                       level=a)
      print(coeff[0][0].shape)
      return coeff[0][0]
    else:
      return signal
    
  def decompose(self, signal, entropy='shannon'):
    pth = ['']
    self._signal = self._pre_decompose(signal, self._start_level)
    self._coeff_dict[''] = np.squeeze(self._signal)
    for l in range(self._max_level):
      pth_new = []
      for p in pth:
        coeff = pywt.swt(
            self._coeff_dict[p],
            wavelet=self._wavelet,
            level=self._max_level - len(p),
            start_level=len(p))
        p_run = p
        for i, C in enumerate(coeff[::-1]):
          self._coeff_dict[p_run + 'A'] = C[0]
          self._coeff_dict[p_run + 'D'] = C[1]
          self._entropy_dict[p_run + 'A'] = 0.0
          self._entropy_dict[p_run + 'D'] = 0.0
          for c in C[0]:
            self._entropy_dict[p_run + 'A'] += self._get_entropy(c, signal, entropy)
          self._entropy_dict[p_run + 'A'] = self._entropy_dict[p_run + 'A'] / 2 ** (len(p_run) + 2.0 + self._start_level)
          for c in C[1]:
            self._entropy_dict[p_run + 'D'] += self._get_entropy(c, signal, entropy)
          self._entropy_dict[p_run + 'D'] = self._entropy_dict[p_run + 'D'] / 2 ** (len(p_run) + 2.0 + self._start_level)
          if i < len(coeff) - 1 and len(p_run) < self._max_level - 1:
            pth_new.append(p_run + 'D')
            p_run = p_run + 'A'
      pth = list(pth_new)

  def get_level(self, level, order='freq', thresholding=None, threshold=None):
    assert order in ['natural', 'freq']
    r = []
    result_coeffs = []
    result_energies = []
    for k in self._coeff_dict:
      if len(k) == level:
        r.append(k)
    if order == 'freq':
      graycode_order = self._get_graycode_order(level)
      for p in graycode_order:
        if p in r:
          result_coeffs.append(self._coeff_dict[p])
    else:
      print('The natural order is not supported yet.')
      exit(1)
    # apply the thressholding
    if thresholding in ['hard', 'soft']:
      if isinstance(threshold, (int, float)):
        if thresholding == 'hard':
          result_coeffs = hard(result_coeffs, threshold)
        else:
          result_coeffs = soft(result_coeffs, threshold)
      else:
        print('Threshold must be an integer or float number')
        exit(1)
    return result_coeffs

  def get_coefficient_vector(self, name):
    return self._coeff_dict[name]

  def _get_graycode_order(self, level, x='A', y='D'):
    graycode_order = [x, y]
    for i in range(level - 1):
      graycode_order = [x + path for path in graycode_order] + \
                       [y + path for path in graycode_order[::-1]]
    return graycode_order

  def _get_entropy(self, c, s, entropy):
    if entropy == 'shannon':
      return -np.log(c ** 2 / np.linalg.norm(s, ord=2) ** 2 ) * c ** 2 / np.linalg.norm(s, ord=2) ** 2
    elif entropy == "log-entropy":
      return np.log(c ** 2 / np.linalg.norm(s, ord=2) ** 2 )


  def best_basis(self, threshold=0.0):
    best_entropy = {}
    best_entropy_fr = {}
    levels = self._max_level
    cur_keys = [k for k in self._coeff_dict if len(k)==levels]
    lev_ord = self._get_graycode_order(levels)
    frqs = np.linspace(0.0 + 0.5/(2 * len(lev_ord)), 0.5 - 0.5/(2 * len(lev_ord)), len(lev_ord))
    for i in range(len(lev_ord)):
      best_entropy_fr[lev_ord[i]] = frqs[i]
    for k in cur_keys:
      best_entropy[k] = self._entropy_dict[k]
    for lev in range(levels-1, 0, -1):
      cur_keys = [k for k in self._coeff_dict if len(k)==lev]
      lev_ord = self._get_graycode_order(lev)
      frqs = np.linspace(0.0 + 0.5/(2 * len(lev_ord)), 0.5 - 0.5/(2 * len(lev_ord)), len(lev_ord))
      for i in range(len(lev_ord)):
        best_entropy_fr[lev_ord[i]] = frqs[i]
      for k in cur_keys:
        if self._entropy_dict[k] < self._entropy_dict[k+'A'] + self._entropy_dict[k+'D']:
          best_entropy[k] = self._entropy_dict[k]
          rx = re.compile(k+'.')
          del_key = [q for q in best_entropy if rx.search(q)]
          for d in del_key:
            del best_entropy[d]
            del best_entropy_fr[d]
          # print('prunning')
        else:
          best_entropy[k] = self._entropy_dict[k+'A'] + self._entropy_dict[k+'D']
    for lev in range(1, levels):
      cur_keys = [k for k in self._coeff_dict if len(k)==lev]
      for k in cur_keys:
        if k+'A' in best_entropy.keys():
          del best_entropy[k]
          del best_entropy_fr[k]
    total_entropy = sum(best_entropy.values())
    sorted_x = sorted(best_entropy_fr.items(), key=lambda kv: kv[1])
    best_tree = []
    sig_norm = np.linalg.norm(self._signal, 2) ** 2
    for x in sorted_x:
      best_tree.append([x[0], x[1], np.linalg.norm(self._coeff_dict[x[0]],2)**2/(sig_norm*2**len(x[0])), best_entropy[x[0]]])
    if threshold > 0:
      pass
      best_tree_t = []
      for leaf in best_tree:
        if leaf[2] >= threshold:
          best_tree_t.append(leaf)
      return best_tree_t
    else:
      return best_tree

  def feature_extraction(self, best_tree):
    feature_list = []
    selected_subbands = []
    for leaf in best_tree:
      feature_list.append(self._coeff_dict[leaf[0]])
      selected_subbands.append(leaf[0])
    feature_matrix = np.abs(np.array(feature_list))
    for i in range(feature_matrix.shape[0]):
      feature_matrix[i, :] = feature_matrix[i, :]/np.linalg.norm(feature_matrix[i, :], 1) * feature_matrix.shape[1]
    return 10 * np.log(feature_matrix + 1), selected_subbands

  def plot_best_basis(self):
    best_tree = self.best_basis()
    coeff = []
    p = []
    m = 0
    for c in best_tree:
      if len(c[0]) > m:
        m = len(c[0])
    # print(m)
    for c in best_tree:
      l = m - len(c[0]) + 1
      p.append(c[2])
      # print(l)
      for ell in range(l):
        coeff.append(np.abs(self._coeff_dict[c[0]]))
    coeff = np.array(coeff)
    print(best_tree)
    plt.plot(p, '*')
    plt.figure()
    # plt.show()
    plt.pcolor(coeff)
    plt.show()




if __name__ == '__main__':
  act = np.load('/home/kesmarag/Github/pscs-earthquakes/retreat_2019/act.npz')
  sim = np.load('/home/kesmarag/Github/pscs-earthquakes/retreat_2019/sim1.npz')
  x = act['arr_0']
  x = x[:,0]
  # x = np.diff(x)
  y = sim['arr_0']
  y = y[:,0]
  # y = np.diff(y)
  # dx = np.abs(np.squeeze(y))
  # plt.plot(dx)
  # plt.show()
  # exit(0)
  swpt = SWPT(max_level=7, wavelet='db4')
  swpt.decompose(x, 'shannon')
  tree = swpt.best_basis()
  print(tree)
  fm, sc = swpt.feature_extraction(tree, 0.025)
  plt.figure()
  plt.pcolor(fm)
  plt.colorbar()
  swpt.decompose(y, 'shannon')
  fm, sc = swpt.feature_extraction(tree, 0.025)
  plt.figure()
  plt.pcolor(fm)
  plt.colorbar()
  plt.show()
  # swpt.decompose(x, 'shannon')
  # wp4 = swpt.get_level(4)
  # print(swpt._entropy_dict)
  # best = swpt.best_basis()
  # swpt.plot_best_basis()
  # plt.plot(swpt.get_coefficient_vector('AAAAAD'))
  # plt.plot(swpt.get_coefficient_vector('AAAAAAD'))
  # plt.plot(swpt.get_coefficient_vector('AAAADDA'))
  # plt.show()
  # print(best)
