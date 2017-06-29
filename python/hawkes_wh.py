
import numpy as np
import pandas as pd
from numba import jit


@jit(nopython=True, nogil=True)
def _increment_counters(steps, idx_i, idx_j, co_counters, counters, begin_time, last_time):
    """increment counters"""
    num_steps = len(steps)
    begin_idx = 0
    end_idx = 0
    n_i = len(idx_i)
    for t_j in idx_j:
        if begin_idx < n_i:
            begin_idx += np.searchsorted(idx_i[begin_idx:], t_j + steps[0])
        if end_idx < n_i:
            end_idx += np.searchsorted(idx_i[end_idx:], t_j + steps[-1] + 1)

        if begin_idx == end_idx:
            if begin_idx == 0:
                end_idx = 1
            else:
                begin_idx -= 1

        steps_j = steps + t_j
        idx = 0
        for t_i in idx_i[begin_idx:end_idx]:
            idx += np.searchsorted(steps_j[idx:], t_i)
            co_counters[idx] += 1
            if idx == num_steps:
                break

        first_idx = np.searchsorted(steps_j, begin_time)
        last_idx = np.searchsorted(steps_j, last_time)
        counters[first_idx:last_idx + 1] += 1


class HawkesWh:
    """Wiener Hopf for Hawkes"""

    def __init__(self,
                 key="key",
                 incremental_law=False,
                 shrink=True,
                 time_unit=1000000000):
        """init"""
        self.key = key
        self.shrink = shrink
        self.incremental_law = incremental_law
        self.time_unit = time_unit

        self.num_steps = 0
        self.zero_pos = 0
        self.steps = np.array([], dtype="<i8")
        self.step_sizes = pd.Series([], dtype="<i8")
        self.min_step_size = 0

        self.num_bins = 0
        self.bins = np.array([], dtype="<i8")
        self.bin_widths = np.array([], dtype="<i8")
        self.bins_end = 0
        self.bin_pos = np.array([], dtype="<f8")

        self.num_features = 0
        self.features = []

        self.lambdas = np.array([], dtype="<f8")
        self.co_counters = []
        self.counters = []
        self.cond_laws = []

        self.cond_law_mat = np.array([], dtype="<f8")
        self.cond_law_vec = np.array([], dtype="<f8")
        self.kernels = np.array([], dtype="<f8")
        self.cum_kernels = np.array([], dtype="<f8")
        self.kernel_norms = np.array([], dtype="<f8")

        self.mus = np.array([], dtype="<f8")
        self.dressed_norms = np.array([], dtype="<f8")
        self.dressed_norms_bar = np.array([], dtype="<f8")


    def set_steps(self, num_steps, step_size, coeff):
        """init steps"""
        self.min_step_size = step_size
        deltas = np.logspace(0, coeff, num_steps + 1).round().astype("<i8")
        steps = (np.cumsum(deltas) - 1) * step_size
        self.steps = np.hstack([-steps[::-1], steps[1:]])
        self.num_steps = len(self.steps)
        self.zero_pos = num_steps


    def get_grid(self, max_idx):
        """get grid"""
        begin_time = (1 - max_idx) * self.min_step_size
        last_time = max_idx * self.min_step_size
        end_idx = np.searchsorted(self.steps, last_time)
        grid = np.arange(begin_time, self.steps[end_idx] + 1, self.min_step_size)
        return grid


    def set_bins(self, num_bins, coeff):
        """init bins"""
        deltas = np.logspace(0, coeff, num_bins + 1).round().astype("<i8")
        bins = np.cumsum(deltas) - 1
        self.bin_widths = np.diff(bins)
        self.bins = bins[:-1]
        self.num_bins = num_bins
        self.bins_end = bins[-1]
        self.bin_pos = self.bins + self.bin_widths / 2
        self.bin_pos *= self.min_step_size / self.time_unit


    def init_counters(self, features):
        """init counters"""
        self.num_features = len(features)
        self.features = features.copy()

        self.lambdas = np.zeros(self.num_features, dtype="<f8")
        self.co_counters = []
        self.counters = []
        self.cond_laws = []
        zeros = np.zeros(self.num_steps + 1, dtype="<i8")
        zeros_series = pd.Series(zeros[1:-1], index=self.steps[1:], dtype="<f8")
        for i in range(self.num_features):
            self.co_counters.append([])
            self.counters.append([])
            self.cond_laws.append([])
            for j in range(self.num_features):
                self.co_counters[i].append(zeros.copy())
                self.counters[i].append(zeros.copy())
                self.cond_laws[i].append(zeros_series.copy())

        self.step_sizes = pd.Series(np.diff(self.steps) / self.time_unit, index=self.steps[1:])


    def init(self, num_steps, step_size, coeff, features):
        """global init"""
        self.set_steps(num_steps, step_size, coeff)
        self.init_counters(features)


    def get_conditional_law(self, co_counters, counters, counter_j, lambda_i, is_diagonal):
        """get conditional law"""
        if self.shrink:
            cond_law = (co_counters / counter_j) / self.step_sizes
            cond_law -= lambda_i * (counters / counter_j)
        else:
            cond_law = (co_counters / counters) / self.step_sizes
            cond_law.fillna(0, inplace=True)
            cond_law -= lambda_i

        if is_diagonal:
            cond_law[0] -= 1 / self.step_sizes[0]

        return cond_law


    def increment_counters(self, data):
        """increment counters"""
        idx = data.index.values.astype("<i8")
        tot_length = (idx[-1] - idx[0]) / self.time_unit

        for i in range(self.num_features):
            idx_i = idx[data[self.key] == self.features[i]].copy()
            n_i = len(idx_i)
            if n_i == 0:
                continue

            if self.incremental_law:
                lambda_i = n_i / tot_length
                self.lambdas[i] += lambda_i
            else:
                self.lambdas[i] += n_i

            for j in range(self.num_features):
                idx_j = idx[data[self.key] == self.features[j]].copy()
                n_j = len(idx_j)
                if n_j == 0:
                    continue

                _increment_counters(self.steps,
                                    idx_i,
                                    idx_j,
                                    self.co_counters[i][j],
                                    self.counters[i][j],
                                    idx[0],
                                    idx[-1])

                if self.incremental_law:
                    cond_law = self.get_conditional_law(
                        self.co_counters[i][j][1:-1],
                        self.counters[i][j][1:-1],
                        n_j,
                        lambda_i,
                        i == j)

                    self.cond_laws[i][j] += cond_law
                    self.co_counters[i][j][:] = 0
                    self.counters[i][j][:] = 0

        if self.incremental_law:
            return 1
        else:
            return tot_length


    def compute_conditional_laws(self, data_iter, set_features, progress=None):
        """Compute conditional laws"""
        num = 0
        for data in data_iter:
            data.sort_index(inplace=True)
            hours = data.index.to_datetime().hour
            data = data[(hours > 8) & (hours < 17)]

            set_features(data, self.key)

            num += self.increment_counters(data)

            if progress is not None:
                progress.value += 1

        for i in range(self.num_features):
            self.lambdas[i] /= num

            for j in range(self.num_features):
                if self.incremental_law:
                    self.cond_laws[i][j] /= num
                else:
                    self.cond_laws[i][j] = self.get_conditional_law(
                        self.co_counters[i][j][1:-1],
                        self.counters[i][j][1:-1],
                        self.counters[j][j][self.zero_pos],
                        self.lambdas[i],
                        i == j)


    def sym_cond_law(self):
        """symmetrize conditional laws"""
        cond_laws = []
        for i in range(self.num_features):
            cond_laws.append([])
            for j in range(self.num_features):
                cond_laws[i].append(self.cond_laws[i][j].copy())

                if self.incremental_law:
                    factors = .5
                else:
                    counters_ij = self.counters[i][j][1:-1]
                    counters_ji = self.counters[j][i][-2:0:-1]
                    factors = counters_ij / (counters_ij + counters_ji)

                cond_laws[i][j] *= factors

                if self.lambdas[j] > 0:
                    factors = (1 - factors) * self.lambdas[i] / self.lambdas[j]
                else:
                    factors = 1 - factors
                cond_laws[i][j] += self.cond_laws[j][i].values[::-1] * factors
        self.cond_laws = cond_laws


    def fill_cond_law_matrices(self, smooth=True):
        """compute K"""

        dim = self.num_features * self.num_bins
        self.cond_law_mat = np.zeros((dim, dim))
        self.cond_law_vec = np.zeros((dim, self.num_features))

        grid = self.get_grid(self.bins_end)

        for j in range(self.num_features):
            lines = slice(self.num_bins * j, self.num_bins * (j + 1))
            for i in range(self.num_features):
                cols = slice(self.num_bins * i, self.num_bins * (i + 1))
                slice_k = self.cond_law_mat[lines, cols]

                values = self.cond_laws[i][j].reindex(grid).bfill().values[:2 * self.bins_end]

                self.cond_law_vec[lines, i] = np.add.reduceat(values[self.bins_end:],
                                                              self.bins) / self.bin_widths

                if smooth:
                    values[self.bins_end + 1:] += values[self.bins_end:-1].copy()
                    values[self.bins_end + 1:] *= .5

                    values[1:self.bins_end] += values[:self.bins_end - 1].copy()
                    values[1:self.bins_end] *= .5
                else:
                    values[self.bins_end + 1:] = values[self.bins_end:-1].copy()

                values = values[:0:-1] * (self.min_step_size / self.time_unit)

                for t in range(self.num_bins):
                    begin_idx = self.bins_end - 1 - self.bins[t] - self.bin_widths[t] // 2
                    end_idx = begin_idx + self.bins_end
                    slice_k[t] = np.add.reduceat(values[begin_idx:end_idx], self.bins)

                    if i == j:
                        slice_k[t, t] += 1


    def compute_kernels(self):
        """compute kernels"""
        kernels = np.linalg.solve(self.cond_law_mat, self.cond_law_vec)
        kernels = kernels.reshape(self.num_features, self.num_bins, self.num_features)
        self.kernels = kernels.transpose([2, 1, 0])

        weights = self.bin_widths * self.min_step_size / self.time_unit
        weighted_kernels = np.multiply(self.kernels, weights.reshape(1, self.num_bins, 1))
        self.cum_kernels = np.cumsum(weighted_kernels, axis=1)


    def compute_norms(self):
        """compute statistics"""
        self.kernel_norms = self.cum_kernels[:, -1, :]

        identity = np.eye(self.num_features)
        id_phi = identity - self.kernel_norms
        self.mus = np.dot(id_phi, self.lambdas)
        self.dressed_norms = np.dot(self.kernel_norms, np.linalg.inv(id_phi))
        self.dressed_norms_bar = np.divide(
            np.multiply(self.dressed_norms, self.mus),
            self.lambdas.reshape(self.num_features, 1))


    def plot_kernels(self, plt, layout, labels, logscale=False, cumulated=False):
        """plot kernels"""
        dim1 = len(layout)
        dim2 = len(layout[0])
        axes = plt.subplots(dim1, dim2, sharex=True, figsize=(14, 8))[1]

        if cumulated:
            phi = self.cum_kernels
        else:
            phi = self.kernels

        for k in range(dim1):
            for l in range(dim2):
                for idx in layout[l][k]:
                    i = idx[0]
                    j = idx[1]
                    axes[k, l].plot(self.bin_pos, phi[i, :, j],
                                    label=labels[i] + " <- " + labels[j])
                    axes[k, l].legend()
                    axes[k, l].grid(True)
                    if logscale:
                        axes[k, l].set_xscale("log")
