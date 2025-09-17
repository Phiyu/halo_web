from __future__ import annotations

import numpy as np
from pyhipp.core import DataDict
from pyhipp.stats.random import Rng
from pyhipp.stats.summary import Summary
from pyhipp.stats.binning import _BinnedData

# 假设你的 field.py 和 halo.py 文件与此脚本在同一目录下
# 否则需要调整导入路径
from ..samples.field import TidalField
from ..samples.halo import HaloSample

class HaloWebCross:
    """
    计算 Halo 样本与宇宙网 (TidalField) 之间的交叉关联函数。
    这个类是 `GalaxyWebCross` 的一个适配版本，专门用于处理 HaloSample 对象。
    """

    def __init__(self, 
                 h_samp: HaloSample,
                 tf: TidalField, *,
                 rng: Rng | int, 
                 lr_range: list = [-0.1, 1.2], 
                 n_bins: int = 8):
        
        # --- 关键修改部分：从 HaloSample 提取坐标 ---
        # 原始代码从 'x_sim_cor' 提取坐标，我们将其改为从 'x', 'y', 'z' 合并
        self.xs = np.column_stack([
            h_samp['x'], h_samp['y'], h_samp['z']
        ])
        
        # HaloSample 中没有 'weight' 字段，所以我们默认所有 halo 权重相等
        n_xs = h_samp.n_objs
        self.ws = np.ones(n_xs) / n_xs
        # --- 修改结束 ---

        self.tf = tf
        self.rng = Rng(rng)
        self.set_bin(lr_range, n_bins)
        
        print(f"HaloWebCross initialized with {h_samp.n_objs} halos.")

    def corrs(self, lam_off=0., mass_weighted=True):
        '''
        计算 halo 与四种宇宙网结构（void, sheet, filament, knot）的关联函数。
        
        @mass_weighted: 是否按场点的质量（密度）进行加权。
        '''
        tf = self.tf
        # 筛选出在重构区域内的场点
        recon_mask = tf.recon_mask
        lams = tf.lams[recon_mask]
        
        # 根据本征值数量对场点进行分类
        n_lam = (lams >= lam_off).sum(1)
        
        # 如果需要质量加权，则获取密度
        rho = tf.delta[recon_mask] + 1.
        
        corrs = []
        for i_web in range(4): # 0: void, 1: sheet, 2: filament, 3: knot
            # 为当前宇宙网类型创建权重
            wgts = (n_lam == i_web).astype(float)
            if mass_weighted:
                wgts *= rho
                
            corr = self._w_corr(wgts)
            corrs.append(corr)
            
        return DataDict({'web_typed': corrs})

    def _w_corr(self, wgts: np.ndarray):
        '''
        计算关联函数的核心函数。
        DD(r) / DR(r) - 1
        '''
        assert len(wgts) == len(self.tf.xs_recon)
        nn_ids, bin_ids = self.nn_ids, self.bin_ids
        b_data = self.b_data
        
        # 计算 DD(r): 在每个 halo 周围的球壳内，对场点权重求和
        tot_wgts = np.zeros_like(self.cnts)
        for i, (nn_id, bin_id) in enumerate(zip(nn_ids, bin_ids)):
            b_data.reset()
            b_data.add_n_chked(bin_id, wgts[nn_id])
            tot_wgts[i] = b_data.data

        # 计算 DR(r): 随机分布下期望的场点权重
        mesh = self.tf.mesh
        V_cell = mesh.l_grid**3
        rho_mean = wgts.sum() / (len(wgts) * V_cell)
        wgt_exp = rho_mean * self.dVs
        
        # 计算 xi(r)
        corr = tot_wgts / wgt_exp.clip(1.0e-6) - 1.
        return corr

    def bootstrap(self, outs: np.ndarray, n: int):
        """ 对单组结果进行 bootstrap 重采样以估计误差。"""
        ws = self.ws
        if n == 0:
            mean = (outs * ws[:, None]).sum(0)
            return Summary.on(np.array([mean]))

        means = np.zeros((n, outs.shape[1]))
        for i in range(n):
            # 根据 halo 的权重进行重采样
            ids = self.rng.choice(len(outs), len(outs), p=ws)
            means[i] = outs[ids].mean(0)
        return Summary.on(means)

    def set_bin(self, lr_range, n_bins):
        """ 初始化径向 binning 和近邻搜索。"""
        lr_min, lr_max = lr_range
        dlr = (lr_max - lr_min) / n_bins
        b_data = _BinnedData(n_bins + 1)

        kdt = self.tf.kdt_recon
        xs = self.xs
        # 对每个 halo，在最大半径内搜索所有场点邻居
        nn_ids, nn_ds = kdt.query_radius(xs, r=10**lr_max, return_distance=True)

        # 将每个邻居分配到对应的径向 bin
        bin_ids = []
        for nn_d in nn_ds:
            bin_id = np.floor((np.log10(nn_d.clip(1.0e-10)) - lr_min) / dlr).astype(int)
            bin_id += 1
            bin_id.clip(0, out=bin_id)
            bin_ids.append(bin_id)

        # 计算每个 halo 在每个 bin 中的邻居数量 (用于可能的归一化)
        cnts = np.zeros((len(xs), n_bins + 1), dtype=float)
        for i, bin_id in enumerate(bin_ids):
            b_data.reset()
            b_data.cnt_n_chked(bin_id)
            cnts[i] = b_data.data

        # 计算每个球壳的体积
        lr_es = np.linspace(lr_min, lr_max, n_bins + 1)
        lr_es = np.concatenate([[-10.], lr_es]) # 包含中心区域
        r_es = 10**lr_es
        V_es = 4.0 / 3.0 * np.pi * r_es**3
        dVs = np.diff(V_es)
        lrs = 0.5 * (lr_es[:-1] + lr_es[1:])
        lrs[0] = lr_min - dlr # 中心 bin 的位置

        self.lrs = lrs
        self.dVs = dVs
        self.b_data = b_data
        self.nn_ids = nn_ids
        self.bin_ids = bin_ids
        self.cnts = cnts