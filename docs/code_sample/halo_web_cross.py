from pathlib import Path
import numpy as np
from pyhipp import plot

# 假设你的自定义类都在当前目录下
from halo_web import samples, statistics

# --- 1. 设置输入和输出文件路径 ---
# !! 修改为你自己的文件路径 !!
file_halo_sample = './halo_sample/halo_sample_tng.txt'
file_tidal_field = './tidal_field.hdf5'
dir_output = Path('./output_halo_web_cross/')

# --- 2. 设置计算参数 ---
lr_range, n_bins = [0.1, 1.2], 8
n_bootstrap = 100
mass_weighted = True
rng_seed = 10086

def main():
    # --- 3. 加载数据 ---
    print("Loading halo sample...")
    # 使用你的 HaloSample.from_file 方法加载数据
    # 注意：你的 from_file 方法目前只支持从 txt 读取 Quijote 格式
    # 我们需要先手动加载 txt，再创建 HaloSample
    data = np.loadtxt(file_halo_sample)
    halo_sample = samples.HaloSample(data={
        'mass': data[:, 0],
        'zf': data[:, 1],
        'x': data[:, 2],
        'y': data[:, 3],
        'z': data[:, 4],
        'vx': data[:, 5],
        'vy': data[:, 6],
        'vz': data[:, 7],
    })

    print("Loading tidal field...")
    tidal_field = samples.TidalField.from_file(file_tidal_field)
    
    # --- 4. 初始化 HaloWebCross 并计算关联函数 ---
    print("\nCalculating cross-correlation functions...")
    hwc = statistics.HaloWebCross(
        halo_sample, tidal_field,
        lr_range=lr_range, n_bins=n_bins, rng=rng_seed
    )
    
    # 计算关联函数
    ccfs_raw = hwc.corrs(mass_weighted=mass_weighted)['web_typed']
    
    # --- 5. 进行 Bootstrap 误差分析并保存结果 ---
    print(f"Running {n_bootstrap} bootstrap resamples...")
    web_types = ['void', 'sheet', 'filament', 'knot']
    out_2pccf = {}
    
    header = ('# Columns:\n'
              '# 1: log(r) [Mpc/h]\n'
              '# 2: xi(r) (median)\n'
              '# 3: 16th percentile\n'
              '# 4: 84th percentile')

    dir_output.mkdir(parents=True, exist_ok=True)
    
    for ccf_raw, web_type in zip(ccfs_raw, web_types):
        # bootstrap 返回一个 Summary 对象
        summary = hwc.bootstrap(ccf_raw, n_bootstrap)
        median, (p16, p84) = summary.median, summary.percentiles([16, 84])
        
        # 准备保存的数据
        output_data = np.column_stack([hwc.lrs, median, p16, p84])
        out_2pccf[web_type] = output_data
        
        # 保存到文件
        file_out = dir_output / f'halo_{web_type}_cross_corr.txt'
        np.savetxt(file_out, output_data, header=header, comments='')
        print(f"Saved result to {file_out}")

    # --- 6. 绘图 ---
    print("\nGenerating plot...")
    fig, axs = plot.subplots((2,2), sharex=True, sharey=True, 
                             space=.1, subsize=5.5, layout='none')
    axs_f = axs.flat

    for i, web_type in enumerate(web_types):
        lrs, med, p16, p84 = out_2pccf[web_type].T
        y_err = [med - p16, p84 - med] # 非对称误差棒
            
        ax = axs_f[i]
        ax.errorbar(lrs, med, yerr=y_err, fmt='-o', capsize=3, label=f'Halo-{web_type}')
        ax.axhline(0, color='grey', linestyle='--', lw=1) # 添加 y=0 参考线

    axs.leg(loc='best').label(r'$\log_{10}\,[r / (h^{-1}{\rm Mpc})]$', r'$\xi(r)$')
    fig.suptitle('Halo-Cosmic Web Cross-Correlation', fontsize=16)
    
    plot_file = dir_output / 'halo_web_cross.png'
    plot.savefig(plot_file)
    print(f"Saved plot to {plot_file}")

if __name__ == '__main__':
    main()