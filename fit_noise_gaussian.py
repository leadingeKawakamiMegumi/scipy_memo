"""
ノイズ分布をガウス分布でフィッティングする関数

Created with Claude Cowork (Opus 4.7)_20260427
Verified and revised by Megumi

array1: 実測データ
array2: ノイズ除去後のデータ
noise = array2 - array1 として計算し、
ヒストグラムをガウス分布でフィットして中央値(mu)と標準偏差(sigma)を返す
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def gaussian(x, a, mu, sigma):
    """ガウス関数: a * exp(-(x - mu)^2 / (2 * sigma^2))"""
    return a * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def fit_noise_gaussian(array1, array2, bins=100, plot=False, verbose=True):
    """
    ノイズ分布をガウス分布でフィッティングする

    Parameters
    ----------
    array1 : array_like
        実測データ
    array2 : array_like
        ノイズ除去後のデータ
    bins : int, optional
        ヒストグラムのビン数（デフォルト: 100）
    plot : bool, optional
        フィッティング結果をプロットするかどうか（デフォルト: False）
    verbose : bool, optional
        結果を標準出力に表示するかどうか（デフォルト: True）

    Returns
    -------
    mu : float
        フィッティングで得られた中央値（ガウス分布の中心）
    sigma : float
        フィッティングで得られた標準偏差σ
    """
    # 入力をnumpy配列に変換
    array1 = np.asarray(array1, dtype=float).ravel()
    array2 = np.asarray(array2, dtype=float).ravel()

    if array1.shape != array2.shape:
        raise ValueError(
            f"array1とarray2の形状が一致しません: {array1.shape} vs {array2.shape}"
        )

    # ノイズの計算
    #noise = array2 - array1 
    noise = array1 - array2 #Modified by Megumi : Noise = Raw data - Denoised data


    # NaN/inf を除去
    noise = noise[np.isfinite(noise)]
    if noise.size == 0:
        raise ValueError("有効なノイズデータがありません")

    # ヒストグラムの作成
    counts, bin_edges = np.histogram(noise, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 初期値: ノイズの平均と標準偏差をそのまま使う
    a0 = counts.max() 
    mu0 = float(np.mean(noise))
    sigma0 = float(np.std(noise))
    if sigma0 == 0:
        sigma0 = 1e-6  # ゼロ割り防止

    # ガウシアンフィッティング
    try:
        popt, pcov = curve_fit(
            gaussian,
            bin_centers,
            counts,
            p0=[a0, mu0, sigma0],
        )
        a_fit, mu_fit, sigma_fit = popt
        sigma_fit = abs(sigma_fit)  # σは正の値
    except RuntimeError as e:
        raise RuntimeError(f"ガウシアンフィッティングに失敗しました: {e}")

    if verbose:
        print(f"フィッティング結果:")
        print(f"  中央値 (mu)    = {mu_fit:.6g}")
        print(f"  標準偏差 (σ)   = {sigma_fit:.6g}")
        print(f"  振幅 (a)       = {a_fit:.6g}")
        print(f"  サンプル数     = {noise.size}")

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(noise, bins=bins, alpha=0.6, label="noise histogram")
        x_smooth = np.linspace(bin_edges[0], bin_edges[-1], 500)
        ax.plot(
            x_smooth,
            gaussian(x_smooth, *popt),
            "r-",
            linewidth=2,
            label=f"Gaussian fit\nμ={mu_fit:.4g}, σ={sigma_fit:.4g}",
        )
        ax.set_xlabel("noise (array2 - array1)")
        ax.set_ylabel("count")
        ax.set_title("Noise distribution & Gaussian fit")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return mu_fit, sigma_fit


# 使用例
if __name__ == "__main__":
    # サンプルデータの生成
    rng = np.random.default_rng(0)
    n = 10000
    true_signal = np.sin(np.linspace(0, 4 * np.pi, n))
    noise_true = rng.normal(loc=0.05, scale=0.2, size=n)  # μ=0.05, σ=0.2
    array1 = true_signal + noise_true   # 実測データ
    array2 = true_signal                # ノイズ除去後のデータ

    mu, sigma = fit_noise_gaussian(array1, array2, bins=80, plot=True)
    print(f"\n戻り値: mu={mu}, sigma={sigma}")


#　補足
#first bin is [1, 2) (including 1, but excluding 2) and the second [2, 3). The last bin, however, is [3, 4], which includes 4.
#https://numpy.org/devdocs/reference/generated/numpy.histogram.html