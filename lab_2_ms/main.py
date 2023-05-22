import math
import numpy as np
import matplotlib.pyplot as plt
import fitz
import pandas as pd
import scipy


variant = 45
exp_pdf = "distribution_data/MS_D_Exp.pdf"
norm_pdf = "distribution_data/MS_D_Norm.pdf"
uniform_pdf = "distribution_data/MS_D_Unif.pdf"


def read_array_pdf(v, filename, flag=False):
    with fitz.Document(filename) as doc:
        for page in doc.pages():
            if page.search_for(f"Вариант\n{v}"):
                page_text = page.get_text()
                page_text = page_text.replace(",", ".").split("\n")
                k = page_text.index(f"{v}") + 1
                if flag:
                    a = float(page_text[k].split(" ")[1])
                    b = float(page_text[k + 1].split(" ")[1])
                    k += 2
                    return a, b, list(map(float, page_text[k:k + 200]))
                return np.array(list(map(float, page_text[k:k + 200])))


def create_table(data, column_format, resize=False):
    table = data.to_latex(index=False,  column_format=column_format, index_names=False,
                          float_format=lambda x: '{:.5f}'.format(x).rstrip('0').rstrip('.')).\
        replace("\\\\", "\\\\ \n\\hline").\
        replace("\\toprule", "\\hline").\
        replace("\\midrule", "").\
        replace("\\bottomrule", "")
    if resize:
        table = "\\begin{table}[htbp]\n \\centering\n \\resizebox{\\linewidth}{!}{\n" + table + "}\n\\end{table}"
    return table + "\n"


def f_exp(a, lam):
    return lam * np.exp(-lam * a)


def big_f_exp(a, lam):
    return 1 - np.exp(-lam * a)


def func_a_k_uniform(x, a, b):
    return (x - a) / (b - a)


def task_3_4(uniform_dist, a, b):
    a_0, a_m = a, b
    bins = np.round(np.linspace(a_0, a_m, m + 1), 5)
    count_intervals = np.histogram(uniform_dist, bins=8)[0]
    freq_intervals = np.round(count_intervals / N, decimals=5)
    norm_dist_sort = np.sort(uniform_dist)

    data_1 = pd.DataFrame(np.reshape(uniform_dist, (20, 10)))
    data_2 = pd.DataFrame(np.reshape(norm_dist_sort, (20, 10)))

    bins_print = ['$[' + str(bins[0]) + ', ' + str(bins[1]) + ']$']
    for i in range(1, m):
        bins_print.append('$(' + str(bins[i]) + ', ' + str(bins[i + 1]) + ']$')
    data_3 = pd.DataFrame({
        'Интервалы': bins_print + [""],
        '$n_i$': np.append(count_intervals, np.sum(count_intervals)),
        '$w_i$': np.append(freq_intervals, np.sum(freq_intervals))
    })

    f_a_k = func_a_k_uniform(bins, a, b)
    p_k = f_a_k[1:] - f_a_k[:-1]
    data_4 = pd.DataFrame({
        "$k$": np.append(np.arange(m + 1), ""),
        "$a_k$": np.append(bins, ""),
        "$f(a_k)$": np.append(np.full(m + 1, round(1 / (b - a), 5)), ""),
        "$F(a_k)$": np.append(f_a_k, ""),
        "$p_k^*$": np.append(np.append("-", np.round(p_k, 5)), np.round(np.sum(p_k), 5))
    })

    mod_freq_p = np.abs(freq_intervals - p_k)
    chi_squared_stat = N * (freq_intervals - p_k) ** 2 / p_k
    data_5 = pd.DataFrame({
        '$k$': np.append(np.arange(1, m + 1), ""),
        'Интервал': bins_print + [""],
        '$w_k$': np.append(freq_intervals, np.sum(freq_intervals)),
        '$p_k^*$': np.append(p_k, np.sum(p_k)),
        '$|w_k - p_k^*|$': np.append(mod_freq_p, np.max(mod_freq_p)),
        '$\\frac{N(w_k - p_k^*)^2}{p_k^*}$': np.append(chi_squared_stat, np.sum(chi_squared_stat))
    })

    crit_val = scipy.stats.chi2.ppf(1 - alpha, m - 3)

    plt.hist(uniform_dist, bins=bins, density=True)
    x = np.linspace(a_0, a_m, 1000)
    y = np.ones_like(x) / (b-a)
    plt.plot(x, y)
    plt.savefig('task_3_4/hist_uniform.png')
    plt.close()

    x_1 = norm_dist_sort
    f_n = np.arange(1, N + 1) / N
    plt.step(x_1, f_n)

    f_x = func_a_k_uniform(norm_dist_sort, a, b)
    plt.plot(norm_dist_sort, f_x)
    plt.savefig('task_3_4/empirical_uniform.png')
    plt.close()

    arr = np.zeros(200)
    for i in range(1, N):
        arr[i] = max(abs(f_n[i] - f_x[i]), abs(f_n[i - 1] - f_x[i]))
    ind = np.argmax(arr)
    x_ = norm_dist_sort[ind]
    d_n = np.max(arr)
    f_x_ = f_x[ind]
    f_n_x_ = f_n[ind]
    f_n_x_1 = f_n[ind - 1]
    d_n_sqrt = d_n * N**0.5
    data_6 = pd.DataFrame({
        "$a$": a,
        "$b$": b,
        "$N$": N,
        "$D_N$": d_n,
        "$D_N\sqrt{N}$": d_n_sqrt,
        "$x^*$": x_,
        "$F(x^*)$": f_x_,
        "$F_N(x^*)$": f_n_x_,
        "$F_N(x^* - 0)$": f_n_x_1
    }, index=[0])
    crit_val_kolmogorov = scipy.stats.kstwobign.ppf(1 - alpha)

    table_1 = create_table(data_1, "|c" * 10 + "|", resize=True)
    table_2 = create_table(data_2, "|c" * 10 + "|", resize=True)
    table_3 = create_table(data_3, "|c|c|c|")
    table_4 = create_table(data_4, '|c' * 5 + '|')
    table_5 = create_table(data_5, '|c' * 6 + '|', resize=True)
    table_6 = create_table(data_6, '|c' * 9 + '|', resize=True)
    return table_1, table_2, table_3, table_4, table_5, table_6, crit_val, crit_val_kolmogorov


def task_2(norm_dist):
    a_0, a_m = np.min(norm_dist), np.max(norm_dist)
    bins = np.round(np.linspace(a_0, a_m, m + 1), 5)
    mean_interval = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])
    count_intervals = np.histogram(norm_dist, bins=8)[0]
    freq_intervals = np.round(count_intervals / N, decimals=5)
    norm_dist_sort = np.sort(norm_dist)

    data_1 = pd.DataFrame(np.reshape(norm_dist, (20, 10)))
    data_2 = pd.DataFrame(np.reshape(norm_dist_sort, (20, 10)))

    bins_print = ['$[' + str(bins[0]) + ', ' + str(bins[1]) + ']$']
    for i in range(1, m):
        bins_print.append('$(' + str(bins[i]) + ', ' + str(bins[i + 1]) + ']$')
    data_3 = pd.DataFrame({
        'Интервалы': bins_print + [""],
        '$n_i$': np.append(count_intervals, np.sum(count_intervals)),
        '$w_i$': np.append(freq_intervals, np.sum(freq_intervals))
    })

    mean = np.sum(mean_interval * freq_intervals)
    var = np.sum((mean_interval - mean)**2 * freq_intervals)
    sigma = var**0.5
    print("Показатели нормального распределения:\n", f"Мат ожидание {mean}\n", f"Дисперсия {var}\n", f"Сигма {sigma}")

    a_k = (bins - mean) / sigma
    density = 1 / sigma * (scipy.stats.norm.pdf(a_k, loc=0, scale=1))
    standard_norm = scipy.stats.norm.cdf(a_k, loc=0, scale=1)
    p_k = standard_norm[2:-1] - standard_norm[1:-2]
    p_k = np.append(p_k, 1 - standard_norm[-2])
    p_k = np.insert(p_k, 0, standard_norm[1])
    data_4 = pd.DataFrame({
        "$k$": np.append(np.arange(m + 1), ""),
        "$a_k$": np.append(bins, ""),
        "$\\frac{a_k - \widetilde{a}}{\widetilde{\sigma}}$": np.append(np.round(a_k, 5), ""),
        "$\\frac{1}{\widetilde{\sigma}} "
        "\\varphi\Bigg(\\frac{a_k-\widetilde{a}}{\widetilde{\sigma}}\Bigg)$": np.append(np.round(density, 5), ""),
        "$\Phi\Bigg(\\frac{a_k-\widetilde{a}}{\widetilde{\sigma}}\Bigg)$": np.append(np.round(standard_norm, 5), ""),
        "$p_k^*$": np.append(np.append("-", np.round(p_k, 5)), np.round(np.sum(p_k), 5))
    })

    mod_freq_p = np.abs(freq_intervals - p_k)
    chi_squared_stat = N * (freq_intervals - p_k) ** 2 / p_k
    data_5 = pd.DataFrame({
        '$k$': np.append(np.arange(1, m + 1), ""),
        'Интервал': bins_print + [""],
        '$w_k$': np.append(freq_intervals, np.sum(freq_intervals)),
        '$p_k^*$': np.append(p_k, np.sum(p_k)),
        '$|w_k - p_k^*|$': np.append(mod_freq_p, np.max(mod_freq_p)),
        '$\\frac{N(w_k - p_k^*)^2}{p_k^*}$': np.append(chi_squared_stat, np.sum(chi_squared_stat))
    })
    crit_val = scipy.stats.chi2.ppf(1 - alpha, m - 3)

    plt.hist(norm_dist, bins=bins, density=True)
    x = np.linspace(a_0, a_m, 1000)
    y = scipy.stats.norm.pdf(x, loc=mean, scale=var)
    plt.plot(x, y)
    plt.savefig('task_2/hist_norm.png')
    plt.close()

    table_1 = create_table(data_1, "|c" * 10 + "|", resize=True)
    table_2 = create_table(data_2, "|c" * 10 + "|", resize=True)
    table_3 = create_table(data_3, "|c|c|c|")
    table_4 = create_table(data_4, '|c' * 7 + '|', resize=True)
    table_5 = create_table(data_5, '|c' * 6 + '|', resize=True)
    return table_1, table_2, table_3, table_4, table_5, crit_val


def task_1_5(exp_dist):
    a_0, a_m = 0, np.max(exp_dist)
    bins = np.round(np.linspace(a_0, a_m, m + 1), 5)
    mean_interval = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])
    count_intervals = np.histogram(exp_dist, bins=8)[0]
    freq_intervals = np.round(count_intervals / N, decimals=5)
    exp_dist_sort = np.sort(exp_dist)

    data_1 = pd.DataFrame(np.reshape(exp_dist, (20, 10)))
    data_2 = pd.DataFrame(np.reshape(exp_dist_sort, (20, 10)))

    bins_print = ['$[' + str(bins[0]) + ', ' + str(bins[1]) + ']$']
    for i in range(1, m):
        bins_print.append('$(' + str(bins[i]) + ', ' + str(bins[i + 1]) + ']$')
    data_3 = pd.DataFrame({
        'Интервалы': bins_print + [""],
        '$n_i$': np.append(count_intervals, np.sum(count_intervals)),
        '$w_i$': np.append(freq_intervals, np.sum(freq_intervals))
    })

    first_moment = np.sum(mean_interval * freq_intervals)
    lamb_ = 1 / first_moment
    print("Лямбда показательного распределения: ", lamb_)
    f_lower = f_exp(bins, lamb_)
    f_upper = big_f_exp(bins, lamb_)
    p_k = f_upper[1:-1] - f_upper[:-2]
    p_k = np.append(p_k, 1 - f_upper[-2])
    data_4 = pd.DataFrame({
        "$k$": np.append(np.arange(0, m + 1), [""]),
        "$a_k$": np.append(np.round(bins, 5), [""]),
        "$f(a_k, \widetilde{\lambda})$": np.append(np.round(f_lower, 5), ""),
        "$F(a_k, \widetilde{\lambda})$": np.append(np.round(f_upper, 5), ""),
        "$p_k^*$": np.append(np.append(np.array(["$-$"]), np.round(p_k, 5)), np.round(np.sum(p_k), 5))
    })

    mod_freq_p = np.abs(freq_intervals - p_k)
    chi_squared_stat = N * (freq_intervals - p_k)**2 / p_k
    data_5 = pd.DataFrame({
        '$k$': np.append(np.arange(1, m + 1), ""),
        'Интервал': bins_print + [""],
        '$w_k$': np.append(freq_intervals, np.sum(freq_intervals)),
        '$p_k^*$': np.append(p_k, np.sum(p_k)),
        '$|w_k - p_k^*|$': np.append(mod_freq_p, np.max(mod_freq_p)),
        '$\\frac{N(w_k - p_k^*)^2}{p_k^*}$': np.append(chi_squared_stat, np.sum(chi_squared_stat))
    })
    crit_val = scipy.stats.chi2.ppf(1 - alpha, m - 2)

    plt.hist(exp_dist_sort, bins=bins, density=True)
    x = np.linspace(a_0, a_m, 1000)
    y = f_exp(x, lamb_)
    plt.plot(x, y)
    plt.savefig('task_1_5/hist_exp.png')
    plt.close()

    f_n = np.arange(1, N + 1) / N
    plt.step(exp_dist_sort, f_n)

    f_x = big_f_exp(exp_dist_sort, lamb)
    plt.plot(exp_dist_sort, f_x)
    plt.savefig('task_1_5/empirical_exp.png')
    plt.close()

    arr = np.zeros(200)
    for i in range(1, N):
        arr[i] = max(abs(f_n[i] - f_x[i]), abs(f_n[i - 1] - f_x[i]))
    ind = np.argmax(arr)
    x_ = exp_dist_sort[ind]
    d_n = np.max(arr)
    f_x_ = f_x[ind]
    f_n_x_ = f_n[ind]
    f_n_x_1 = f_n[ind - 1]
    d_n_sqrt = d_n * N ** 0.5
    data_6 = pd.DataFrame({
        "$a$": a_0,
        "$b$": a_m,
        "$N$": N,
        "$D_N$": d_n,
        "$D_N\sqrt{N}$": d_n_sqrt,
        "$x^*$": x_,
        "$F(x^*)$": f_x_,
        "$F_N(x^*)$": f_n_x_,
        "$F_N(x^* - 0)$": f_n_x_1
    }, index=[0])

    crit_val_kolmogorov = scipy.stats.kstwobign.ppf(1 - alpha)

    table_1 = create_table(data_1, "|c" * 10 + "|", resize=True)
    table_2 = create_table(data_2, "|c" * 10 + "|", resize=True)
    table_3 = create_table(data_3, "|c|c|c|")
    table_4 = create_table(data_4, '|c' * 5 + '|')
    table_5 = create_table(data_5, '|c' * 6 + '|', resize=True)
    table_6 = create_table(data_6, '|c' * 9 + '|', resize=True)
    return table_1, table_2, table_3, table_4, table_5, table_6, crit_val, crit_val_kolmogorov


if __name__ == "__main__":
    f_3 = open('task_3_4/tab_crit_kalm.txt', 'w')
    f_2 = open('task_2/tab_crit.txt', 'w')
    f_1 = open('task_1_5/tab_crit_kalm.txt', 'w')
    exp = read_array_pdf(variant, exp_pdf)
    norm = read_array_pdf(variant, norm_pdf)
    a_uniform, b_uniform, uniform = read_array_pdf(variant, uniform_pdf, flag=True)
    lamb = 1.275
    alpha = 0.05
    N = len(exp)
    m = 1 + math.floor(np.log2(N))
    for w in task_1_5(exp):
        f_1.write(str(w) + "\n\n\n")
    f_1.close()
    for w in task_2(norm):
        f_2.write(str(w) + "\n\n\n")
    f_2.close()
    for w in task_3_4(uniform, a_uniform, b_uniform):
        f_3.write(str(w) + "\n\n\n")
    f_3.close()
