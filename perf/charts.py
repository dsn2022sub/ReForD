import os
import re

import matplotlib.pyplot as plt
import numpy as np

import perf


def get_metrics(log_file):
    _, _, f_1, history = perf.get_max(log_file)
    avg_acc = 0.0
    avg_prec = 0.0
    avg_recall = 0.0
    for item in history:
        acc_str, prec_str, recall_str, *_ = item.split('\t')
        acc = float(acc_str.split(':')[-1])
        prec = float(prec_str.split(':')[-1])
        recall = float(recall_str.split(':')[-1])
        size = len(history)
        avg_acc += acc / size
        avg_prec += prec / size
        avg_recall += recall / size
    return avg_acc, avg_prec, avg_recall, f_1


def visualize(dataset):
    # redundancy to speed up query
    perf_d_i_l = dict()
    perf_d_i_w = dict()
    perf_i_d_l = dict()
    perf_i_d_w = dict()
    depths = set()
    intervals = set()

    all_files = os.listdir('.')
    for file in all_files:
        if file.startswith(dataset):
            reg = re.compile(f'{dataset}-sketch-d-(?P<depth>\\d+)-i-(?P<interval>\\d+)-(?P<env>\\w+).log')
            reg_match_dict = reg.match(file).groupdict()

            depth = int(reg_match_dict['depth'])
            interval = int(reg_match_dict['interval'])
            env = reg_match_dict['env']
            depths.add(depth)
            intervals.add(interval)

            metrics = get_metrics(file)
            if env == 'l':
                perf_d_i_l.setdefault(depth, dict())
                perf_i_d_l.setdefault(interval, dict())
                perf_d_i_l[depth][interval] = metrics
                perf_i_d_l[interval][depth] = metrics
            elif env == 'w':
                perf_d_i_w.setdefault(depth, dict())
                perf_i_d_w.setdefault(interval, dict())
                perf_d_i_w[depth][interval] = metrics
                perf_i_d_w[interval][depth] = metrics

    # font config
    params = {
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        # 'font.style': 'italic',
        'font.weight': 'normal',
        'axes.titlesize': 'medium',
    }
    plt.rcParams.update(params)
    # figure config
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.4, hspace=1.0)
    axes = fig.subplots(2, 2)

    # relation between depth and metrics
    depths = sorted(list(depths))
    d_size = len(depths)

    # in Linux
    acc_l = [0] * d_size
    prec_l = [0] * d_size
    recall_l = [0] * d_size
    f_l = [0] * d_size
    # emphasize the best param
    best_d_idx = 0
    best_f = 0
    for idx, d in enumerate(depths):
        perf_i_l = perf_d_i_l[d]
        max_f = 0
        for i in intervals:
            acc, prec, recall, f = perf_i_l[i]
            if f > max_f:
                acc_l[idx] = acc
                prec_l[idx] = prec
                recall_l[idx] = recall
                f_l[idx] = f
                max_f = f
        if max_f > best_f:
            best_f = max_f
            best_d_idx = idx

    # use '*' to emphasize the param
    labels = [str(d) + '*' if idx == best_d_idx else d for idx, d in enumerate(depths)]
    bar_width = 0.2
    idx_acc = np.arange(len(labels)) * 1.2
    idx_prec = idx_acc + bar_width
    idx_recall = idx_prec + bar_width
    idx_f = idx_recall + bar_width

    # lines
    axes[0, 0].axhline(0.5, color='#C0C4CC', linewidth=1, zorder=0)
    axes[0, 0].axhline(1.0, color='#C0C4CC', linewidth=1, zorder=0)
    # bars
    axes[0, 0].bar(idx_acc, acc_l, width=bar_width, color='#4F81BD', label='Accuracy')
    axes[0, 0].bar(idx_prec, prec_l, width=bar_width, color='#C0504D', label='Precision')
    axes[0, 0].bar(idx_recall, recall_l, width=bar_width, color='#9BBB59', label='Recall')
    axes[0, 0].bar(idx_f, f_l, width=bar_width, color='#9F4C7C', label='F-Score')
    # metrics
    axes[0, 0].set_xticks(idx_acc + 1.5 * bar_width)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_xlabel('Depth')
    axes[0, 0].set_title('Linux')

    # in Windows
    acc_w = [0] * d_size
    prec_w = [0] * d_size
    recall_w = [0] * d_size
    f_w = [0] * d_size
    # emphasize the best param
    best_d_idx = 0
    best_f = 0
    for idx, d in enumerate(depths):
        perf_i_w = perf_d_i_w[d]
        max_f = 0
        for i in intervals:
            acc, prec, recall, f = perf_i_w[i]
            if f > max_f:
                acc_w[idx] = acc
                prec_w[idx] = prec
                recall_w[idx] = recall
                f_w[idx] = f
                max_f = f
        if max_f > best_f:
            best_f = max_f
            best_d_idx = idx

    # use '*' to emphasize the param
    labels = [str(d) + '*' if idx == best_d_idx else d for idx, d in enumerate(depths)]
    bar_width = 0.2
    idx_acc = np.arange(len(labels)) * 1.2
    idx_prec = idx_acc + bar_width
    idx_recall = idx_prec + bar_width
    idx_f = idx_recall + bar_width

    # lines
    axes[0, 1].axhline(0.5, color='#C0C4CC', linewidth=1, zorder=0)
    axes[0, 1].axhline(1.0, color='#C0C4CC', linewidth=1, zorder=0)
    # bars
    axes[0, 1].bar(idx_acc, acc_w, width=bar_width, color='#4F81BD', label='Accuracy')
    axes[0, 1].bar(idx_prec, prec_w, width=bar_width, color='#C0504D', label='Precision')
    axes[0, 1].bar(idx_recall, recall_w, width=bar_width, color='#9BBB59', label='Recall')
    axes[0, 1].bar(idx_f, f_w, width=bar_width, color='#9F4C7C', label='F-Score')
    # metrics
    axes[0, 1].set_xticks(idx_acc + 1.5 * bar_width)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_xlabel('Depth')
    axes[0, 1].set_title('Windows')

    # relation between depth and metrics
    intervals = sorted(list(intervals))
    i_size = len(intervals)

    # in Linux
    acc_l = [0] * i_size
    prec_l = [0] * i_size
    recall_l = [0] * i_size
    f_l = [0] * i_size
    # emphasize the best param
    best_i_idx = 0
    best_f = 0
    for idx, i in enumerate(intervals):
        perf_d_l = perf_i_d_l[i]
        max_f = 0
        for d in depths:
            acc, prec, recall, f = perf_d_l[d]
            if f > max_f:
                acc_l[idx] = acc
                prec_l[idx] = prec
                recall_l[idx] = recall
                f_l[idx] = f
                max_f = f
        if max_f > best_f:
            best_f = max_f
            best_i_idx = idx

    # use '*' to emphasize the param
    labels = [str(i) + '*' if idx == best_i_idx else i for idx, i in enumerate(intervals)]
    bar_width = 0.2
    idx_acc = np.arange(len(labels)) * 1.2
    idx_prec = idx_acc + bar_width
    idx_recall = idx_prec + bar_width
    idx_f = idx_recall + bar_width

    # lines
    axes[1, 0].axhline(0.5, color='#C0C4CC', linewidth=1, zorder=0)
    axes[1, 0].axhline(1.0, color='#C0C4CC', linewidth=1, zorder=0)
    # bars
    axes[1, 0].bar(idx_acc, acc_l, width=bar_width, color='#4F81BD', label='Accuracy')
    axes[1, 0].bar(idx_prec, prec_l, width=bar_width, color='#C0504D', label='Precision')
    axes[1, 0].bar(idx_recall, recall_l, width=bar_width, color='#9BBB59', label='Recall')
    axes[1, 0].bar(idx_f, f_l, width=bar_width, color='#9F4C7C', label='F-Score')
    # metrics
    axes[1, 0].set_xticks(idx_acc + 1.5 * bar_width)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_xlabel('Interval')
    axes[1, 0].set_title('Linux')

    # in Windows
    acc_w = [0] * i_size
    prec_w = [0] * i_size
    recall_w = [0] * i_size
    f_w = [0] * i_size
    # emphasize the best param
    best_i_idx = 0
    best_f = 0
    for idx, i in enumerate(intervals):
        perf_d_w = perf_i_d_w[i]
        max_f = 0
        for d in depths:
            acc, prec, recall, f = perf_d_w[d]
            if f > max_f:
                acc_w[idx] = acc
                prec_w[idx] = prec
                recall_w[idx] = recall
                f_w[idx] = f
                max_f = f
        if max_f > best_f:
            best_f = max_f
            best_i_idx = idx

    # use '*' to emphasize the param
    labels = [str(i) + '*' if idx == best_i_idx else i for idx, i in enumerate(intervals)]
    bar_width = 0.2
    idx_acc = np.arange(len(labels)) * 1.2
    idx_prec = idx_acc + bar_width
    idx_recall = idx_prec + bar_width
    idx_f = idx_recall + bar_width

    # lines
    axes[1, 1].axhline(0.5, color='#C0C4CC', linewidth=1, zorder=0)
    axes[1, 1].axhline(1.0, color='#C0C4CC', linewidth=1, zorder=0)
    # bars
    axes[1, 1].bar(idx_acc, acc_w, width=bar_width, color='#4F81BD', label='Accuracy')
    axes[1, 1].bar(idx_prec, prec_w, width=bar_width, color='#C0504D', label='Precision')
    axes[1, 1].bar(idx_recall, recall_w, width=bar_width, color='#9BBB59', label='Recall')
    axes[1, 1].bar(idx_f, f_w, width=bar_width, color='#9F4C7C', label='F-Score')
    # metrics
    axes[1, 1].set_xticks(idx_acc + 1.5 * bar_width)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_xlabel('Interval')
    axes[1, 1].set_title('Windows')

    # legend
    lines, labels = axes[-1, -1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='center')

    plt.show()


if __name__ == '__main__':
    visualize('wget')
    visualize('shellshock')
