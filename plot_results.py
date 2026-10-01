"""Draw every result figure from the metrics written by the other scripts.

    python plot_results.py

Figures whose input metrics are missing are skipped.
"""
import matplotlib.pyplot as plt
import numpy as np

from common import DISPLAY_NAMES, FIGURES, METRICS, read_json

FOUR = ['bert', 'religion-bert', 'mbert', 'multi-religion-bert']
COLORS = {'bert': '#4C72B0', 'religion-bert': '#DD8452', 'mbert': '#55A868',
          'multi-religion-bert': '#C44E52', 'xlmr': '#8172B2'}

# (label, task, key, scale to 0-1)
SUMMARY = [
    ('Sim Pearson', 'sim', 'pearson', 1), ('Sim Spearman', 'sim', 'spearman', 1),
    ('Cls Accuracy', 'cls', 'accuracy', 1), ('Cls Macro F1', 'cls', 'macro_f1', 1),
    ('QA Exact Match', 'qa', 'exact_match', 100), ('QA Token F1', 'qa', 'f1', 100),
]
# (figure file, title, task, [(panel title, key, label format)], zoom y-axis)
TASK_FIGURES = [
    ('fig4_1_semantic_similarity.png', 'Semantic Similarity Performance', 'sim',
     [('Pearson Correlation', 'pearson', '{:.4f}'), ('Spearman Correlation', 'spearman', '{:.4f}')], True),
    ('fig4_2_classification.png', 'Book Classification Performance', 'cls',
     [('Accuracy', 'accuracy', '{:.4f}'), ('Macro F1', 'macro_f1', '{:.4f}')], False),
    ('fig4_3_qa.png', 'Question Answering Performance', 'qa',
     [('Exact Match (%)', 'exact_match', '{:.2f}%'), ('Token F1 (%)', 'f1', '{:.2f}%')], False),
]

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 150})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def style_axis(ax):
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)


def bar_panel(ax, variants, values, title, fmt, zoom=False, width=0.5):
    bars = ax.bar([DISPLAY_NAMES[v] for v in variants], values, color=[COLORS[v] for v in variants],
                  width=width, edgecolor='white')
    ax.bar_label(bars, labels=[fmt.format(v) for v in values], fontsize=9, padding=2)
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.set_ylim(*((min(values) * 0.97, max(values) * 1.03) if zoom else (0, max(values) * 1.15)))
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right', fontsize=9)
    style_axis(ax)


def grouped_bars(ax, groups, series, values, colors, labels):
    """values[series][group] as side-by-side bars, one cluster per group."""
    x, width = np.arange(len(groups)), 0.8 / len(series)
    for i, s in enumerate(series):
        offset = (i - len(series) / 2 + 0.5) * width
        ax.bar(x + offset, [values[s][g] for g in groups], width, label=labels[s], color=colors[s],
               edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    style_axis(ax)


def benchmark_figures():
    results = {v: {t: read_json(METRICS / f'{v}_{t}.json') for t in ['sim', 'cls', 'qa']} for v in FOUR}

    for filename, title, task, panels, zoom in TASK_FIGURES:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title, fontsize=13, fontweight='bold')
        for ax, (panel_title, key, fmt) in zip(axes, panels):
            bar_panel(ax, FOUR, [results[v][task][key] for v in FOUR], panel_title, fmt, zoom)
        fig.tight_layout()
        save(fig, filename)

    normalised = {v: {label: results[v][task][key] / scale for label, task, key, scale in SUMMARY} for v in FOUR}
    labels = [s[0] for s in SUMMARY]

    fig, ax = plt.subplots(figsize=(14, 6))
    grouped_bars(ax, labels, FOUR, normalised, COLORS, DISPLAY_NAMES)
    ax.set_ylabel('Score (normalised 0-1)')
    ax.set_ylim(0, 1)
    ax.set_title('Model Comparison Across All Tasks', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    save(fig, 'fig4_4_all_tasks_combined.png')

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'polar': True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    for v in FOUR:
        vals = [normalised[v][label] for label in labels]
        ax.plot(angles + angles[:1], vals + vals[:1], 'o-', linewidth=2, label=DISPLAY_NAMES[v], color=COLORS[v])
        ax.fill(angles + angles[:1], vals + vals[:1], alpha=0.08, color=COLORS[v])
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    ax.set_title('Model Performance Across All Metrics', fontweight='bold', y=1.08)
    save(fig, 'fig4_5_radar.png')

    comparisons = [('religion-bert', 'bert'), ('multi-religion-bert', 'mbert')]
    gains = np.array([[normalised[a][label] - normalised[b][label] for label in labels] for a, b in comparisons])
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(gains, cmap='RdYlGn', aspect='auto', vmin=-0.05, vmax=0.10)
    plt.colorbar(im, ax=ax, label='Improvement', shrink=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_yticks(range(len(comparisons)))
    ax.set_yticklabels([f'{DISPLAY_NAMES[a]}\nvs {DISPLAY_NAMES[b]}' for a, b in comparisons])
    for (i, j), val in np.ndenumerate(gains):
        ax.text(j, i, f'{val:+.4f}', ha='center', va='center', fontsize=9)
    ax.set_title('Domain-adapted Model Improvement over Baselines', fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig4_6_improvement_heatmap.png')


def pretraining_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Pre-training Validation Loss', fontsize=13, fontweight='bold')
    for ax, corpus, variant, title in [(axes[0], 'english', 'religion-bert', 'ReligionBERT (English Bible)'),
                                       (axes[1], 'multilingual', 'multi-religion-bert', 'MultiReligionBERT (Multilingual Bible)')]:
        history = read_json(METRICS / f'pretrain_{corpus}.json')
        points = [(e['step'], e['eval_loss']) for e in history if 'eval_loss' in e]
        ax.plot(*zip(*points), 'o-', color=COLORS[variant], linewidth=2, markersize=4)
        ax.set_title(title)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Validation Loss (MLM)')
        style_axis(ax)
    fig.tight_layout()
    save(fig, 'fig4_7_pretraining_loss_curves.png')


def perplexity_figure():
    results = read_json(METRICS / 'perplexity.json')
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_panel(ax, list(results), list(results.values()), 'Perplexity (lower is better)', '{:.2f}', width=0.4)
    save(fig, 'fig4_8_perplexity.png')


def crosslingual_figures():
    results = read_json(METRICS / 'crosslingual.json')
    variants = list(results)
    languages = list(results[variants[0]])
    metrics = [('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1')]
    values = {m: {v: {lang: results[v][lang][m] for lang in languages} for v in variants} for m, _ in metrics}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Zero-shot Cross-lingual Classification', fontsize=13, fontweight='bold')
    for ax, (m, title) in zip(axes, metrics):
        grouped_bars(ax, languages, variants, values[m], COLORS, DISPLAY_NAMES)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, 'fig4_9_crosslingual_grouped.png')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Cross-lingual Performance Heatmap', fontsize=13, fontweight='bold')
    for ax, (m, title) in zip(axes, metrics):
        grid = np.array([[values[m][v][lang] for lang in languages] for v in variants])
        im = ax.imshow(grid, cmap='YlOrRd', aspect='auto', vmin=0, vmax=grid.max() * 1.1)
        plt.colorbar(im, ax=ax, shrink=0.8, label=title)
        ax.set_xticks(range(len(languages)))
        ax.set_xticklabels(languages, rotation=15, ha='right')
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels([DISPLAY_NAMES[v] for v in variants])
        ax.set_title(title)
        for (i, j), val in np.ndenumerate(grid):
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9)
    fig.tight_layout()
    save(fig, 'fig4_10_crosslingual_heatmap.png')

    acc = values['accuracy']
    gain = [acc['multi-religion-bert'][lang] - acc['mbert'][lang] for lang in languages]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(languages, gain, color=['#2ecc71' if g >= 0 else '#e74c3c' for g in gain], width=0.5, edgecolor='white')
    ax.bar_label(bars, labels=[f'{g:+.4f}' for g in gain], padding=2)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Accuracy Improvement')
    ax.set_title('MultiReligionBERT Gain over mBERT (Accuracy)', fontweight='bold')
    style_axis(ax)
    save(fig, 'fig4_11_crosslingual_improvement.png')


def main():
    for name, draw in [('benchmark', benchmark_figures), ('pretraining', pretraining_curves),
                       ('perplexity', perplexity_figure), ('crosslingual', crosslingual_figures)]:
        try:
            draw()
            print(f'Drew {name} figures')
        except FileNotFoundError as e:
            print(f'Skipping {name} figures, missing {e.filename}')


if __name__ == '__main__':
    main()
