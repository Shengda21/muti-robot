"""Build the closure figure, cost table, and LaTeX macros from analyze_closure summaries."""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = [2, 3, 4, 6]
SCALES = [0.5, 1.0, 2.0, 4.0]
CN = False


def load(p): return json.loads((ROOT / p / 'summary.json').read_text())


def cell(m, s, D=3, ev='closed'): return f'm{m}|{ev}|D{D}|s{s}'


def figure(scaling, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.linewidth': .6})
    if CN: plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['SimHei'], 'axes.unicode_minus': False})
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.9))
    cols = ['#9ecae1', '#4292c6', '#2171b5', '#08306b']
    for i, s in enumerate(SCALES):
        xs = list(range(len(MS)))
        e = [scaling[cell(m, s)]['extra_eval_saving_pct'] for m in MS]
        ax[0].errorbar([x + (i - 1.5) * .06 for x in xs], [v[0] for v in e],
                       yerr=[[v[0] - v[1] for v in e], [v[2] - v[0] for v in e]], color=cols[i], marker='o', ms=2.5, lw=.9, capsize=1.2, elinewidth=.5,
                       label=f'$\\lambda={s:g}$')
        d = [scaling[cell(m, s)]['J_minus_Ctb_pctT'] for m in MS]
        ax[1].errorbar([x + (i - 1.5) * .06 for x in xs], [v[0] for v in d],
                       yerr=[[v[0] - v[1] for v in d], [v[2] - v[0] for v in d]], color=cols[i], marker='o', ms=2.5, lw=.9, capsize=1.2, elinewidth=.5)
    ax[1].axhline(0, color='k', lw=.5, ls=':')
    labs = (('(a) 额外跳过的评估', '相对名义筛选的比例 (%)'), ('(b) $J$ 搜索 vs. $C$ 搜索', '留出 $J/T$ 差值 (%)')) if CN else \
           (('(a) extra evaluations skipped', '% vs nominal screening'), ('(b) $J$ search vs. $C$ search', 'test $J/T$ difference (%)'))
    for a, (t, yl) in zip(ax, labs):
        a.set_xticks(range(len(MS))); a.set_xticklabels(MS); a.set_xlabel('机器人数 $m$' if CN else 'robots $m$'); a.set_title(t, fontsize=7); a.set_ylabel(yl)
        a.tick_params(width=.5, length=2); a.spines[['top', 'right']].set_visible(False)
    ax[0].legend(frameon=False, fontsize=5.5, handlelength=1.2, loc='upper right')
    fig.tight_layout(pad=.3, w_pad=.6)
    fig.savefig(out, bbox_inches='tight')


def cost_table(cost):
    T = {
        True: ('闭式，$D{=}3$', '闭式，$D{=}12$', '网格重放，$32\\times12$',
               '评估器成本与临界路径筛选相对名义筛选的CPU节省，$m=3$。$\\rho$为延迟评估的CPU时间除以无筛选时构建调度的CPU时间。',
               '评估器&$\\rho$&额外跳过&CPU节省（ms）'),
        False: ('Closed form, $D{=}3$', 'Closed form, $D{=}12$', 'Grid replay, $32\\times12$',
                'Evaluator cost and the CPU saving of critical-path over nominal screening, $m=3$. $\\rho$ is delay-evaluation CPU divided by schedule-building CPU without screening.',
                'Evaluator&$\\rho$&Extra skipped&CPU saved (ms)')}[CN]
    rows = [(T[0], 'm3|closed|D3|s1.0'), (T[1], 'm3|closed|D12|s1.0'), (T[2], 'm3|grid32|D12|s1.0')]
    L = ['\\begin{table}[t]', '\\centering', '\\caption{' + T[3] + '}',
         '\\label{tab:cost}', '\\setlength{\\tabcolsep}{2.4pt}\\footnotesize',
         '\\begin{tabular}{lrrr}\\toprule', T[4] + '\\\\\\midrule']
    for name, k in rows:
        r = cost[k]; e = r['extra_eval_saving_pct']; c = r['cpu_saving_nom_minus_crit_ms']
        L.append(f"{name}&{r['rho_eval_over_build']:.2f}&{e[0]:.1f}\\%&{num(c[0])} [{num(c[1])}, {num(c[2])}]\\\\")
    L += ['\\bottomrule\\end{tabular}\\end{table}']
    return '\n'.join(L)


def ci(t, d=2): return f'{t[0]:.{d}f}$\\,$[{t[1]:.{d}f}, {t[2]:.{d}f}]'


def num(x, d=2, sign=False):
    t = f'{x:+.{d}f}' if sign else f'{x:.{d}f}'
    return t.replace('-', '$-$').replace('+', '$+$')


def pm(t, d=2, unit=None):
    if unit is None: unit = '步' if CN else ' steps'
    iv = f'[{num(t[1], d)}, {num(t[2], d)}]'
    return num(t[0], d, True) + unit + (f'（区间{iv}）' if CN else f' (interval {iv})')


def macros(sc, co, ex):
    N = {2: 'Two', 3: 'Three', 4: 'Four', 6: 'Six'}
    L = []
    PCT = '\\%'
    def add(name, val): L.append('\\newcommand{\\' + name + '}{' + str(val) + '}')
    ntasks = len(list((ROOT / 'results/closure/confirm_scaling').glob('t*.json')))
    add('cfScenes', sc[cell(3, 1.0)]['scenes']); add('cfTasks', ntasks); add('cfRuns', f'{ntasks * 16 * 10:,}')
    for m in MS:
        add(f'cfExtra{N[m]}', f"{sc[cell(m, 1.0)]['extra_eval_saving_pct'][0]:.1f}" + PCT)
    for m in (2, 6):
        v = [sc[cell(m, s)]['extra_eval_saving_pct'][0] for s in SCALES]
        add(f'cfExtraRange{N[m]}', f'{min(v):.1f}--{max(v):.1f}' + PCT)
    P = sc[cell(3, 1.0)]; add('cfPrimary', pm(P['J_minus_Ctb_steps'])); add('cfPrimaryWeak', pm(P['J_minus_Crerank_steps']))
    add('cfPrimaryPct', num(-P['J_minus_Ctb_pctT'][0]))
    W = sc[cell(3, 4.0)]; add('cfLong', pm(W['J_minus_Ctb_steps'])); add('cfLongHolm', f"{W['J_minus_Ctb_steps_holm_p']:.3f}")
    Wp = sc[cell(3, 4.0)]['J_minus_Ctb_pctT']; add('cfLongPct', f'{-Wp[0]:.2f}')
    add('cfWeakLong', pm(sc[cell(3, 4.0)]['J_minus_Crerank_steps']))
    add('cfWeakSix', pm(sc[cell(6, 4.0)]['J_minus_Crerank_steps']))
    c = list(sc.values())
    cnt = lambda k, f: sum(f(v[k]) for v in c)
    add('cfNeg', cnt('J_minus_Ctb_steps', lambda t: t[0] < 0)); add('cfExcl', cnt('J_minus_Ctb_steps', lambda t: t[2] < 0))
    sec = [v for k, v in sc.items() if k != cell(3, 1.0)]
    add('cfHolm', sum(v['J_minus_Ctb_steps_holm_p'] < .05 for v in sec))
    add('cfNegW', cnt('J_minus_Crerank_steps', lambda t: t[0] < 0)); add('cfExclW', cnt('J_minus_Crerank_steps', lambda t: t[2] < 0))
    add('cfHolmW', sum(v['J_minus_Crerank_steps_holm_p'] < .05 for v in sec))
    add('cfTbGain', f"{-min(v['Ctb_minus_C_steps'][0] for v in c):.2f}")
    add('cfExploreNeg', sum(v['J_minus_Ctb_steps'][0] < 0 for v in ex.values())); add('cfExploreExcl', sum(v['J_minus_Ctb_steps'][2] < 0 for v in ex.values()))
    add('cfExplorePrimary', pm(ex[cell(3, 1.0)]['J_minus_Ctb_steps']))
    for k, nm in (('m3|closed|D3|s1.0', 'A'), ('m3|closed|D12|s1.0', 'B'), ('m3|grid32|D12|s1.0', 'C')):
        add(f'cfCpu{nm}', pm(co[k]['cpu_saving_nom_minus_crit_ms'], unit=' ms')); add(f'cfRho{nm}', f"{co[k]['rho_eval_over_build']:.2f}")
    add('cfCpuCv', num(co['m3|grid32|D12|s1.0']['cpu_saving_nom_minus_crit_ms'][0], 1))
    add('cfCpuNoneC', f"{co['m3|grid32|D12|s1.0']['cpu_saving_none_minus_crit_ms'][0]:.1f}")
    add('cfCpuBuildC', f"{co['m3|grid32|D12|s1.0']['J_none']['cpu_ms']:.1f}")
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scaling', required=True); ap.add_argument('--cost', required=True)
    ap.add_argument('--explore', required=True); ap.add_argument('--figure', default='paper/v2/figures/closure_effects.pdf')
    ap.add_argument('--tex', default='paper/v2/results_closure_table.tex')
    ap.add_argument('--macros', default='paper/v2/results_closure_numbers.tex')
    ap.add_argument('--lang', choices=['en', 'cn'], default='en')
    a = ap.parse_args()
    global CN
    CN = a.lang == 'cn'
    sc, co = load(a.scaling), load(a.cost)
    figure(sc, ROOT / a.figure)
    (ROOT / a.macros).write_text(macros(sc, co, load(a.explore)) + '\n', encoding='utf-8')
    (ROOT / a.tex).write_text(cost_table(co) + '\n', encoding='utf-8')
    print('ok')


if __name__ == '__main__':
    main()
