"""Rebuild submission Tables 2-5 and Figures 2-3 from released aggregates.

Table 1 and Figure 1 describe the study and pipeline, rather than numerical results.
Run without --figures for a standard-library-only aggregate check.
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    with (ROOT / 'results' / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def mean(rows, key):
    return st.mean(float(r[key]) for r in rows)

def one(rows, **where):
    matches = [r for r in rows if all(r[k] == v for k, v in where.items())]
    if len(matches) != 1:
        raise ValueError((where, len(matches)))
    return matches[0]

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--figures', action='store_true', help='Requires matplotlib')
    args = parser.parse_args()
    out = ROOT / 'build/submission'
    out.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, actual, expected, digits):
        ok = abs(actual - expected) <= 0.50001 * 10 ** (-digits)
        checks.append(dict(check=name, actual=actual, expected=expected, passed=ok))
        if not ok:
            raise ValueError(checks[-1])

    def save(name, rows):
        with (out / name).open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    support = read('primary/support.csv')
    sources = sorted({r['source'] for r in support})
    table2 = []
    for source in sources:
        r = one(support, source=source, endpoint='soh30')
        row = dict(source=source, cells=int(r['total_cells']), origins=int(r['total_windows']))
        for target in ('rul90', 'rul80'):
            events = int(one(support, source=source, endpoint=target)['evaluable_cells'])
            row[target + '_events'] = events
            row[target + '_censored_percent'] = 100 * (1 - events / row['cells'])
        table2.append(row)
    check('total_cells', sum(r['cells'] for r in table2), 781, 0)
    check('total_origins', sum(r['origins'] for r in table2), 607090, 0)
    total = dict(source='Total',cells=sum(r['cells'] for r in table2),origins=sum(r['origins'] for r in table2))
    for target in ('rul90','rul80'):
        total[target+'_events'] = sum(r[target+'_events'] for r in table2)
        total[target+'_censored_percent'] = 100*(1-total[target+'_events']/total['cells'])
    table2.append(total)
    save('table2.csv', table2)

    agg = read('primary/aggregate_metrics.csv')
    table3 = []
    for display in ('TCN','GRU','Ridge','HGB','Extra Trees','Trend','Persistence or Median','Mean or Log-mean'):
        row = dict(system=display)
        for target in ('soh30','rul90','rul80'):
            model = ('Persistence' if target == 'soh30' else 'Median') if display == 'Persistence or Median' else ('Mean' if display == 'Mean or Log-mean' else display)
            r = one(agg, system=model, endpoint=target)
            row[target] = float(r['primary'])
            row[target + '_seed_sd'] = r['primary_seed_sd']
            if target != 'rul80':
                ref = one(agg, system='Persistence' if target == 'soh30' else 'Mean', endpoint=target)
                row[target + '_skill_percent'] = 100 * (1 - row[target] / float(ref['primary']))
        table3.append(row)
    check('Extra Trees RUL90', float(one(agg,system='Extra Trees',endpoint='rul90')['primary']), .6247, 4)
    save('table3.csv', table3)

    classical = read('submission/selected_classical_by_source.csv')
    cov = read('coverage/source_balanced_summary.csv')
    hcp = read('hcp/source_balanced_summary.csv')
    table4, fig3 = [], []
    for target in ('soh30','rul90','rul80'):
        c = [r for r in classical if r['target'] == target and r['status'] == 'PASS']
        if len(c) != (11 if target == 'rul90' else 12):
            raise ValueError('Unexpected selected classical support')
        table4.append(dict(target=target, method='Selected classical, group max', width=mean(c,'mean_interval_width')))
        fig3.append(dict(target=target,method='Selected classical, group max',pointwise=100*mean(c,'marginal_coverage'),all_history=100*mean(c,'family_level_simultaneous_coverage')))
        for recipe, label, data in [('family_max','TCN, group max',cov),('window','TCN, per origin',cov),('hcp_family_window','TCN, HCP',hcp)]:
            r = one(data, model='TCN.2026',target=target,recipe=recipe,subgroup='all')
            fig3.append(dict(target=target,method=label,pointwise=100*float(r['marginal_coverage']),all_history=100*float(r['family_simultaneous_coverage'])))
            if recipe != 'hcp_family_window':
                table4.append(dict(target=target,method=label,width=float(r['mean_interval_width'])))
    for target, expected in [('soh30',.1685),('rul90',436.92),('rul80',741.10)]:
        check('classical width ' + target, next(r['width'] for r in table4 if r['target']==target and r['method'].startswith('Selected')),expected,4 if target=='soh30' else 2)
    save('table4.csv', table4)
    save('figure3.csv', fig3)
    for r in read('diagnostics/selected_interval_summary.csv'):
        if r['system'] == 'Selected classical':
            actual=one(fig3,target=r['target'],method='Selected classical, group max')
            check('classical pointwise '+r['target'],actual['pointwise'],float(r['pointwise_coverage_percent']),2)
            check('classical all-history '+r['target'],actual['all_history'],float(r['all_group_coverage_percent']),2)
    fig2 = [r for r in read('primary/source_summary.csv') if r['system'] in ('TCN','GRU','Extra Trees','Trend') and r['endpoint'] in ('soh30','rul90')]
    save('figure2.csv', fig2)

    sensitivity = read('censoring_and_snl/aggregate_sensitivity.csv')
    table5 = []
    for scenario in ('exclude_four_SNL_proxy_cells','exclude_all_SNL'):
        r = one(sensitivity,scenario=scenario,system='Extra Trees',endpoint='rul90')
        table5.append(dict(check=scenario,scope='Extra Trees rul90',reference=float(one(agg,system='Extra Trees',endpoint='rul90')['primary']),changed=float(r['primary'])))
    leak = read('submission/leakage_overall_metrics.csv')
    leak_sources = read('submission/leakage_source_metrics.csv')
    for r in leak:
        ss=[x for x in leak_sources if all(x[k]==r[k] for k in ('target','mode','seed'))]
        metric='mae' if r['target']=='soh30' else 'rmsle'
        check('leakage source mean '+str((r['target'],r['mode'],r['seed'])),mean(ss,metric),float(r[metric]),12)
    for target in ('soh30','rul90'):
        metric = 'mae' if target=='soh30' else 'rmsle'
        clean = [float(r[metric]) for r in leak if r['target']==target and r['mode']=='family_disjoint']
        shared = [float(r[metric]) for r in leak if r['target']==target and r['mode']=='random_window_negative_control']
        table5.append(dict(check='Shared history rows',scope='Extra Trees '+target,reference=f'{min(clean)} to {max(clean)}',changed=f'{min(shared)} to {max(shared)}'))
    joint = read('submission/stanford_components_full_precision.csv')
    for component, model in [('Stanford','extra_trees_locked'),('Stanford_2','extra_trees_locked'),('Stanford','causal_linear_trend')]:
        r=one(joint,component=component,model_id=model,target='rul90')
        table5.append(dict(check='Joint Stanford',scope=model+' '+component+' rul90',reference=r['separate_holdout_metric'],changed=r['joint_training_exclusion_component_metric']))
    save('table5.csv',table5)
    residual = read('submission/residual_paired_deltas.csv')
    for key,value in [('direct_mae',.048019),('residual_mae',.012974),('trend_mae',.011597)]:
        check(key,mean(residual,key),value,6)
    check('residual beats direct',sum(float(r['residual_mae'])<float(r['direct_mae']) for r in residual),11,0)
    check('residual beats Trend',sum(float(r['residual_mae'])<float(r['trend_mae']) for r in residual),3,0)
    xjtu = read('submission/xjtu_group_score_summary.csv')
    for model,qhat,covered in [('TCN.2026',1.7531,0),('TCN.2027',2.7334,38)]:
        r=one(xjtu,model=model,recipe='family_max')
        check(model+' threshold',float(r['qhat']),qhat,4)
        check(model+' covered groups',int(r['covered_groups']),covered,0)
    zero = [r for r in read('submission/width_decomposition_by_source.csv') if r['model']=='TCN.2026' and r['recipe']=='window']
    check('per-origin zero thresholds',sum(float(r['qhat'])==0 for r in zero),19,0)
    check('per-origin scored contexts',len(zero),35,0)
    checkpoint = json.loads((ROOT/'results/common_input/checkpoint_sensitivity.json').read_text())
    check('later better validation runs',sum(r['later_better'] for r in checkpoint),17,0)
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')

    if args.figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2,figsize=(9,5),layout='constrained')
        for ax,target in zip(axes,('soh30','rul90')):
            ss=[s for s in sources if not(target=='rul90' and s=='CALB')]
            for model,marker in [('TCN','o'),('GRU','s'),('Extra Trees','^'),('Trend','D')]:
                values=[float(one(fig2,source=s,endpoint=target,system=model)['primary']) for s in ss]
                ax.scatter(values,range(len(ss)),label=model,marker=marker,s=25)
            ax.set_yticks(range(len(ss)),ss); ax.set_xlabel('SoH30 MAE' if target=='soh30' else 'RUL90 RMSLE')
            if target=='soh30': ax.set_xscale('log')
            ax.grid(axis='x',alpha=.2)
        axes[0].legend(fontsize=8)
        for suffix in ('pdf','png'): fig.savefig(out/('figure2.'+suffix),dpi=180)
        plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
        for ax,target in zip(axes,('soh30','rul90','rul80')):
            rr=[r for r in fig3 if r['target']==target]
            for offset,key,label in [(-.17,'pointwise','Pointwise'),(.17,'all_history','All-history')]:
                ax.barh([i+offset for i in range(4)],[r[key] for r in rr],height=.3,label=label)
            ax.set_yticks(range(4),[r['method'] for r in rr] if target=='soh30' else []);ax.invert_yaxis();ax.axvline(80,color='black',linestyle='--',linewidth=.8)
            ax.set_xlim(0,100);ax.set_title({'soh30':'SoH30','rul90':'RUL90','rul80':'RUL80'}[target]);ax.set_xlabel('Coverage (%)')
        axes[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.18),ncol=2)
        for suffix in ('pdf','png'): fig.savefig(out/('figure3.'+suffix),dpi=180)
        plt.close(fig)
    print(json.dumps(dict(status='SUBMISSION_AGGREGATES_PASS',checks=len(checks),figures=args.figures,output='build/submission')))

if __name__ == '__main__':
    main()
