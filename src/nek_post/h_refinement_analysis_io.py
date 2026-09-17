"""Step 4 tables and plots; no writes to source field-comparison artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_table(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty analysis table.')
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save_analysis(output, meshes, refinement, analysis, source_hashes):
    """Write a new analysis tree with machine-readable rejection reasons."""
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    report=dict(meshes=meshes,mesh_refinement=refinement,analysis=analysis,
                finest_available_numerical_reference=analysis['case_order'][-1],
                source_sha256=source_hashes,source_artifacts_unchanged=True)
    (root/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    mesh_rows=[]
    for case,mesh in meshes.items():
        for axis,d in mesh['directional'].items():
            mesh_rows.append(dict(case=case,axis=axis,elements=d['element_count'],
                                  **d['spacing'],representative_spacing=d['representative_spacing']))
    write_table(root/'directional_mesh_resolution.csv',mesh_rows)
    ratio_rows=[]
    for transition in refinement['transitions']:
        for axis,d in transition['directional'].items():
            for segment in d['segments']:
                ratio_rows.append(dict(coarse_case=transition['coarse_case'],fine_case=transition['fine_case'],
                                       axis=axis,**segment))
    write_table(root/'local_refinement_ratios.csv',ratio_rows)
    metrics=[]; summary=[]
    for result in analysis['convergence']:
        info=dict(target_time=result['target_time'],field=result['field'])
        for pair,values in result['pairwise'].items():
            metrics.append(dict(info,pair=pair,**values,valid_point_count=result['valid_point_count'],
                                valid_mask_fraction=result['valid_mask_fraction']))
        summary.append(dict(info,classification=result['classification'],
            successive_difference_ratio=result['successive_difference_ratio'],
            finest_reference_error_ratio=result['finest_reference_error_ratio'],
            difference_vector_cosine=result['difference_vector_cosine'],
            observed_order=result['observed_order'],
            reasons='; '.join(result['classification_reasons']),
            order_rejection_reasons='; '.join(result['order_rejection_reasons']),
            time_spread=result['time_spread'],valid_mask_fraction=result['valid_mask_fraction']))
    write_table(root/'pairwise_differences.csv',metrics)
    write_table(root/'convergence_classification.csv',summary)
    write_table(root/'morphology_diagnostics.csv',analysis['morphology'])
    write_table(root/'translation_diagnostics.csv',analysis['translation'])
    finest_rows=[]
    for result in analysis['convergence']:
        finest_rows.extend(result['source_step3_rows'])
    write_table(root/'finest_reference_errors.csv',finest_rows)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,axis in zip(axes,('x','y','z')):
        for case,mesh in meshes.items():
            d=mesh['directional'][axis]
            centers=[(a+b)/2 for a,b in d['element_intervals']]
            ax.plot(centers,d['element_widths'],label=case)
        ax.set_xlabel(f'Physical {axis}');ax.set_ylabel(f'Element width Δ{axis}')
        ax.grid(alpha=.3);ax.legend()
    fig.suptitle('Actual element spacing: physical coordinates, not post-processing grid')
    fig.tight_layout();fig.savefig(root/'directional_mesh_spacing.png',dpi=180);plt.close(fig)
    fields=list(dict.fromkeys(r['field'] for r in analysis['convergence']))
    fig,axes=plt.subplots(1,len(fields),figsize=(14,4),squeeze=False)
    for ax,field in zip(axes[0],fields):
        selected=[r for r in analysis['convergence'] if r['field']==field]
        for key,label in (('coarse_medium','H − VH'),('medium_fine','VH − VVH'),('coarse_fine','H − VVH')):
            ax.plot([r['target_time'] for r in selected],[r['pairwise'][key]['rms_l2'] for r in selected],marker='o',label=label)
        ax.set_title(field.replace('_',' '));ax.set_xlabel('Target physical time')
        ax.set_ylabel('Common-mask RMS difference');ax.legend();ax.grid(alpha=.3)
    fig.suptitle('Pairwise differences; VVH is the finest available numerical reference')
    fig.tight_layout();fig.savefig(root/'pairwise_differences.png',dpi=180);plt.close(fig)
