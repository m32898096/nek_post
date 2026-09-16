"""Analyze actual mesh refinement and saved common-grid solution differences."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from nek_post.h_refinement import HRefinementStudy
from nek_post.paths import ProjectPaths
from nek_post.h_refinement_mesh import read_mesh_report, analyze_mesh_refinement
from nek_post.h_refinement_analysis import analyze_saved_comparisons, tree_hashes
from nek_post.h_refinement_analysis_io import save_analysis

REPO_ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths-config',type=Path,default=REPO_ROOT/'config/paths.yaml')
    parser.add_argument('--study-config',type=Path,default=REPO_ROOT/'config/h_refinement.yaml')
    parser.add_argument('--input-dir',type=Path,default=REPO_ROOT/'results/h_refinement/field_comparison')
    parser.add_argument('--output-dir',type=Path,default=REPO_ROOT/'results/h_refinement/convergence_analysis')
    parser.add_argument('--change-tolerance',type=float,default=.05)
    parser.add_argument('--max-shift',type=float,default=.5)
    args=parser.parse_args()
    try:
        source,output=args.input_dir.resolve(),args.output_dir.resolve()
        paths=ProjectPaths.from_yaml(args.paths_config)
        study=HRefinementStudy.from_yaml(args.study_config)
        if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Output must be new and separate from all Step 3 artifacts.')
        if 'h_refinement' not in output.parts or 'poly_order_compare' in output.parts:
            raise ValueError('Use a separate h_refinement result tree.')
        if any(output.is_relative_to(p.resolve()) for p in paths.case_dirs.values()):
            raise ValueError('Output must be outside raw case directories.')
        inventory=json.loads((source/'inventory.json').read_text())
        before=tree_hashes(source)
        indexed={c['case']:c for c in inventory['cases']}
        meshes={}
        for case in study.cases:
            path=paths.case_dir(case)/indexed[case]['mesh_source_file']
            print(f'Inspecting actual element geometry: {case}',flush=True)
            mesh=read_mesh_report(path)
            if mesh['number_of_elements'] != indexed[case]['number_of_elements']:
                raise ValueError(f'{case}: mesh count differs from Step 3 inventory.')
            if any(d['polynomial_order']!=study.expected_polynomial_order for d in mesh['directional'].values()):
                raise ValueError(f'{case}: polynomial order differs from configured study.')
            meshes[case]=mesh
        refinement=analyze_mesh_refinement(meshes)
        scalar=refinement['scalar_h_justified']
        h_values=[refinement['scalar_h_by_case'][c] for c in study.cases] if scalar else None
        analysis=analyze_saved_comparisons(source,study.cases,h_values=h_values,scalar_h_justified=scalar,
                                          change_tolerance=args.change_tolerance,max_shift=args.max_shift)
        after=tree_hashes(source)
        if before!=after:
            raise ValueError('Step 3 artifact hashes changed during analysis; no analysis published.')
        save_analysis(output,meshes,refinement,analysis,before)
        print(f'Saved analysis to {output}; Step 3 SHA256 hashes unchanged.',flush=True)
    except (ValueError,KeyError,OSError) as exc:
        parser.exit(1,f'ERROR: {exc}\n')


if __name__=='__main__':
    main()
