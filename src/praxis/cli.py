import argparse
import json
from .io import write_json


def main():
    parser = argparse.ArgumentParser(prog='praxis')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Execute task methods on current physical data')
    for name in ('data-root', 'rules-root', 'output'):
        run.add_argument('--'+name, required=True)
    run.add_argument('--tasks', '--split', dest='split', required=True, help='Task/episode list; --split retained for compatibility')
    score = commands.add_parser('score', help='Independent scoring; reads ground truth')
    score.add_argument('--run', required=True)
    score.add_argument('--data-root', required=True)
    score.add_argument('--output', required=True)
    compare = commands.add_parser('compare', help='Compare against frozen R2 predictions')
    for name in ('run', 'reference-root', 'reference-manifest', 'output'):
        compare.add_argument('--'+name, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        from .runner import run
        result = run(args.data_root, args.split, args.rules_root, args.output)
        print(json.dumps({'prediction_count': result['prediction_count'], 'status_counts': result['status_counts']}))
    else:
        from .evaluation import score, compare
        from pathlib import Path
        if Path(args.output).exists():
            parser.error('refusing result overwrite')
        result = score(args.run, args.data_root) if args.command == 'score' else compare(args.run, args.reference_root, args.reference_manifest)
        write_json(args.output, result)
        print(json.dumps(result.get('micro', result)))
        if args.command == 'compare' and result['differences']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
