"""Apply the standalone reallocation rule to an explicit JSON attack capture.

This command does not load a dataset, generator, classifier or checkpoint. Node
IDs and all random keys must be supplied explicitly; no IDs are reindexed.
"""
import argparse
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.target_reallocation import reallocate_candidates


REQUIRED_KEYS = {'adjacency', 'target_ids', 'candidate_ids', 'injected_ids',
                 'seed', 'group_key'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='JSON capture with explicit IDs, seed and group key.')
    parser.add_argument('--output', required=True, help='New output JSON path; an existing file is never overwritten.')
    args = parser.parse_args()
    try:
        input_path, output_path = pathlib.Path(args.input), pathlib.Path(args.output)
        if output_path.exists():
            raise FileExistsError('Output already exists: '+str(output_path))
        with input_path.open(encoding='utf-8') as stream:
            capture = json.load(stream)
        if not isinstance(capture, dict) or set(capture) != REQUIRED_KEYS:
            actual = set(capture) if isinstance(capture, dict) else set()
            missing, extra = sorted(REQUIRED_KEYS-actual), sorted(actual-REQUIRED_KEYS)
            raise ValueError('Expected exactly the explicit capture keys; missing=%s, extra=%s.' % (missing, extra))
        result = reallocate_candidates(capture['adjacency'], capture['target_ids'],
            capture['candidate_ids'], capture['injected_ids'],
            seed=capture['seed'], group_key=capture['group_key'])
        # Exclusive creation also protects against a file appearing after the
        # existence check. The complete core output is retained without filtering.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write('\n')
        print('Wrote reallocation output to '+str(output_path))
    except (OSError, TypeError, ValueError) as error:
        parser.exit(2, 'Error: '+str(error)+'\n')


if __name__ == '__main__':
    main()
