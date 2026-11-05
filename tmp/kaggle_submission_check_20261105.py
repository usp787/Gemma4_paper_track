"""Read-only Kaggle requirement refresh; all snapshots stay in this workspace."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[1]
source = root.parent / 'Gemma4_kaggle_repo/scripts/kaggle_review.py'
spec = importlib.util.spec_from_file_location('kaggle_review', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
destination = root / 'tmp/kaggle_submission_check_20261105'
destination.mkdir(exist_ok=True)
requests = [('overview', None), ('pages', None), ('topics', None),
            ('topic', 743255), ('topic', 745712)]

def read_one(request):
    resource, topic = request
    folder = destination / (f'topic_{topic}' if topic else resource)
    folder.mkdir(exist_ok=True)
    args = argparse.Namespace(competition='gemma-4-developer-agent-paper',
                              topic_id=topic, page_size=200, page_token=None)
    record = module.capture(resource, args, folder)
    return {key: record.get(key) for key in
            ['resource', 'source_url', 'ok', 'error', 'started_at_utc',
             'finished_at_utc', 'next_page_token']}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        requests = [('topic', int(topic)) for topic in sys.argv[1:]]
    with ThreadPoolExecutor(max_workers=5) as pool:
        for result in pool.map(read_one, requests):
            print(json.dumps(result, ensure_ascii=False), flush=True)
