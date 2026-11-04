import argparse
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = root.parent / 'Gemma4_kaggle_repo/scripts/kaggle_review.py'
spec = importlib.util.spec_from_file_location('kaggle_review', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
destination = Path(__file__).parent / 'kaggle'
destination.mkdir(exist_ok=True)
for resource, topic in [('pages', None), ('topic', 743255)]:
    args = argparse.Namespace(competition='gemma-4-developer-agent-paper', topic_id=topic, page_size=200, page_token=None)
    record = module.capture(resource, args, destination)
    print(json.dumps(record, ensure_ascii=False, indent=2))
