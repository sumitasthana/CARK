"""Build the offline presentation from the committed, verified comparison evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
evidence = json.loads((ROOT / 'docs/reports/evidence/paired-study-2026-10-10.json').read_text(encoding='utf-8'))
rows = evidence['selected_comparisons']
assert len(rows) == 6 and all(all(r['pair_checks'].values()) for r in rows)
template = (ROOT / 'scripts/templates/paired_presentation.html').read_text(encoding='utf-8')
page = template.replace('__DATA__', json.dumps(evidence).replace('<', '\\u003c'))
assert '\u2014' not in page
output = ROOT / 'docs/reports/Paired-study-2026-10-10-presentation.html'
output.write_text(page, encoding='utf-8')
print(output)
