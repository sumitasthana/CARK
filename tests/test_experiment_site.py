"""Check scientific values, source controls, and static report navigation."""
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_experiment_site import build


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids=[]
        self.links=[]
        self.charts=[]
    def handle_starttag(self,tag,attrs):
        fields=dict(attrs)
        if 'id' in fields: self.ids.append(fields['id'])
        if tag=='a': self.links.append(fields.get('href',''))
        if tag=='svg': self.charts.append(fields)


class ExperimentSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets=build()

    def test_site_links_and_accessible_figures(self):
        parser=ReportParser()
        text=self.assets['index.html'].decode()
        parser.feed(text)
        self.assertEqual(len(parser.ids),len(set(parser.ids)))
        self.assertEqual(len(parser.charts),6)
        for chart in parser.charts:
            self.assertEqual(chart['role'],'img')
            self.assertTrue(chart['aria-label'])
        for link in parser.links:
            if link.startswith('#'): self.assertIn(link[1:],parser.ids)
            elif link and not link.startswith('https://'): self.assertIn(link,self.assets)
        self.assertNotIn(chr(0x2014),text)
        self.assertNotIn('R2_SECRET_ACCESS_KEY',text)
        self.assertNotIn('cloudflarestorage.com',text)
        self.assertIn('Further experiments paused',text)

    def test_chart_values_and_comparison_scope(self):
        data=json.loads(self.assets['data/chart-data.json'])
        pair=data['pair']
        self.assertTrue(all(pair['pair_checks'].values()))
        self.assertEqual(pair['forget_task']['after_unlearning_accuracy'],10)
        self.assertEqual(pair['forget_task']['after_later_learning_accuracy'],10)
        self.assertAlmostEqual(pair['new_task_difference'],-5.4)
        for actual,expected in zip(data['sequence']['means'],[41.76,42.64,43.64,41.76]):
            self.assertAlmostEqual(actual,expected)
        self.assertEqual([t['beta'] for t in data['beta_trials'] if t['reused']],[.01])
        retained=pair['retained_tasks']
        spill=sum(abs(pair['after_unlearning_accuracies'][t]-pair['starting_accuracies'][t]) for t in retained)
        self.assertAlmostEqual(spill,10.8)
        self.assertNotIn('3',retained)
        self.assertNotIn('15',retained)


if __name__=='__main__': unittest.main()
