import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('publication', ROOT / 'scripts/publish_experiment_trajectory.py')
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


class PublicationTests(unittest.TestCase):
    def test_cleanup_handles_readonly_git_files_and_rejects_other_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            scratch = temporary_root / 'cark-trajectory-publish-test'
            scratch.mkdir()
            packed = scratch / 'pack.idx'
            packed.write_bytes(b'git pack')
            os.chmod(packed, 0o444)
            publication.remove_checkout(scratch, temporary_root)
            self.assertFalse(scratch.exists())
            with self.assertRaises(ValueError):
                publication.remove_checkout(temporary_root, temporary_root)

    def test_publication_copies_only_record_and_svg_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            source, wiki = Path(directory) / 'source', Path(directory) / 'wiki'
            (source / publication.PAGE).parent.mkdir(parents=True)
            (source / publication.PAGE).write_text('# Experiment trajectory\n\nResult.\n')
            figures = source / 'docs/reports/trajectory/figures/2026-10-04'
            figures.mkdir(parents=True)
            (figures / 'chart.svg').write_text('<svg/>')
            (figures / 'private.json').write_text('{}')
            wiki.mkdir()
            (wiki / 'Home.md').write_text('Keep this page')
            copied = publication.copy_record(source, wiki)
            self.assertEqual(copied, ['Experiment-trajectory.md', 'figures/2026-10-04/chart.svg'])
            self.assertEqual((wiki / 'Home.md').read_text(), 'Keep this page')
            self.assertFalse((wiki / 'figures/2026-10-04/private.json').exists())
            self.assertEqual((wiki / 'Experiment-trajectory.md').read_text(), '# Experiment trajectory\n\nResult.\n')

    def test_bad_record_is_rejected_before_replacing_wiki_page(self):
        with tempfile.TemporaryDirectory() as directory:
            source, wiki = Path(directory) / 'source', Path(directory) / 'wiki'
            (source / publication.PAGE).parent.mkdir(parents=True)
            (source / publication.PAGE).write_text('An unrelated document')
            wiki.mkdir()
            (wiki / 'Experiment-trajectory.md').write_text('Existing record')
            with self.assertRaises(ValueError):
                publication.copy_record(source, wiki)
            self.assertEqual((wiki / 'Experiment-trajectory.md').read_text(), 'Existing record')


if __name__ == '__main__':
    unittest.main()
