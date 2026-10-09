"""Notebook display only. Does not change training or persisted study state."""
import html
import time


class CompactBar:
    def __init__(self, dashboard, total, description, unit='step'):
        self.dashboard, self.total, self.description = dashboard, total or 0, description
        self.unit, self.n, self.detail = unit, 0, ''
        dashboard.bars.append(self)
        dashboard.render(force=True)

    def update(self, n=1):
        self.n += n
        self.dashboard.render()

    def set_postfix_str(self, text):
        self.detail = text
        self.dashboard.render()

    def reset(self, total=None):
        self.n = 0
        if total is not None:
            self.total = total
        self.dashboard.render(force=True)

    def write(self, text):
        self.dashboard.message = str(text)
        self.dashboard.render(force=True)

    def close(self):
        if self in self.dashboard.bars:
            self.dashboard.bars.remove(self)
        self.dashboard.render(force=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class StudyDashboard:
    def __init__(self, seed, emit=None, clock=time.monotonic):
        self.seed, self.clock = seed, clock
        self.emit = emit
        self.handle = None
        self.last_render = float('-inf')
        self.bars, self.patches = [], []
        self.jobs, self.pointers = {}, {}
        self.current_job, self.current_request = None, ''
        self.message = 'Checking saved R2 progress'
        self.latest_save = 'No new save this session'

    def bar(self, total, description, enabled=True, leave=True):
        return CompactBar(self, total, description)

    def transfer_bar(self, total, description, enabled=True):
        return CompactBar(self, total, description, 'B')

    def download_bar(self, *args, **kwargs):
        return CompactBar(self, kwargs.get('total', 0),
            kwargs.get('desc') or 'Download dataset', kwargs.get('unit', 'B'))

    def install(self):
        import uncle.paired_study as paired
        import uncle.study_training as training
        import uncle.storage as storage_module
        import torchvision.datasets.utils as dataset_utils
        for module, attribute, replacement in (
            (paired, 'progress_bar', self.bar),
            (training, 'progress_bar', self.bar),
            (storage_module, '_transfer_bar', self.transfer_bar),
            (dataset_utils, 'tqdm', self.download_bar),
        ):
            self.patches.append((module, attribute, getattr(module, attribute)))
            setattr(module, attribute, replacement)
        self.render(force=True)

    def restore(self):
        for module, attribute, original in reversed(self.patches):
            setattr(module, attribute, original)
        self.patches.clear()

    def attach_here(self):
        # Keep the live panel beside the running cell, with no stale setup panel.
        if self.handle is not None:
            from IPython.display import HTML
            self.handle.update(HTML(''))
            self.handle = None
        self.render(force=True)

    def configure(self, plan, storage, all_jobs):
        # Count the narrowed study across all planned seeds, not the deferred 45-job matrix.
        self.jobs = {job['id']: job for job in all_jobs if job['order'] == 'order_01'
            and (job['kind'] != 'unlearned' or job.get('forget_task') in ('3', '14'))}
        self.pointers = {key: storage.progress(job) for key, job in self.jobs.items()}
        self.message = 'Ready; completed work will be skipped'
        self.render(force=True)

    def start_job(self, job):
        self.current_job = job['id']
        self.current_request = 'Restore model and prepare data'
        self.message = 'Running'
        self.render(force=True)

    def request(self, action, task):
        self.current_request = ('Learn ' if action == 'learn' else 'Unlearn ') + str(task)
        self.render(force=True)

    def saved(self, job, pointer):
        self.pointers[job['id']] = pointer
        self.latest_save = pointer['updated_at'] + ' (UTC)'
        self.render(force=True)

    def completed_requests(self, key):
        pointer = self.pointers.get(key)
        if not pointer:
            return 0
        job = self.jobs[key]
        total = len(job['operations'])
        if pointer['status'] == 'complete':
            return total
        inherited = len(self.jobs[job['source']]['operations']) if job['source'] else 0
        return max(0, min(total, pointer['completed_requests'] - inherited))

    def counts(self, seed=None):
        keys = [key for key, job in self.jobs.items() if seed is None or job['seed'] == seed]
        total = sum(len(self.jobs[key]['operations']) for key in keys)
        done = sum(self.completed_requests(key) for key in keys)
        complete = sum(bool(self.pointers.get(key) and
            self.pointers[key]['status'] == 'complete') for key in keys)
        pairs = sum(bool(self.pointers.get(key) and self.pointers[key]['status'] == 'complete'
            and (self.pointers.get('controls/' + self.jobs[key]['order'] + '/seed_' +
                str(self.jobs[key]['seed']) + '/learn_15') or {}).get('status') == 'complete')
            for key in keys if self.jobs[key]['kind'] == 'unlearned')
        return done, total, complete, len(keys), pairs

    @staticmethod
    def meter(n, total):
        return '<progress value="%s" max="%s" style="width:100%%"></progress>' % (n, total or 1)

    def markup(self):
        escape = lambda value: html.escape(str(value))
        done, total, complete, jobs_total, pairs = self.counts()
        seed_done, seed_total, seed_jobs, seed_jobs_total, seed_pairs = self.counts(self.seed)
        lines = ['<div style="border:1px solid #bbb;padding:12px;max-width:850px">',
            '<b>Paired study: order 01, U3/L15 and U14/L15</b>',
            '<p>Source: L3 → L0 → L9 → L5 → L17 → L1 → L7 → L14</p>',
            '<p>All planned seeds: %s/%s jobs complete; %s/%s comparisons complete; '
            '%s/%s requests saved complete.</p>' %
            (complete, jobs_total, pairs, jobs_total // 2, done, total),
            self.meter(done, total),
            '<p><b>Current seed %s:</b> %s/%s jobs; %s/2 comparisons; %s/%s requests saved complete.</p>' %
            (self.seed, seed_jobs, seed_jobs_total, seed_pairs, seed_done, seed_total),
            self.meter(seed_done, seed_total),
            '<p>Current job: %s<br>Request: %s<br>Status: %s</p>' %
            (escape(self.current_job or 'Not started'), escape(self.current_request), escape(self.message))]
        lines.append('<table><tr><th>Selected seed jobs</th><th>Saved status</th></tr>')
        for key, job in self.jobs.items():
            if job['seed'] != self.seed:
                continue
            pointer = self.pointers.get(key)
            label = ('Source (8 lessons)' if job['kind'] == 'source' else
                'Control: L15' if job['kind'] == 'control' else 'U%s then L15' % job['forget_task'])
            status = pointer['status'] if pointer else 'pending'
            lines.append('<tr><td>%s</td><td>%s</td></tr>' % (escape(label), escape(status)))
        lines.append('</table>')
        if self.bars:
            bar = self.bars[-1]
            amount = ('%.1f / %.1f MiB' % (bar.n / 2**20, bar.total / 2**20)
                if bar.unit == 'B' else '%s / %s steps' % (bar.n, bar.total))
            lines += ['<p>%s: %s. %s</p>' % (escape(bar.description), escape(amount), escape(bar.detail)),
                self.meter(bar.n, bar.total)]
        lines += ['<p>Latest verified R2 save: %s</p>' % escape(self.latest_save),
            '<small>Request counts are not GPU-time estimates. Deferred orders and targets are excluded.</small></div>']
        return ''.join(lines)

    def render(self, force=False):
        now = self.clock()
        if not force and now - self.last_render < 1.0:
            return
        self.last_render = now
        markup = self.markup()
        if self.emit:
            self.emit(markup)
            return
        from IPython.display import HTML, display
        if self.handle is None:
            self.handle = display(HTML(markup), display_id=True)
        else:
            self.handle.update(HTML(markup))

    def finish(self, message):
        self.message = message
        self.bars.clear()
        self.render(force=True)


dashboard = StudyDashboard(SEED)
dashboard.install()
