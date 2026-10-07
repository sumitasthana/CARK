"""Private R2 file transfers. Training uses local files, not a mounted bucket."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _key(value):
    parts = value.split('/')
    if not value or '\\' in value or any(p in ('', '.', '..') for p in parts) or ':' in value:
        raise ValueError('Use a relative R2 file name with forward slashes.')
    return value


class R2Store:
    """One writer per experiment folder; uploads are checked by reading them back."""

    def __init__(self, client, bucket):
        self.client = client
        self.bucket = bucket

    @classmethod
    def connect(cls, endpoint, bucket, access_key, secret_key):
        if not re.fullmatch(r'https://[a-z0-9.]+\.r2\.cloudflarestorage\.com', endpoint):
            raise ValueError('Use the HTTPS S3 API endpoint shown in your R2 dashboard.')
        import boto3
        from botocore.config import Config
        client = boto3.client('s3', endpoint_url=endpoint, region_name='auto',
            aws_access_key_id=access_key, aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4', retries={'mode': 'standard', 'max_attempts': 5},
                request_checksum_calculation='when_required', response_checksum_validation='when_required'))
        return cls(client, bucket)

    def info(self, key):
        key = _key(key)
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as error:
            code = getattr(error, 'response', {}).get('Error', {}).get('Code')
            if code in ('404', 'NoSuchKey', 'NotFound'):
                return None
            raise

    def keys(self, prefix):
        _key(prefix.rstrip('/'))
        pages = self.client.get_paginator('list_objects_v2').paginate(Bucket=self.bucket, Prefix=prefix)
        for page in pages:
            for item in page.get('Contents', []):
                yield item['Key']

    def verify(self, key, expected_sha256, expected_size):
        response = self.client.get_object(Bucket=self.bucket, Key=_key(key))
        body = response['Body']
        digest, size = hashlib.sha256(), 0
        try:
            for chunk in iter(lambda: body.read(1024 * 1024), b''):
                digest.update(chunk)
                size += len(chunk)
        finally:
            body.close()
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise ValueError(f'R2 file verification failed: {key}')

    def upload(self, path, key, *, overwrite=False):
        path, key = Path(path), _key(key)
        digest, size = sha256(path), path.stat().st_size
        existing = self.info(key)
        same = existing is not None and existing.get('Metadata', {}).get('sha256') == digest and existing['ContentLength'] == size
        if existing is not None and not same and not overwrite:
            raise FileExistsError(f'R2 already has a different file: {key}. Choose a new experiment name.')
        if not same:
            self.client.upload_file(str(path), self.bucket, key, ExtraArgs={'Metadata': {'sha256': digest}})
        self.verify(key, digest, size)
        if sha256(path) != digest:
            raise ValueError(f'The local file changed during upload: {path}')
        return {'key': key, 'sha256': digest, 'bytes': size}

    def download(self, key, cache):
        key = _key(key)
        info = self.info(key)
        if info is None:
            raise FileNotFoundError(f'No file in R2: {key}. Run the CPU copy first.')
        digest = info.get('Metadata', {}).get('sha256', '')
        if not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError(f'R2 file has no recorded SHA256: {key}. Upload it through R2Store first.')
        # Include the bucket and full key so two objects cannot share a cache entry.
        identity = hashlib.sha256((self.bucket + '/' + key).encode()).hexdigest()
        path = Path(cache) / identity / PurePosixPath(key).name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size == info['ContentLength'] and sha256(path) == digest:
            return path
        temporary = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
        try:
            self.client.download_file(self.bucket, key, str(temporary))
            if temporary.stat().st_size != info['ContentLength'] or sha256(temporary) != digest:
                raise ValueError(f'Downloaded file verification failed: {key}')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def copy_tree(self, root, prefix, *, on_file=None, exclude_dirs=(), suffixes=None):
        """Copy allowed files, pruning excluded folders before reading their contents."""
        root, prefix = Path(root).resolve(), _key(prefix)
        if not root.is_dir():
            raise FileNotFoundError(root)
        excluded = {name.casefold() for name in exclude_dirs}
        allowed = None if suffixes is None else {suffix.casefold() for suffix in suffixes}
        count, total = 0, 0
        skipped_dirs, skipped_files = [], 0
        def walk_error(error):
            raise error
        for directory, folders, files in os.walk(root, topdown=True, onerror=walk_error):
            directory = Path(directory)
            for name in list(folders):
                path = directory / name
                if name.casefold() in excluded:
                    folders.remove(name)
                    skipped_dirs.append(path.relative_to(root).as_posix())
                elif path.is_symlink():
                    raise ValueError(f'Copy ordinary folders instead of symbolic links: {path}')
            folders.sort()
            for name in sorted(files):
                path = directory / name
                if allowed is not None and path.suffix.casefold() not in allowed:
                    skipped_files += 1
                    continue
                if path.is_symlink():
                    raise ValueError(f'Copy ordinary files instead of symbolic links: {path}')
                path.resolve().relative_to(root)
                result = self.upload(path, prefix + '/' + path.relative_to(root).as_posix())
                count += 1
                total += result['bytes']
                if on_file is not None:
                    on_file(result)
        return {'files': count, 'bytes': total, 'skipped_directories': skipped_dirs,
                'skipped_files': skipped_files}

    def publish_lesson(self, output, prefix, *, complete=False):
        """Publish the final model before the completed report. Epochs save reports only."""
        import json
        output, prefix = Path(output), _key(prefix)
        report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
        if complete:
            if report['status'] != 'complete':
                raise ValueError('The lesson has not finished.')
            self.upload(output / 'checkpoint.pt', prefix + '/checkpoint.pt')
        elif report['status'] == 'complete':
            raise ValueError('Publish the final model before publishing its completed report.')
        self.upload(output / 'settings.json', prefix + '/settings.json')
        self.upload(output / 'report.json', prefix + '/report.json', overwrite=True)
