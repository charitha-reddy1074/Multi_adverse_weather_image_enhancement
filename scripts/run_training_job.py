"""Run training with durable output, completion status and timestamped LOG entries."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', help='Checkpoint to restore into a new output directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'status.json'
    if status_path.exists() or (output / 'latest.pt').exists():
        raise FileExistsError(f'Use a new output directory; existing run found: {output}')
    command = [sys.executable, '-B', '-u', str(root / 'train.py'),
               '--config', args.config, '--output', str(output)]
    if args.resume:
        checkpoint = (root / args.resume).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Resume checkpoint missing: {checkpoint}')
        command.extend(['--resume', str(checkpoint)])
    status = dict(state='starting', started_at=datetime.now().astimezone().isoformat(),
                  config=args.config, output=str(output), command=command,
                  initialization=f'resume from {checkpoint}' if args.resume else 'fresh; no checkpoint loaded')

    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2), encoding='utf8')
        temporary.replace(status_path)

    def log(message):
        stamp = datetime.now().astimezone().isoformat(timespec='seconds')
        with (root / 'LOG.md').open('a', encoding='utf8') as stream:
            stream.write(f'\n\n## {stamp} - Training job\n\n- {message}\n')

    save()
    try:
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONUTF8='1',
                   PYTHONDONTWRITEBYTECODE='1', TQDM_MININTERVAL='30')
        with (output / 'training.log').open('w', encoding='utf8') as stream:
            process = subprocess.Popen(command, cwd=root, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            status.update(state='running', pid=process.pid)
            save()
            log(f'Started training ({status["initialization"]}) with {args.config}; PID {process.pid}; '
                f'output {output}. Training and test membership are fixed by the config.')
            code = process.wait()
        status.update(state='completed' if code == 0 else 'failed', exit_code=code)
    except Exception as error:
        status.update(state='failed', error=str(error), exit_code=1)
    status['finished_at'] = datetime.now().astimezone().isoformat()
    status['final_test_available'] = (output / 'final_test.json').is_file()
    save()
    log(f'Training {status["state"]}: {args.config}; exit code {status["exit_code"]}; '
        f'output {output}. See training.log and status.json for evidence.')
    return status['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
