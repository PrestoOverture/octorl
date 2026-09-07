"""Read-only pytest accounting runner, executed ONLY inside the sandbox.

Events use a separate file, never pytest's printed summary. This is not a trust
boundary against arbitrary in-process Python: code can introspect/forge events.
The host checks the complete protocol against reference-collected node IDs.
"""
import json
import os
import sys

import pytest
from hypothesis import settings

_WORKSPACE = os.path.abspath(os.environ.get('OCTORL_WORKSPACE', '/workspace'))
_GRADING = os.path.abspath(os.environ.get('OCTORL_GRADING', '/grading'))


def main():
    event_path, *args = sys.argv[1:]
    fd = os.open(event_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    write = os.write
    encode = json.dumps

    def emit(kind, **data):
        write(fd, (encode({'event': kind, **data}, sort_keys=True) + '\n').encode())

    def protected(path):
        if not isinstance(path, (str, bytes)):
            return False
        path = os.path.abspath(os.fsdecode(path))
        parts = path.split('/')
        return path.startswith(_GRADING + '/') or (
            path.startswith(_WORKSPACE + '/') and (
                'tests' in parts or 'hidden_tests' in parts or parts[-1] == 'conftest.py'))

    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            if protected(path) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                emit('protected_write', path=os.fsdecode(path))
        elif event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.rename'}:
            if args and protected(args[0]):
                emit('protected_write', path=os.fsdecode(args[0]))
            if event == 'os.rename' and len(args) > 1 and protected(args[1]):
                emit('protected_write', path=os.fsdecode(args[1]))

    sys.addaudithook(audit)
    # Persistent example databases would make examples depend on prior rollouts.
    settings.register_profile('octorl-grading', database=None, deadline=None)
    settings.load_profile('octorl-grading')

    class Accounting:
        def pytest_collection_finish(self, session):
            emit('collection', ids=[item.nodeid for item in session.items])

        def pytest_collectreport(self, report):
            if report.failed:
                emit('collection_error', nodeid=report.nodeid)

        def pytest_runtest_logreport(self, report):
            emit('test', nodeid=report.nodeid, when=report.when, outcome=report.outcome)

        def pytest_sessionfinish(self, session, exitstatus):
            emit('finish', exit_code=int(exitstatus))

    emit('start', protocol=1)
    sys.path.insert(0, _WORKSPACE)
    code = pytest.main(['-p', 'no:cacheprovider', *args], plugins=[Accounting()])
    emit('returned', exit_code=int(code))
    os.close(fd)
    return int(code)


if __name__ == '__main__':
    raise SystemExit(main())
