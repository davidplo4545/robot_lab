#!/usr/bin/env python3
"""Unlock / lock the FR3 from the command line instead of the Desk web page.

Runs on the CONTROL PC (the only machine that can reach the robot at 172.16.0.2).

    desk.py unlock          take control, unlock the joints, activate FCI, restart the driver
                            service (franka-moveit) so it connects to the fresh FCI session
    desk.py unlock --force  same, but take control from someone else (e.g. an open
                            browser tab): you must press the circle button on the robot
    desk.py lock            stop the driver service, deactivate FCI, lock the joints
    desk.py shutdown        lock, then shut the control box's system down; switch the box
                            off once the robot's lights are out
    desk.py release         give up Desk control so the web page can take it

From this PC:  ssh -t control ~/desk_venv/bin/python ~/desk.py unlock
(or add to this PC's ~/.bashrc:  desk() { ssh -t control ~/desk_venv/bin/python ~/desk.py "$@"; }
 and then just type:  desk unlock)

Setup on the control PC (once):
    python3 -m venv ~/desk_venv
    ~/desk_venv/bin/pip install panda-python
    # login file, readable only by you:
    install -m 600 /dev/null ~/.config/franka_desk.ini
    nano ~/.config/franka_desk.ini      # see format below

~/.config/franka_desk.ini:
    [desk]
    host = 172.16.0.2
    user = <Desk username>
    password = <Desk password>

The release of the user stop (the button) stays manual on purpose.
"""
import argparse
import configparser
import logging
import os
import stat
import subprocess
import sys

CONFIG = os.path.expanduser('~/.config/franka_desk.ini')
SERVICE = 'franka-moveit'   # the systemd unit running the franka_ros2 launch


def load_login():
    if not os.path.exists(CONFIG):
        sys.exit(f'missing {CONFIG}: see the setup notes at the top of {__file__}')
    if os.stat(CONFIG).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(f'warning: {CONFIG} is readable by other users; run: chmod 600 {CONFIG}',
              file=sys.stderr)
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG)
    try:
        d = cfg['desk']
        return d.get('host', '172.16.0.2'), d['user'], d['password']
    except KeyError as e:
        sys.exit(f'{CONFIG}: missing {e} (needs a [desk] section with user and password)')


def main():
    ap = argparse.ArgumentParser(description='Unlock/lock the FR3 without the Desk web page.')
    ap.add_argument('action', choices=['unlock', 'lock', 'shutdown', 'release'])
    ap.add_argument('--force', action='store_true',
                    help='take control from another session (press the circle button on the robot)')
    ap.add_argument('--no-service', action='store_true',
                    help=f"don't restart/stop the {SERVICE} driver service")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        from panda_py import Desk
    except ImportError:
        sys.exit('panda_py not found: run with ~/desk_venv/bin/python (see setup notes)')

    host, user, password = load_login()
    try:
        desk = Desk(host, user, password, platform='fr3')   # logs in, tries to take control
    except Exception as e:  # noqa: BLE001 - network/login errors come in several types
        sys.exit(f'could not log in to Desk at {host}: {e}\n'
                 '(is the control box on and booted? right username/password?)')

    if args.action == 'release':
        desk.release_control()
        print('released Desk control')
        return

    if not desk.has_control():
        if not args.force:
            sys.exit('someone else has Desk control (probably an open Desk browser tab).\n'
                     'Close it / release control there, or rerun with --force and press the '
                     'circle button on the robot.')
        print('taking control: press the CIRCLE button on the robot now ...')
        desk.take_control(force=True)

    if args.action == 'unlock':
        print('unlocking joints ...')
        desk.unlock()
        print('activating FCI ...')
        desk.activate_fci()
        # Any FCI change drops the driver's connection, and the launch doesn't notice:
        # restart it so it connects fresh to the newly activated FCI.
        if service('restart', args.no_service):
            print('done: joints unlocked, FCI active, driver restarted (ready in ~10 s)')
        else:
            print('done: joints unlocked, FCI active')
    else:  # lock or shutdown
        service('stop', args.no_service)   # stop the driver cleanly before FCI goes away
        print('deactivating FCI ...')
        desk.deactivate_fci()
        print('locking joints ...')
        desk.lock()
        print('done: FCI off, joints locked')
        if args.action == 'shutdown':
            shutdown(desk)


def shutdown(desk):
    """Ask the control box to shut down its system (like Desk's menu -> Shutdown).

    panda_py has no shutdown call; this is the request franka_lock_unlock's
    franka_shutdown.py sends (POST /admin/api/shutdown with the control token). That tool
    was tested on older Desk versions, so the token goes in both the body and the header
    (panda_py's FR3 calls use the header)."""
    import requests
    token = desk._token.token
    print('shutting down the control box ...')
    try:
        r = desk._request('post', '/admin/api/shutdown', json={'token': token},
                          headers={'X-Control-Token': token}, check=False)
    except (requests.exceptions.RequestException, ConnectionError):
        r = None   # the box can go down before it finishes answering: that's success
    if r is not None and not 200 <= r.status_code < 300:
        sys.exit(f'shutdown request refused ({r.status_code}): {r.text.strip()[:200]}\n'
                 'Use Desk in the browser (menu -> Shutdown) instead.')
    print('shutdown started: wait until the lights on the robot go out, then switch the '
          'control box off')


def service(action, skip):
    """Restart/stop the driver service (franka-moveit); False if skipped or not installed."""
    if skip:
        return False
    if subprocess.run(['systemctl', 'cat', SERVICE], capture_output=True).returncode != 0:
        print(f'({SERVICE} service not installed: start/stop the launch yourself)')
        return False
    print(f'{action} {SERVICE} driver (sudo) ...')
    if subprocess.run(['sudo', 'systemctl', action, SERVICE]).returncode != 0:
        print(f'warning: could not {action} {SERVICE}; run: sudo systemctl {action} {SERVICE}',
              file=sys.stderr)
        return False
    return True


if __name__ == '__main__':
    main()
