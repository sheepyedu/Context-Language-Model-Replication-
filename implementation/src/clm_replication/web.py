"""Public search/fetch CLI, with no model or task-reference access."""

import argparse
import json
import os
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('search', 'fetch'))
    parser.add_argument('value')
    args = parser.parse_args()
    key = 'query' if args.action == 'search' else 'url'
    base = os.environ['OPENAI_BASE_URL'].removesuffix('/v1')
    request = urllib.request.Request(
        base + '/clm/web', data=json.dumps({'action': args.action, key: args.value}).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['SESSION_ID']},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        print(response.read().decode())


if __name__ == '__main__':
    main()
