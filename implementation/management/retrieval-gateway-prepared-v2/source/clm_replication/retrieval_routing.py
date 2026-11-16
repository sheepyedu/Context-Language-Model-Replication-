"""Choose the retrieval backend from driver-owned task metadata."""

BACKENDS = {
    'public_web': 'http://127.0.0.1:19180',
    'browsecomp_plus_fixed': 'http://127.0.0.1:19181/retrieve',
}


def retrieval_backend(task_metadata):
    profile = task_metadata.get('retrieval_profile', 'public_web')
    if task_metadata.get('dataset') == 'browsecomp_plus' and profile != 'browsecomp_plus_fixed':
        raise ValueError('BrowseComp-Plus tasks require the frozen corpus')
    if profile not in BACKENDS:
        raise ValueError('Unknown driver-owned retrieval profile')
    return BACKENDS[profile]


async def retrieve(payload, task_metadata, client):
    # The payload cannot choose a backend or transmit evaluator labels.
    action = payload.get('action')
    fields = {'search': {'action', 'query'}, 'fetch': {'action', 'url'}}
    if action not in fields or set(payload) != fields[action]:
        raise ValueError('Expected search/query or fetch/url')
    response = await client.post(retrieval_backend(task_metadata), json=payload)
    response.raise_for_status()
    return response.json()
