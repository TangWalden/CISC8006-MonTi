import importlib
import json
import requests

download_module = importlib.import_module('gdown.download')
for name, file_id in [('best_models', '1A5d3l4-4_N9jUzok9J6TFd_DHALDBDgV'), ('Datasets', '1sLcalw491WFjZ3rzgD32DldsSU6MNjHx')]:
    session = requests.Session()
    url = 'https://drive.google.com/uc?id=' + file_id
    for attempt in range(4):
        response = session.get(url, stream=True, timeout=(90, 90))
        print(name, attempt, response.status_code, dict((k, v) for k, v in response.headers.items() if k.lower() in ['content-length', 'content-type', 'accept-ranges']), flush=True)
        if 'text/html' not in response.headers.get('Content-Type', ''):
            response.close()
            break
        url = download_module.get_url_from_gdrive_confirmation(response.text)
        response.close()
    response = session.get(url, headers={'Range': 'bytes=-65536'}, stream=True, timeout=(90, 90))
    print(name, 'RANGE', response.status_code, dict((k, v) for k, v in response.headers.items() if k.lower() in ['content-length', 'content-type', 'content-range', 'accept-ranges']), flush=True)
    if response.status_code == 206:
        tail = response.content
        open('downloads/' + name + '_tail.bin', 'wb').write(tail)
    response.close()
