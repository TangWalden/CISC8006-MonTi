import json
import gdown

url = 'https://drive.google.com/drive/folders/1E2FE8zDIS4XZA6Fxe5l3hTrXUnfhJ2gr'
items = gdown.download_folder(url, skip_download=True, quiet=False)
print(json.dumps([{'id': x.id, 'path': x.path} for x in (items or [])], indent=2))
