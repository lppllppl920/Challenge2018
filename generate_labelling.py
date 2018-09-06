from pathlib import Path
import json
import urllib.request

root = Path("G:\Johns Hopkins University\Challenge\davinci_surgical_video/video_1\labels")
filename_list = list(root.glob("*.json"))
count = 0

for filename in filename_list:
    with open(str(filename), 'r') as f:
        loaded_json = json.load(f)
        for item in loaded_json:
            urllib.request.urlretrieve(item['Labeled Data'], str(root / "color_{:05d}.png").format(count))
            urllib.request.urlretrieve(item['Masks']['Instrument'], str(root / "mask_{:05d}.png").format(count))
            count += 1
        # print(loaded_json)

