import csv

with open('/home/rapids/data/BEAM/ModelScripts/data/clinical/dictionary.csv') as f:
    reader = csv.DictReader(f)
    vols = sorted(set(row["Variable_name"] for row in reader if row["Variable_name"].startswith("THK_")))

print('[' + ', '.join(f'"{v}"' for v in vols) + ']')

