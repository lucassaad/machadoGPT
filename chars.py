import os 

path = 'inputs'

for f in  os.listdir(path):
    f = os.path.join(path, f)
    with open(f,'r') as file:
        print(f"Número de caracteres em {f}: {len(file.read())}")