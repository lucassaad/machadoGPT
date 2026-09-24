import os 

DATASETS_DIR = 'datasets'
OUTPUT_DIR = "inputs"

def concatenar_dataset(dataset_path, ouptut_file):
    with open(f"{ouptut_file}", 'w') as out:
        for nome_arquivo in os.listdir(dataset_path):
            arquivo_path = os.path.join(dataset_path, nome_arquivo)
            with open(arquivo_path, 'r') as input:
                out.write(input.read())

def main(): 
    for nome in os.listdir(DATASETS_DIR):
        caminho_completo = os.path.join(DATASETS_DIR, nome)

        if not os.path.isdir(caminho_completo): continue

        output_file = os.path.join(OUTPUT_DIR, f"{nome}.txt")
        concatenar_dataset(caminho_completo, output_file)

if __name__ == "__main__":
    main()