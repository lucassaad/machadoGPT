import subprocess
import sys

TIPOS_LITERARIOS = ["texto", "poesia", "teatro"]


def main():
    for tipo_literario in TIPOS_LITERARIOS:
        print(f"\n=== Treinando: {tipo_literario} ===\n")

        result = subprocess.run(
            [sys.executable, "train.py", "resume", tipo_literario]
        )

        if result.returncode != 0:
            print(f"Erro ao treinar '{tipo_literario}' (código {result.returncode})")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()