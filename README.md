# MachadoGPT - Lucas Saad Rodrigues - 231035395

Projeto de geração de texto baseado em um modelo GPT treinado com obras de Machado de Assis.

O modelo utiliza tokenização em nível de caractere e uma arquitetura Transformer com self-attention causal.

## Estrutura do projeto

```text
machadoGPT/
├── machado.ipynb          # Notebook principal para execução e avaliação
├── prepare.py             # Preparação e codificação dos dados
├── train.py               # Treinamento e geração de textos
├── train_all.py           # Treinamento dos três estilos literários
├── inputs/                # Textos de entrada
├── inputs-bin/            # Dados codificados para treinamento
├── checkpoints/           # Modelos salvos durante o treinamento
├── out/                   # Textos gerados
└── README.md              # Documentação do projeto
```

## Arquivos Python

### `machado.ipynb`

Notebook principal do projeto. Reúne as etapas de preparação dos dados, definição do modelo, treinamento e geração de textos.

Este é o arquivo recomendado para a execução e avaliação pelo professor. As células devem ser executadas em ordem, a partir da raiz do projeto.

### `prepare.py`

Prepara os textos para serem utilizados pelo modelo.

O script:

- identifica os caracteres presentes no texto;
- cria o vocabulário;
- converte cada caractere em um número;
- divide os dados em 90% para treinamento e 10% para validação;
- gera os arquivos `train.bin`, `val.bin` e `meta.pkl`.

Os dados processados são armazenados em:

```text
inputs-bin/<tipo_literario>/
```

### `train.py`

Implementa e treina o modelo GPT.

O script:

- carrega os dados preparados;
- cria o modelo Transformer;
- calcula a loss de treinamento e validação;
- salva checkpoints;
- gera um texto de aproximadamente 2000 caracteres;
- salva o resultado em `out/<tipo_literario>/ouput.txt`.

O nome `ouput.txt` é mantido conforme definido no código.

### `train_all.py`

Executa o treinamento dos três estilos literários em sequência:

- `texto`;
- `poesia`;
- `teatro`.

Esse script utiliza o modo `resume`, portanto é necessário que existam checkpoints correspondentes.

## Dados de entrada

A pasta `inputs/` contém os textos de Machado de Assis separados por estilo literário:

- `poesia.txt`;
- `teatro.txt`;
- `texto.txt`.

Os arquivos processados são armazenados em `inputs-bin/`:

- `train.bin`: dados de treinamento codificados;
- `val.bin`: dados de validação codificados;
- `meta.pkl`: vocabulário e mapeamentos entre caracteres e números.


## Como executar pelo notebook

A execução recomendada é pelo arquivo `machado.ipynb`.

Na raiz do projeto, execute:

```bash
cd /home/lucassaad/projects/8-sem/redes-neurais/machadoGPT
```

Em seguida, abra o notebook no Jupyter Notebook ou no Visual Studio Code e execute as células em ordem.

Antes do treinamento, selecione o estilo literário:

```python
tipo_literario = 'poesia'
```

As opções disponíveis são:

```python
'poesia'
'teatro'
'texto'
```

## Como executar pelos scripts Python

### Preparar os dados

Os dados são preparados automaticamente pelo `train.py` caso os arquivos binários ainda não existam.

Também é possível executar a preparação diretamente:

```bash
python prepare.py
```

### Treinar um modelo do zero

```bash
python train.py scratch poesia
```

As opções de estilo são:

```text
texto
poesia
teatro
```

Exemplos:

```bash
python train.py scratch texto
python train.py scratch poesia
python train.py scratch teatro
```

### Continuar um treinamento

Para continuar um treinamento salvo anteriormente:

```bash
python train.py resume poesia
```

Nesse caso, deve existir o arquivo:

```text
checkpoints/poesia/ckpt.pt
```

### Treinar todos os estilos

```bash
python train_all.py
```

Esse comando treina `texto`, `poesia` e `teatro` em sequência utilizando checkpoints existentes.

## Checkpoints e resultados

Os checkpoints são armazenados em:

```text
checkpoints/<tipo_literario>/ckpt.pt
```

Os textos gerados são armazenados em:

```text
out/<tipo_literario>/ouput.txt
```

## Execução para avaliação

Embora o desenvolvimento e os testes também tenham sido realizados pelos scripts `.py`, o arquivo recomendado para avaliação é o notebook `machado.ipynb`.

O professor poderá:

1. Abrir o notebook;
2. Executar as células em ordem;
3. Selecionar o estilo literário;
4. Utilizar `scratch` para iniciar um treinamento novo ou `resume` para carregar um checkpoint;
5. Acompanhar os valores de loss;
6. Verificar o texto gerado.

Os scripts Python também estão disponíveis para facilitar a reprodução automatizada do treinamento.

## Observações

- O modelo utiliza GPU CUDA quando disponível.
- Na ausência de GPU, o treinamento é executado na CPU.
- A seed `1337` é utilizada para favorecer a reprodutibilidade.
- O modelo trabalha com caracteres, e não com palavras inteiras.
- O notebook e os scripts devem ser executados a partir da raiz do projeto.
- O modo `resume` exige que o checkpoint correspondente já exista.