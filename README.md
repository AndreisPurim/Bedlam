# Metadata privacy and obliviousness in distributed learning via a bulletin board: a proof-of-concept

Este repositório implementa uma prova de conceito de aprendizado distribuído com foco em privacidade de metadados. A ideia central é que proteger apenas o conteúdo das mensagens não impede vazamentos por meio de padrões de comunicação, como tempo, volume, frequência e associação entre remetente e destinatário. Para lidar com isso, o sistema introduz um bulletin board, chamado BEDLAM (Board-Enabled Distributed Learning via Anonymous Messaging), por meio do qual as mensagens são publicadas e recuperadas sem comunicação direta entre as partes. A proposta foi pensada para ser compatível com diferentes formas de aprendizado distribuído, em especial split learning e federated learning.

Na implementação atual (toy), o repositório inclui principalmente um protótipo de split learning com MNIST e Fashion-MNIST, em que partes do modelo ficam distribuídas entre clientes e pares intermediários, e toda a comunicação passa pelo quadro. O sistema também inclui mecanismos opcionais como pseudônimos, rotação de chaves, preenchimento de mensagens, modos de agrupamento para reduzir certos vazamentos de metadados e um caminho simplificado de PIR. Além disso, há uma versão básica de federated learning usando a mesma abstração de comunicação, bem como scripts de log, métricas e geração de figuras. Como se trata de uma prova de conceito, o próprio repositório também deixa claras suas limitações atuais, como vazamento residual de tempo e volume, visibilidade de metadados no quadro e uso de mecanismos criptográficos ainda simplificados.

**Título do artigo:** Metadata privacy and obliviousness in distributed learning via a bulletin board: a proof-of-concept

**Resumo:** While many works study payload privacy in distributed learning, metadata and communication patterns between peers can still reveal sensitive information. We treat metadata privacy as a distinct problem, and first introduces a risk and threat model for metadata leakage in distributed learning systems. Building on this analysis, we present a bulletin board-based communication architecture in which peers exchange activations, gradients, and model updates under configurable privacy and efficiency settings. We implement a proof-of-concept code of the proposed design and evaluate it on small-scale distributed learning workloads, which is available on Github

# Estrutura do readme.md

Este README está organizado nas seguintes seções:

- Título projeto: título e resumo do artigo/artefato.
- Estrutura do readme.md: descrição da organização deste documento.
- Selos Considerados: selos alvo na avaliação.
- Informações básicas: ambiente de execução e componentes necessários.
- Dependências: bibliotecas e versões relevantes.
- Preocupações com segurança: riscos e cuidados ao executar.
- Instalação: passos para instalar.
- Teste mínimo: smoke tests e validação rápida.
- Experimentos: passo a passo para reproduzir resultados e rodar scripts principais.
- LICENSE: licença do projeto.

# Selos Considerados

Os selos considerados são: Disponíveis e Funcionais.

# Informações básicas

- Arquitetura implementada: split learning (M1/M3 nos clientes e M2 em peers separados) e baseline federado simples usando o mesmo board como relay.
- Comunicação: gRPC entre peers e board; mensagens podem ser cifradas com AES-GCM e padding configurável.
- Datasets: MNIST e Fashion-MNIST (via keras.datasets, baixados automaticamente no primeiro run).
- Execução: Python + Ray para orquestração local de múltiplos peers. GPU é opcional.

# Dependências

As versões utilizadas estão em requirements.txt. Principais dependências:

- tensorflow==2.20.0, keras==3.11.3
- ray==2.50.0
- grpcio==1.75.1, grpcio-tools==1.75.1
- cryptography==46.0.3, phe==1.5.0
- numpy==2.3.3, pandas==2.3.3, matplotlib==3.10.7
- PyYAML==6.0.3

# Preocupações com segurança

A execução dessa PoC não oferece risco para os avaliadores.

# Instalação

~~~bash
git clone https://github.com/AndreisPurim/Bedlam.git
cd Bedlam
python3 -m venv bedlam_env
source bedlam_env/bin/activate
pip install -r requirements.txt
~~~

# Teste mínimo

Smoke tests rápidos (não treinam modelo completo):

~~~bash
python scripts/smoke_test.py      # encode/decode + replay
python scripts/smoke_board.py     # board: fila/pool/bucket
~~~

Teste federado mínimo:

~~~bash
python scripts/smoke_federated.py
~~~

Resultados esperados:

- smoke_test.py imprime: smoke ok: all tests passed.
- smoke_board.py imprime: smoke ok.
- smoke_federated.py finaliza sem erro e gera um run temporário.

# Experimentos

Os scripts abaixo automatizam execuções. Para reduzir tempo, use --max-steps e --timeout. Os resultados são gravados em runs/ e logs em arquivos por execução.

## Reivindicação #1

Objetivo: comparar arquiteturas de split learning (vanilla, board-blind, double-blind) executando end-to-end.

Arquivos/Configuração: scripts/paper_experiments_v2.py usa config.yaml como base e aplica os flags na execução. Não é necessário editar config.yaml se usar o script.

Comando:

~~~bash
python scripts/paper_experiments_v2.py --arches-split vanilla-split,board-blind,double-blind --datasets fashion-mnist --clients 1 --m2 1 --epochs 2 --max-steps 0
~~~

Tempo esperado: variável conforme hardware e parâmetros. Para execução rápida, reduza epochs ou use --max-steps e --timeout.

Recursos: CPU; RAM proporcional ao número de peers. O diretório runs/ cresce conforme o número de execuções e logs.

Resultados esperados:

- Execuções em runs/paper_v2_* (nome pode variar).
- board_metrics.csv por run (tamanho de filas/pools/buckets/PIR).
- client_*.csv com métricas por cliente; perf_steps_*.csv para vanilla-split e board-blind quando enable_perf_metrics=true.

# LICENSE

MIT. Veja LICENSE para detalhes.
