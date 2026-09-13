# Dashboard — Violência contra a Mulher no Paraná (2025)

Dashboard em **Python + Streamlit** para exposição e comparação de dados públicos sobre violência contra a mulher no Paraná em 2025.

## O que o projeto mostra

- panorama geral dos dados de 2025;
- mapa municipal interativo do Paraná;
- comparação explícita entre **LESFEM/UEL** e **SENASP/Sinesp VDE**;
- aba especial para **estupros contra mulheres**, em nível estadual e mensal;
- leitura territorial com CAPS, Delegacia da Mulher e Regionais de Saúde;
- consulta das bases completas e download em CSV;
- aba de metodologia e fontes.

## Estrutura

```text
dashboard_violencia_mulher_pr_2025/
├─ app.py
├─ requirements.txt
├─ .gitignore
├─ .streamlit/
│  └─ config.toml
├─ assets/
│  └─ paleta.png
└─ dados/
   ├─ base_municipal_2025.csv
   ├─ feminicidio_senasp_2025.csv
   └─ estupro_senasp_2025.csv
```

## Rodar no VS Code — Windows / PowerShell

Abra a pasta do projeto no VS Code e, no terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

O Streamlit normalmente abrirá o navegador automaticamente. Se não abrir, acesse o endereço informado no terminal, geralmente `http://localhost:8501`.

## Publicar no Streamlit Community Cloud

1. Suba esta pasta para um repositório no GitHub.
2. No Streamlit Community Cloud, selecione o repositório.
3. Em **Main file path**, informe `app.py`.
4. Faça o deploy.

O mapa baixa a geometria municipal do Paraná em tempo de execução, sem chave de API.

## Fontes informadas no projeto

- IPARDES — Base de Dados do Estado: https://www.ipardes.pr.gov.br/Pagina/Base-de-Dados-do-Estado
- SENASP / Sinesp VDE: https://www.gov.br/mj/pt-br/assuntos/sua-seguranca/seguranca-publica/estatistica/dados-nacionais-1/base-de-dados-e-notas-metodologicas-dos-gestores-estaduais-sinesp-vde-2022-e-2023
- LESFEM/UEL: https://sites.uel.br/lesfem/base-de-dados-e-nota-metodologica/
- SESP-PR / CAPE: https://www.seguranca.pr.gov.br/CAPE

## Observação metodológica importante

LESFEM e SENASP são mantidos como **fontes distintas**. O dashboard compara os valores, mas não soma as duas bases nem presume que uma substitua a outra. A base de estupro enviada não contém município, portanto o projeto não inventa uma distribuição municipal para esses dados.
