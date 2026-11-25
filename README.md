# Supressão de oscilação de carga em pontes rolantes: input shaping × LQR

Código de simulação do Trabalho de Conclusão de Curso de **Pedro Otávio Malheiros Varella Normando**
(Engenharia de Controle e Automação, Centro Universitário IBMR, 2026).

O trabalho compara, por simulação numérica, cinco formas de movimentar o carro de uma ponte rolante
com carga suspensa:

| Estratégia | Descrição |
|---|---|
| Sem controle | Perfil trapezoidal de velocidade aplicado diretamente ao carro |
| ZV | Perfil convoluído com o modelador *Zero Vibration* (2 impulsos) |
| ZVD | Perfil convoluído com o modelador *Zero Vibration and Derivative* (3 impulsos) |
| LQR | Regulador linear quadrático com seguimento de trajetória (realimentação de posição, velocidade, ângulo e velocidade angular) |
| LQR-V | O mesmo LQR com limitador de velocidade (\|v\| ≤ 0,5 m/s) |

## Modelo

Carro acionado por inversor de frequência (a aceleração do carro é a entrada *u*) e carga pontual
suspensa por cabo rígido de comprimento *L*, sem atrito nem amortecimento:

```
x''       = u
L·θ''     = −g·sen(θ) − u·cos(θ)
```

O modelo não linear acima é usado em todas as simulações. O modelo linearizado em torno de θ = 0
é usado apenas no projeto do LQR (função `ganho_lqr`).

## Como executar

Requer Python 3.9 ou mais recente.

```bash
python3 -m pip install -r requirements.txt
python3 simulacao_ponte_rolante.py
```

A execução leva cerca de 3 minutos. O script imprime as métricas no terminal e gera:

- `figuras/`: esquema do sistema, fluxograma, perfis de referência, resposta nominal,
  comparação LQR × LQR-V e robustez ao comprimento do cabo;
- `resultados/`: séries temporais de cada estratégia (`serie_*.csv`), métricas nominais,
  oscilação residual para cada comprimento de cabo e varredura de sintonia do LQR.

Os parâmetros principais (`L_NOM`, `DIST`, `V_MAX`, `A_MAX`, `A_SAT`, `Q_LQR`, `R_LQR`, `K_LIM`)
ficam no início do arquivo.

## Métricas

- **θmax**: maior |θ| durante toda a simulação.
- **θres**: maior |θ| de 29,67 s a 45 s (8 s após o fim do perfil nominal), igual para todas as estratégias.
- **tc**: primeiro instante a partir do qual o carro fica a menos de 1 cm do destino e |θ| < 0,25° até o fim da simulação.
- **|u|max**: maior aceleração comandada (solicitação do acionamento, não consumo de energia).
- **Sobrepasso** e **vmax**: maior ultrapassagem da posição final e maior velocidade do carro.

## Versões usadas no trabalho

Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1 e Matplotlib 3.9.4.

## Como citar

NORMANDO, P. O. M. V. **ponte-rolante-input-shaping-lqr**: código de simulação do controle
anti-balanço de pontes rolantes por input shaping e LQR. [S. l.]: GitHub, 2026.

## Licença

MIT. Veja o arquivo `LICENSE`.
