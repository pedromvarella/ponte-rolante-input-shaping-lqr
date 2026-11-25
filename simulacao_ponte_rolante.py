"""
Simulação de ponte rolante: supressão de oscilação da carga.
Compara: (1) comando sem modelagem, (2) input shaping ZV, (3) input shaping ZVD,
(4) LQR com seguimento de trajetória e (5) LQR com limitação de velocidade (LQR-V).

Modelo: carro acionado por inversor de frequência (a aceleração do carro é a entrada u),
carga pontual suspensa por cabo rígido de comprimento L, sem amortecimento.
    x'' = u
    L*theta'' = -g*sin(theta) - u*cos(theta)

Uso:  python simulacao_ponte_rolante.py
Gera as figuras em ./figuras, as séries temporais e as tabelas em ./resultados
e imprime as métricas no terminal.

Versões usadas no trabalho: Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, Matplotlib 3.9.4.
"""
import csv
import locale
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

try:
    locale.setlocale(locale.LC_NUMERIC, "pt_BR.UTF-8")
    plt.rcParams["axes.formatter.use_locale"] = True
except locale.Error:
    pass

plt.rcParams.update({"font.family": "serif", "font.size": 12, "legend.fontsize": 10, "figure.dpi": 150})

AQUI = os.path.dirname(os.path.abspath(__file__))
PASTA_FIG = os.path.join(AQUI, "figuras")
PASTA_RES = os.path.join(AQUI, "resultados")
os.makedirs(PASTA_FIG, exist_ok=True)
os.makedirs(PASTA_RES, exist_ok=True)

# ---------------------------------------------------------------- parâmetros
G = 9.81          # gravidade [m/s²]
L_NOM = 5.0       # comprimento nominal do cabo [m]
DIST = 10.0       # deslocamento do carro [m]
V_MAX = 0.5       # velocidade máxima do carro [m/s]
A_MAX = 0.3       # aceleração do perfil de referência [m/s²]
A_SAT = 0.6       # limite de aceleração do acionamento [m/s²]
K_LIM = 10.0      # ganho do limitador de velocidade do LQR-V [1/s]
T_FIM = 45.0      # tempo de simulação [s]
DT = 0.005        # passo de amostragem dos resultados [s]

# Pesos do LQR escolhidos na varredura de sintonia (ver varredura_sintonia)
Q_LQR = np.diag([1.0, 1.0, 50.0, 50.0])
R_LQR = np.array([[1.0]])

# Critério de chegada: carro a menos de 1 cm do destino e ângulo menor que 0,25°,
# mantidos até o fim da simulação
TOL_POS = 0.01
TOL_ANG = np.deg2rad(0.25)


# ---------------------------------------------------------------- referência
def perfil_trapezoidal(t):
    """Aceleração, velocidade e posição de um perfil trapezoidal de velocidade
    (acelera, cruzeiro, desacelera), calculadas de forma analítica."""
    t_ac = V_MAX / A_MAX
    t_cr = DIST / V_MAX - t_ac
    t1, t2, t3 = t_ac, t_ac + t_cr, 2 * t_ac + t_cr
    t = np.asarray(t, dtype=float)
    a, v, x = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    m = (t >= 0) & (t < t1)
    a[m], v[m], x[m] = A_MAX, A_MAX * t[m], A_MAX * t[m] ** 2 / 2
    x1 = A_MAX * t1 ** 2 / 2
    m = (t >= t1) & (t < t2)
    v[m], x[m] = V_MAX, x1 + V_MAX * (t[m] - t1)
    x2 = x1 + V_MAX * t_cr
    m = (t >= t2) & (t < t3)
    tau = t[m] - t2
    a[m], v[m], x[m] = -A_MAX, V_MAX - A_MAX * tau, x2 + V_MAX * tau - A_MAX * tau ** 2 / 2
    m = t >= t3
    x[m] = DIST
    return a, v, x


def modelador(tipo, L):
    """Amplitudes e instantes dos impulsos do modelador (sistema sem amortecimento)."""
    wn = np.sqrt(G / L)
    meio_periodo = np.pi / wn
    if tipo == "ZV":
        return np.array([0.5, 0.5]), np.array([0.0, meio_periodo])
    if tipo == "ZVD":
        return np.array([0.25, 0.5, 0.25]), np.array([0.0, meio_periodo, 2 * meio_periodo])
    return np.array([1.0]), np.array([0.0])


def trajetoria_referencia(t, tipo, L_projeto):
    """Referência modelada: convolução do perfil trapezoidal com os impulsos do modelador."""
    amp, tempos = modelador(tipo, L_projeto)
    partes = [perfil_trapezoidal(np.asarray(t, dtype=float) - ti) for ti in tempos]
    a = sum(A * p[0] for A, p in zip(amp, partes))
    v = sum(A * p[1] for A, p in zip(amp, partes))
    x = sum(A * p[2] for A, p in zip(amp, partes))
    return a, v, x


# ---------------------------------------------------------------- LQR
def ganho_lqr(L, Q=Q_LQR, R=R_LQR):
    """Ganho LQR do modelo linearizado em torno de theta = 0. Estados: [x, x', theta, theta']."""
    A = np.array([[0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, -G / L, 0]])
    B = np.array([[0], [1], [0], [-1 / L]])
    P = solve_continuous_are(A, B, Q, R)
    return (np.linalg.solve(R, B.T @ P)).ravel()


# ---------------------------------------------------------------- simulação
def simular(estrategia, L_real=L_NOM, L_projeto=L_NOM, K=None):
    """estrategia: 'Sem controle', 'ZV', 'ZVD', 'LQR' ou 'LQR-V'.
    L_real é o comprimento do cabo no modelo; L_projeto, o usado no projeto do controlador.
    K permite simular o LQR com outro ganho (varredura de sintonia).
    Condição inicial: carro parado em x = 0 e carga em repouso na vertical."""
    tempo = np.arange(0, T_FIM + DT, DT)
    tipo_ref = estrategia if estrategia in ("ZV", "ZVD") else "nenhum"
    malha_fechada = estrategia in ("LQR", "LQR-V")
    if malha_fechada and K is None:
        K = ganho_lqr(L_projeto)

    def entrada(t, s):
        """Aceleração comandada ao carro, calculada a cada passo do integrador."""
        a_ref, v_ref, x_ref = (float(c) for c in trajetoria_referencia(t, tipo_ref, L_projeto))
        if not malha_fechada:
            u = a_ref
        else:
            erro = np.array([s[0] - x_ref, s[1] - v_ref, s[2], s[3]])
            u = a_ref - K @ erro
            if estrategia == "LQR-V":
                # Limitador de velocidade: perto de ±V_MAX, a aceleração é reduzida
                # para que |v| não ultrapasse o limite do perfil de referência.
                u = min(u, K_LIM * (V_MAX - s[1]))
                u = max(u, K_LIM * (-V_MAX - s[1]))
        return float(np.clip(u, -A_SAT, A_SAT))

    def dinamica(t, s):
        x, v, th, w = s
        u = entrada(t, s)
        return [v, u, w, (-G * np.sin(th) - u * np.cos(th)) / L_real]

    sol = solve_ivp(dinamica, (0, T_FIM), [0, 0, 0, 0], t_eval=tempo,
                    max_step=DT, rtol=1e-8, atol=1e-10)
    u = np.array([entrada(t, s) for t, s in zip(sol.t, sol.y.T)])
    return {"t": sol.t, "x": sol.y[0], "v": sol.y[1], "theta": sol.y[2], "u": u}


def metricas(r):
    t, x, th, u = r["t"], r["x"], r["theta"], r["u"]
    # Tempo de chegada: último instante fora das tolerâncias + 1 amostra, ou seja,
    # a partir dele as duas condições permanecem satisfeitas até o fim da simulação.
    fora = (np.abs(x - DIST) > TOL_POS) | (np.abs(th) > TOL_ANG)
    idx = np.where(fora)[0]
    t_chegada = t[idx[-1] + 1] if len(idx) and idx[-1] + 1 < len(t) else (0.0 if not len(idx) else np.nan)
    # Oscilação residual: maior |theta| na mesma janela para todas as estratégias,
    # de 8 s após o fim do perfil nominal (21,67 s) até o fim da simulação (45 s).
    t_fim_mov = DIST / V_MAX + V_MAX / A_MAX
    residual = np.max(np.abs(th[t > t_fim_mov + 8.0]))
    return {
        "theta_max_graus": np.rad2deg(np.max(np.abs(th))),
        "residual_graus": np.rad2deg(residual),
        "t_chegada_s": t_chegada,
        "u_max": np.max(np.abs(u)),
        "sobrepasso_m": max(0.0, np.max(x) - DIST),
        "v_max": np.max(np.abs(r["v"])),
    }


ESTRATEGIAS = ["Sem controle", "ZV", "ZVD", "LQR", "LQR-V"]
CORES = {"Sem controle": "#888888", "ZV": "#1f77b4", "ZVD": "#2ca02c", "LQR": "#d62728", "LQR-V": "#9467bd"}
ESTILOS = {"Sem controle": "-", "ZV": "--", "ZVD": "-.", "LQR": "-", "LQR-V": ":"}


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG, nome), dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------- figuras
def figura_esquema():
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot([0.3, 5.7], [3.2, 3.2], color="k", lw=3)
    ax.text(5.75, 3.2, "viga", va="center", fontsize=13)
    ax.add_patch(Rectangle((2.2, 3.25), 1.2, 0.45, fc="#d9d9d9", ec="k"))
    ax.text(2.8, 3.47, "carro", ha="center", va="center", fontsize=12)
    th = np.deg2rad(22)
    px, py = 2.8 + 2.2 * np.sin(th), 3.25 - 2.2 * np.cos(th)
    ax.plot([2.8, px], [3.25, py], color="k", lw=1.3)
    ax.add_patch(Circle((px, py), 0.18, fc="#4a4a4a", ec="k"))
    ax.text(px + 0.3, py - 0.05, "carga (m)", va="center", fontsize=13)
    ax.plot([2.8, 2.8], [3.25, 0.9], color="k", lw=0.8, ls=":")
    arco = np.linspace(-np.pi / 2, -np.pi / 2 + th, 30)
    ax.plot(2.8 + 0.9 * np.cos(arco), 3.25 + 0.9 * np.sin(arco), color="k", lw=0.8)
    ax.text(2.97, 2.12, r"$\theta$", fontsize=15)
    ax.text(3.35, 2.45, "L", fontsize=15)
    ax.add_patch(FancyArrowPatch((0.6, 3.85), (2.2, 3.85), arrowstyle="->", mutation_scale=12))
    ax.text(1.3, 3.95, "x", fontsize=15)
    ax.add_patch(FancyArrowPatch((3.5, 3.47), (4.6, 3.47), arrowstyle="->", mutation_scale=12, color="#d62728"))
    ax.text(3.75, 3.6, r"$u=\ddot{x}$", fontsize=14, color="#d62728")
    ax.annotate("g", xy=(6.1, 1.0), xytext=(6.1, 1.8), arrowprops=dict(arrowstyle="->"), ha="center", fontsize=14)
    ax.set_xlim(0, 6.9)
    ax.set_ylim(0.5, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")
    salvar(fig, "fig1_esquema_ponte_rolante.png")


def figura_fluxograma():
    etapas = ["Revisão da\nliteratura", "Modelagem\nmatemática", "Projeto dos\ncontrola-\ndores (ZV,\nZVD, LQR)",
              "Simulação\nnumérica\n(Python)", "Cálculo das\nmétricas", "Análise de\nrobustez e\ncomparação"]
    fig, ax = plt.subplots(figsize=(7.4, 2.0))
    w, h, gap = 1.12, 1.15, 0.16
    for i, txt in enumerate(etapas):
        x0 = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x0, 0), w, h, boxstyle="round,pad=0.03", fc="#e8eef7", ec="#2f4b7c"))
        ax.text(x0 + w / 2, h / 2, txt, ha="center", va="center", fontsize=11.5)
        if i < len(etapas) - 1:
            ax.add_patch(FancyArrowPatch((x0 + w + 0.02, h / 2), (x0 + w + gap - 0.02, h / 2),
                                         arrowstyle="->", mutation_scale=10))
    ax.set_xlim(-0.1, len(etapas) * (w + gap))
    ax.set_ylim(-0.1, h + 0.1)
    ax.axis("off")
    salvar(fig, "fig2_fluxograma_metodologia.png")


def figura_comandos():
    tempo = np.arange(0, 30, DT)
    fig, ax = plt.subplots(2, 1, figsize=(6.2, 4.2), sharex=True)
    for tipo, rotulo in [("nenhum", "Sem modelagem"), ("ZV", "ZV"), ("ZVD", "ZVD")]:
        nome = "Sem controle" if tipo == "nenhum" else tipo
        a, v, _ = trajetoria_referencia(tempo, tipo, L_NOM)
        ax[0].plot(tempo, a, ESTILOS[nome], color=CORES[nome], label=rotulo)
        ax[1].plot(tempo, v, ESTILOS[nome], color=CORES[nome], label=rotulo)
    ax[0].set_ylabel("Aceleração (m/s²)")
    ax[1].set_ylabel("Velocidade (m/s)")
    ax[1].set_xlabel("Tempo (s)")
    ax[0].legend(ncol=3)
    for a in ax:
        a.grid(alpha=0.3)
    salvar(fig, "fig3_comandos_modelados.png")


def figura_resposta(resultados):
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)
    for nome in ["Sem controle", "ZV", "ZVD", "LQR"]:
        r = resultados[nome]
        ax[0].plot(r["t"], r["x"], ESTILOS[nome], color=CORES[nome], label=nome, lw=1.2)
        ax[1].plot(r["t"], np.rad2deg(r["theta"]), ESTILOS[nome], color=CORES[nome], label=nome, lw=1.2)
    ax[0].set_ylabel("Posição do carro (m)")
    ax[1].set_ylabel("Ângulo da carga (°)")
    ax[1].set_xlabel("Tempo (s)")
    ax[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
    ax[1].set_xlim(0, T_FIM)
    salvar(fig, "fig4_resposta_nominal.png")


def figura_robustez():
    Ls = np.linspace(3.0, 7.0, 17)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    curvas = {}
    for nome in ESTRATEGIAS:
        res = [metricas(simular(nome, L_real=Lr))["residual_graus"] for Lr in Ls]
        curvas[nome] = res
        ax.plot(Ls, res, ESTILOS[nome], marker="o", ms=3, color=CORES[nome], label=nome, lw=1.2)
    ax.axvline(L_NOM, color="k", lw=0.6, ls=":")
    ax.set_xlabel("Comprimento real do cabo (m), com projeto para L = 5 m")
    ax.set_ylabel("Oscilação residual (°)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1.0)
    salvar(fig, "fig5_robustez_comprimento_cabo.png")
    return Ls, curvas


def figura_lqr_limitacao(resultados):
    """Compara o LQR original e o LQR-V com o perfil de referência."""
    t = resultados["LQR"]["t"]
    _, v_ref, x_ref = trajetoria_referencia(t, "nenhum", L_NOM)
    fig, ax = plt.subplots(3, 1, figsize=(6.4, 6.4), sharex=True)
    ax[0].axhline(0, color="k", lw=0.9, ls="--", label="Referência")
    ax[1].plot(t, v_ref, color="k", lw=0.9, ls="--", label="Referência")
    for nome, rot in [("LQR", "LQR"), ("LQR-V", "LQR-V (|v| ≤ 0,5 m/s)")]:
        r = resultados[nome]
        ax[0].plot(t, r["x"] - x_ref, ESTILOS[nome], color=CORES[nome], lw=1.4, label=rot)
        ax[1].plot(t, r["v"], ESTILOS[nome], color=CORES[nome], lw=1.4, label=rot)
        ax[2].plot(t, np.rad2deg(r["theta"]), ESTILOS[nome], color=CORES[nome], lw=1.4, label=rot)
    ax[0].set_ylabel("x − x$_{ref}$ (m)")
    ax[1].set_ylabel("Velocidade (m/s)")
    ax[2].set_ylabel("Ângulo (°)")
    ax[2].set_xlabel("Tempo (s)")
    ax[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
    ax[2].set_xlim(0, 35)
    salvar(fig, "fig6_lqr_limitacao_velocidade.png")


def varredura_sintonia():
    """Varredura dos pesos do LQR (R = 1): q_x = q_v e q_theta = q_omega.
    Critérios: menor oscilação residual e menor tempo de chegada em L = 5 m,
    sem saturar o acionamento, e boa robustez no pior caso (L = 7 m)."""
    casos = [(1, 1), (1, 10), (1, 20), (1, 50), (1, 100), (1, 200), (10, 50), (10, 200)]
    linhas = []
    for qx, qth in casos:
        K = ganho_lqr(L_NOM, Q=np.diag([qx, 1.0, qth, qth]))
        m = metricas(simular("LQR", K=K))
        m7 = metricas(simular("LQR", L_real=7.0, K=K))
        linhas.append([qx, qth, *np.round(K, 2), m["theta_max_graus"], m["residual_graus"], m["t_chegada_s"],
                       m["u_max"], m["sobrepasso_m"], m["v_max"], m7["residual_graus"]])
        print("q_x=%-3g q_th=%-4g | θmax %.2f | res %.3f | tc %.2f | sobrep. %.3f | vmax %.3f | res(L=7) %.2f"
              % (qx, qth, m["theta_max_graus"], m["residual_graus"], m["t_chegada_s"], m["sobrepasso_m"],
                 m["v_max"], m7["residual_graus"]))
    with open(os.path.join(PASTA_RES, "varredura_sintonia_lqr.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q_x", "q_theta", "K1", "K2", "K3", "K4", "theta_max_graus", "residual_graus",
                    "t_chegada_s", "u_max", "sobrepasso_m", "v_max", "residual_L7_graus"])
        w.writerows(linhas)


def exportar(resultados, curvas, Ls):
    """Grava as séries temporais nominais, as métricas e a robustez em CSV."""
    for nome, r in resultados.items():
        arq = os.path.join(PASTA_RES, "serie_%s.csv" % nome.replace(" ", "_").lower())
        with open(arq, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "x_m", "v_m_s", "theta_rad", "u_m_s2"])
            w.writerows(np.column_stack([r["t"], r["x"], r["v"], r["theta"], r["u"]]).round(6))
    with open(os.path.join(PASTA_RES, "metricas_nominais.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["estrategia", "theta_max_graus", "residual_graus", "t_chegada_s", "u_max", "sobrepasso_m", "v_max"])
        for nome, r in resultados.items():
            m = metricas(r)
            w.writerow([nome] + [round(float(m[k]), 6) for k in ("theta_max_graus", "residual_graus", "t_chegada_s",
                                                                  "u_max", "sobrepasso_m", "v_max")])
    with open(os.path.join(PASTA_RES, "robustez_residual_graus.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["L_real_m"] + ESTRATEGIAS)
        for i, Lr in enumerate(Ls):
            w.writerow([Lr] + [round(float(curvas[n][i]), 6) for n in ESTRATEGIAS])


if __name__ == "__main__":
    print("Ganho LQR (L = 5 m):", np.round(ganho_lqr(L_NOM), 3))
    print("Frequência natural: %.3f rad/s | Período: %.2f s" % (np.sqrt(G / L_NOM), 2 * np.pi * np.sqrt(L_NOM / G)))
    resultados = {nome: simular(nome) for nome in ESTRATEGIAS}
    print("\nEstratégia     | θ máx (°) | residual (°) | chegada (s) | |u| máx (m/s²) | sobrep. (m) | v máx (m/s)")
    for nome, r in resultados.items():
        m = metricas(r)
        print("%-14s | %9.3f | %12.4f | %11.2f | %14.3f | %11.3f | %11.3f" % (
            nome, m["theta_max_graus"], m["residual_graus"], m["t_chegada_s"], m["u_max"],
            m["sobrepasso_m"], m["v_max"]))
    figura_esquema()
    figura_fluxograma()
    figura_comandos()
    figura_resposta(resultados)
    figura_lqr_limitacao(resultados)
    Ls, curvas = figura_robustez()
    print("\nResidual (°) para L = 3, 4, 5, 6 e 7 m:")
    for nome in ESTRATEGIAS:
        print("%-14s " % nome + "  ".join("%.3f" % curvas[nome][i] for i in (0, 4, 8, 12, 16))
              + "   (faixa: %.2f a %.2f)" % (min(curvas[nome]), max(curvas[nome])))
    print("\nVarredura de sintonia do LQR:")
    varredura_sintonia()
    exportar(resultados, curvas, Ls)
    print("\nFiguras salvas em", PASTA_FIG)
    print("Resultados (CSV) salvos em", PASTA_RES)
