from datetime import datetime
import os
import sqlite3
from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf

app = Flask(__name__)


# --- BANCO DE DADOS LOCAL (SQLite) ---
def init_db():
  conn = sqlite3.connect("dados_robo.db")
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS ativos_favoritos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE
        )
    """)
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS configuracoes (
            chave TEXT PRIMARY KEY,
            valor REAL
        )
    """)
  conn.commit()
  conn.close()


init_db()


def salvar_config_db(chave, valor):
  conn = sqlite3.connect("dados_robo.db")
  cursor = conn.cursor()
  cursor.execute(
      "INSERT OR REPLACE INTO configuracoes (chave, valor) VALUES (?, ?)",
      (chave, valor),
  )
  conn.commit()
  conn.close()


def carregar_config_db(chave, valor_padrao):
  conn = sqlite3.connect("dados_robo.db")
  cursor = conn.cursor()
  cursor.execute("SELECT valor FROM configuracoes WHERE chave = ?", (chave,))
  row = cursor.fetchone()
  conn.close()
  return row[0] if row else valor_padrao


def calcular_rsi(serie, janela=14):
  delta = serie.diff()
  ganho = delta.where(delta > 0, 0).rolling(window=janela).mean()
  perda = (-delta.where(delta < 0, 0)).rolling(window=janela).mean()
  rs = ganho / perda
  return 100 - (100 / (1 + rs))


# --- MOTOR DE BACKTEST E PREDIÇÃO ---
def executar_analise(
    ticker,
    capital_ini,
    risco_pct,
    max_alocacao,
    sl,
    tp,
    taxa_corretagem,
    confianca,
    interval="1d",
    horizonte=5,
):
  try:
    period = "10y" if interval == "1wk" else "7y"
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty or len(data) < 100:
      return None

    if isinstance(data.columns, pd.MultiIndex):
      data.columns = data.columns.droplevel(1)

    data["Retorno_1d"] = data["Close"].pct_change(1)
    data["Retorno_5d"] = data["Close"].pct_change(5)
    data["Retorno_20d"] = data["Close"].pct_change(20)
    data["Volatilidade"] = data["Retorno_1d"].rolling(20).std()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    data["Dist_SMA50"] = (data["Close"] - data["SMA_50"]) / data["SMA_50"]
    data["RSI"] = calcular_rsi(data["Close"], 14)
    data["Volume_Relativo"] = data["Volume"] / data["Volume"].rolling(20).mean()

    data["Preco_Futuro"] = data["Close"].shift(-horizonte)
    data["Target"] = np.where(
        (data["Preco_Futuro"] - data["Close"]) / data["Close"] >= tp, 1, 0
    )

    data_features = data.copy()
    data = data.dropna()

    features = [
        "Retorno_1d",
        "Retorno_5d",
        "Retorno_20d",
        "Volatilidade",
        "Dist_SMA50",
        "RSI",
        "Volume_Relativo",
    ]
    X = data[features]
    y = data["Target"]

    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    df_test = data.iloc[split_idx:].copy()

    if len(X_test) < 10:
      return None

    model = RandomForestClassifier(
        n_estimators=150, max_depth=6, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    df_test["Prob_Alta"] = model.predict_proba(X_test)[:, 1]

    # Predição atual
    X_ultimo = data_features[features].iloc[[-1]]
    prob_atual = model.predict_proba(X_ultimo)[0][1]
    data_atual_mercado = str(data_features.index[-1])[:10]
    preco_atual_mercado = float(data_features["Close"].iloc[-1])

    capital = capital_ini
    posicao = False
    preco_entrada = 0.0
    data_entrada = None
    dias_na_posicao = 0
    preco_stop_atual = 0.0
    preco_gain_atual = 0.0

    historico_trades = []
    capital_historico = []

    for i in range(len(df_test)):
      data_atual = str(df_test.index[i])[:10]
      high_atual = df_test["High"].iloc[i]
      low_atual = df_test["Low"].iloc[i]
      close_atual = df_test["Close"].iloc[i]
      prob = df_test["Prob_Alta"].iloc[i]

      if not posicao:
        if prob >= confianca:
          posicao = True
          preco_entrada = close_atual
          data_entrada = data_atual
          dias_na_posicao = 0

          preco_stop_atual = preco_entrada * (1 - sl)
          preco_gain_atual = preco_entrada * (1 + tp)

          risco_financeiro_desejado = capital * risco_pct
          tamanho_por_risco = risco_financeiro_desejado / sl
          tamanho_maximo_permitido = capital * max_alocacao
          tamanho_posicao = min(capital, tamanho_por_risco, tamanho_maximo_permitido)

          capital -= tamanho_posicao * taxa_corretagem
          capital_alocado = tamanho_posicao
      else:
        dias_na_posicao += 1
        bateu_stop = low_atual <= preco_stop_atual
        bateu_gain = high_atual >= preco_gain_atual

        if bateu_stop and bateu_gain:
          preco_saida = preco_stop_atual
          variacao = -sl
          posicao = False
        elif bateu_stop:
          preco_saida = preco_stop_atual
          variacao = -sl
          posicao = False
        elif bateu_gain:
          preco_saida = preco_gain_atual
          variacao = tp
          posicao = False
        elif dias_na_posicao >= horizonte:
          preco_saida = close_atual
          variacao = (preco_saida - preco_entrada) / preco_entrada
          posicao = False

        if not posicao:
          lucro_op = capital_alocado * variacao
          capital += lucro_op
          capital -= capital_alocado * taxa_corretagem

          historico_trades.append({
              "Data_Entrada": data_entrada,
              "Preco_Entrada": round(float(preco_entrada), 2),
              "Valor_Alocado": round(float(capital_alocado), 2),
              "Data_Saida": data_atual,
              "Preco_Saida": round(float(preco_saida), 2),
              "Lucro_Pct": round(float(variacao * 100), 2),
              "Lucro_Financeiro": round(float(lucro_op), 2),
          })

      capital_historico.append(float(capital))

    total_trades = len(historico_trades)
    vitorias = len([t for t in historico_trades if t["Lucro_Pct"] > 0])
    win_rate = (vitorias / total_trades * 100) if total_trades > 0 else 0.0
    lucro_liq = capital - capital_ini

    serie_cap = pd.Series(capital_historico)
    pico = serie_cap.cummax()
    drawdown = (serie_cap - pico) / pico
    max_drawdown = float(drawdown.min() * 100) if not drawdown.empty else 0.0

    return {
        "ticker": ticker,
        "capital_final": round(capital, 2),
        "lucro_liquido": round(lucro_liq, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": total_trades,
        "max_drawdown": round(max_drawdown, 2),
        "historico_trades": historico_trades,
        "sinal_atual": {
            "data": data_atual_mercado,
            "preco": preco_atual_mercado,
            "probabilidade": round(prob_atual * 100, 1),
            "ativo_recomendado": bool(prob_atual >= confianca),
        },
    }
  except Exception as e:
    print(f"Erro no backtest: {e}")
    return None


# --- ROTAS DO FLASK ---
@app.route("/")
def index():
  return render_template("index.html")


@app.route("/api/ativos", methods=["GET", "POST"])
def gerenciar_ativos():
  conn = sqlite3.connect("dados_robo.db")
  cursor = conn.cursor()
  if request.method == "POST":
    dados = request.json
    ticker = dados.get("ticker", "").upper()
    if ticker:
      try:
        cursor.execute(
            "INSERT OR IGNORE INTO ativos_favoritos (ticker) VALUES (?)",
            (ticker,),
        )
        conn.commit()
      except Exception:
        pass
  cursor.execute("SELECT ticker FROM ativos_favoritos")
  ativos = [row[0] for row in cursor.fetchall()]
  conn.close()
  if not ativos:
    ativos = ["SPY"]
  return jsonify(ativos)


@app.route("/api/analisar", methods=["POST"])
def analisar():
  params = request.json
  ticker = params.get("ticker", "SPY")
  capital = float(params.get("capital", 1000.0))
  risco_pct = float(params.get("risco", 1.0)) / 100
  max_aloc = float(params.get("max_alocacao", 20.0)) / 100
  confianca = float(params.get("confianca", 62.0)) / 100
  sl = float(params.get("stop_loss", 1.5)) / 100
  tp = float(params.get("take_profit", 3.0)) / 100
  taxa = float(params.get("taxa", 0.05)) / 100
  timeframe = params.get("timeframe", "Diário (1D)")

  yf_interval = "1d" if "Diário" in timeframe else "1wk"

  resultado = executar_analise(
      ticker,
      capital,
      risco_pct,
      max_aloc,
      sl,
      tp,
      taxa,
      confianca,
      interval=yf_interval,
  )
  if not resultado:
    return jsonify({"erro": "Falha ao processar dados para este ativo."}), 400

  return jsonify(resultado)


if __name__ == "__main__":
  print("🐝 Servidor Vespa Trade rodando em http://127.0.0.1:5000")
  app.run(debug=True, port=5000)