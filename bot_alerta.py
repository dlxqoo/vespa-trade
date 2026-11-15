import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import requests
import yfinance as yf

# --- SUAS NOVAS CREDENCIAIS DO TELEGRAM (VESPA TRADE) ---
TELEGRAM_TOKEN = "9010278583:AAGBfLSgDQP7jVb9JwnvukfSG79jDpEzny0"
CHAT_ID = "5939626001"


def enviar_alerta_telegram(mensagem):
  url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
  payload = {"chat_id": CHAT_ID, "text": mensagem, "parse_mode": "Markdown"}
  try:
    response = requests.post(url, json=payload)
    if response.status_code != 200:
      print(f"Erro ao enviar Telegram: {response.text}")
    else:
      print("🐝 Alerta enviado com sucesso para o Telegram!")
  except Exception as e:
    print(f"Erro de conexão com Telegram: {e}")


def carregar_config_db(chave, valor_padrao):
  if not os.path.exists("dados_robo.db"):
    return valor_padrao
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


def varredura_vespa_trade():
  print("🐝 [Vespa Trade] Iniciando varredura automatizada de mercado...")

  if not os.path.exists("dados_robo.db"):
    print("Banco de dados 'dados_robo.db' não encontrado.")
    return

  # Carrega os parâmetros sincronizados diretamente do painel web
  capital_inicial = carregar_config_db("capital_inicial", 10000.0)
  risco_por_trade_pct = carregar_config_db("risco_por_trade", 1.0) / 100
  max_alocacao_pct = carregar_config_db("max_alocacao", 20.0) / 100
  limiar_confianca = carregar_config_db("confianca", 62.0) / 100
  stop_loss = carregar_config_db("stop_loss", 1.5) / 100
  take_profit = carregar_config_db("take_profit", 3.0) / 100
  horizonte = 5

  print(
      f"⚙️ Configurações aplicadas -> Confiança Mínima:"
      f" {limiar_confianca*100:.1f}% | Stop Loss: {stop_loss*100}% | Take"
      f" Profit: {take_profit*100}%"
  )

  conn = sqlite3.connect("dados_robo.db")
  df_ativos = pd.read_sql("SELECT ticker FROM ativos_favoritos", conn)
  conn.close()

  lista_ativos = df_ativos["ticker"].tolist()
  if not lista_ativos:
    print("Nenhum ativo cadastrado no banco de dados.")
    return

  sinais_encontrados = []
  resumo_status = []

  for ticker in lista_ativos:
    try:
      print(f"Analisando {ticker}...")
      data = yf.download(ticker, period="7y", interval="1d", progress=False)
      if data.empty or len(data) < 100:
        continue

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
          (data["Preco_Futuro"] - data["Close"]) / data["Close"] >= take_profit,
          1,
          0,
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
      model = RandomForestClassifier(
          n_estimators=150, max_depth=6, random_state=42, class_weight="balanced"
      )
      model.fit(X.iloc[:split_idx], y.iloc[:split_idx])

      X_test = X.iloc[split_idx:]
      df_test = data.iloc[split_idx:].copy()
      df_test["Prob_Alta"] = model.predict_proba(X_test)[:, 1]

      # Cálculo do Win Rate Histórico Recente
      trades_hist = 0
      vitorias_hist = 0
      posicao = False
      p_ent = 0.0
      dias_p = 0
      p_stop = 0.0
      p_gain = 0.0

      for i in range(len(df_test)):
        h_at = df_test["High"].iloc[i]
        l_at = df_test["Low"].iloc[i]
        c_at = df_test["Close"].iloc[i]
        pr = df_test["Prob_Alta"].iloc[i]

        if not posicao:
          if pr >= limiar_confianca:
            posicao = True
            p_ent = c_at
            dias_p = 0
            p_stop = p_ent * (1 - stop_loss)
            p_gain = p_ent * (1 + take_profit)
        else:
          dias_p += 1
          b_stop = l_at <= p_stop
          b_gain = h_at >= p_gain

          if b_stop or b_gain or dias_p >= horizonte:
            trades_hist += 1
            if b_gain and not b_stop:
              vitorias_hist += 1
            posicao = False

      win_rate_hist = (
          (vitorias_hist / trades_hist * 100) if trades_hist > 0 else 0.0
      )

      # Predição Atual
      X_ultimo = data_features[features].iloc[[-1]]
      prob_atual = model.predict_proba(X_ultimo)[0][1]
      preco_atual = data_features["Close"].iloc[-1]
      data_ref = str(data_features.index[-1])[:10]
      prob_pct = prob_atual * 100

      resumo_status.append(
          f"• `{ticker}`: **{prob_pct:.1f}%**"
          + (" (🚀 Compra!)" if prob_atual >= limiar_confianca else "")
      )

      if trades_hist > 0:
        score_operacional = (prob_pct * 0.4) + (min(win_rate_hist, 100) * 0.6)
      else:
        score_operacional = prob_pct * 0.8

      # Verifica se gera sinal de compra
      if prob_atual >= limiar_confianca:
        valor_stop = preco_atual * (1 - stop_loss)
        valor_gain = preco_atual * (1 + take_profit)

        risco_fin = capital_inicial * risco_por_trade_pct
        tam_por_risco = risco_fin / stop_loss
        tam_max = capital_inicial * max_alocacao_pct
        valor_alocado = min(capital_inicial, tam_por_risco, tam_max)

        msg = (
            f"🚀 *SINAL DE COMPRA - VESPA TRADE*\n\n"
            f"📈 *Ativo:* `{ticker}`\n"
            f"📅 *Data Ref:* `{data_ref}`\n"
            f"💰 *Preço Atual:* `{preco_atual:,.2f}`\n"
            f"🎯 *Confiança IA:* `{prob_pct:.1f}%` (Mínimo:"
            f" {limiar_confianca*100:.0f}%)\n"
            f"📊 *Win Rate Histórico:* `{win_rate_hist:.1f}%` ({trades_hist}"
            f" trades)\n"
            f"⭐ *Score Operacional:* `{score_operacional:.1f}%`\n\n"
            f"⚙️ *Ficha OCO para Corretora:*\n"
            f"• **Capital Alocado:** `R$ / $ {valor_alocado:,.2f}`\n"
            f"• **Stop Loss (-{stop_loss*100}%):** `{valor_stop:,.2f}`\n"
            f"• **Take Profit (+{take_profit*100}%):** `{valor_gain:,.2f}`\n"
        )
        sinais_encontrados.append(msg)

    except Exception as e:
        print(f"Erro ao processar {ticker}: {e}")

  # Consolidar relatório para o Telegram
  resumo_texto = "\n".join(resumo_status) if resumo_status else "Nenhum ativo."

  if sinais_encontrados:
    mensagem_final = (
        f"🐝 *RELATÓRIO VESPA TRADE - OPORTUNIDADES*\n"
        f"⚙️ *Régua de Confiança:* `{limiar_confianca*100:.1f}%`\n\n"
        f"📊 *Status Geral dos Ativos:*\n{resumo_texto}\n\n"
        f"-------------------\n"
        + "\n-------------------\n".join(sinais_encontrados)
    )
  else:
    mensagem_final = (
        f"🐝 *RELATÓRIO DIÁRIO - VESPA TRADE*\n"
        f"⚙️ *Régua de Confiança:* `{limiar_confianca*100:.1f}%`\n\n"
        f"📊 *Status Geral dos Ativos:*\n{resumo_texto}\n\n"
        f"⏳ *Nenhum ativo atingiu o limiar mínimo hoje. Protegendo o capital.*"
    )

  enviar_alerta_telegram(mensagem_final)


if __name__ == "__main__":
  varredura_vespa_trade()