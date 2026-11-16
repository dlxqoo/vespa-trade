import asyncio
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import yfinance as yf

# --- CONFIGURAÇÃO DO BOT DO TELEGRAM ---
TELEGRAM_TOKEN = "9010278583:AAGBfLSgDQP7jVb9JwnvukfSG79jDpEzny0"
CONFIG_FILE = "config_telegram.json"


# --- GERENCIAMENTO DE PREFERÊNCIAS (JSON) ---
def carregar_config():
  if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
      return json.load(f)
  return {}


def salvar_config(data):
  with open(CONFIG_FILE, "w") as f:
    json.dump(data, f, indent=4)


# --- COMANDOS DO TELEGRAM ---
async def set_ativos(update: Update, context: ContextTypes.DEFAULT_TYPE):
  chat_id = str(update.effective_chat.id)
  args = context.args  # Ex: /ativos PETR4.SA VALE3.SA BTC-USD

  if not args:
    await update.message.reply_text(
        "⚠️ Use o formato correto:\n`/ativos PETR4.SA VALE3.SA`",
        parse_mode="Markdown",
    )
    return

  config = carregar_config()
  if chat_id not in config:
    config[chat_id] = {"ativos": [], "confianca_minima": 62.0}

  config[chat_id]["ativos"] = [ativo.upper() for ativo in args]
  salvar_config(config)

  await update.message.reply_text(
      f"✅ Ativos monitorados atualizados:\n"
      f"_{', '.join(config[chat_id]['ativos'])}_",
      parse_mode="Markdown",
  )


async def set_confianca(update: Update, context: ContextTypes.DEFAULT_TYPE):
  chat_id = str(update.effective_chat.id)
  args = context.args

  if not args or not args[0].replace(".", "", 1).isdigit():
    await update.message.reply_text(
        "⚠️ Use o formato correto:\n`/confianca 70` (para 70% mínimo)",
        parse_mode="Markdown",
    )
    return

  valor = float(args[0])
  config = carregar_config()
  if chat_id not in config:
    config[chat_id] = {
        "ativos": ["PETR4.SA", "VALE3.SA"],
        "confianca_minima": 62.0,
    }

  config[chat_id]["confianca_minima"] = valor
  salvar_config(config)

  await update.message.reply_text(
      f"✅ Confiança mínima ajustada para: *{valor}%*", parse_mode="Markdown"
  )


async def status_comando(update: Update, context: ContextTypes.DEFAULT_TYPE):
  chat_id = str(update.effective_chat.id)
  config = carregar_config()

  if chat_id not in config or not config[chat_id]["ativos"]:
    await update.message.reply_text(
        "⚙️ Você ainda não configurou ativos.\nUse `/ativos PETR4.SA` e"
        " `/confianca 65`",
        parse_mode="Markdown",
    )
    return

  user_conf = config[chat_id]
  ativos_str = ", ".join(user_conf["ativos"])
  await update.message.reply_text(
      f"📊 *Suas Configurações Atuais:*\n\n"
      f"• *Ativos:* `{ativos_str}`\n"
      f"• *Confiança Mínima:* `{user_conf['confianca_minima']}%`",
      parse_mode="Markdown",
  )


# --- FUNÇÕES DE ANÁLISE QUANTITATIVA ---
def enviar_alerta_telegram(chat_id, mensagem):
  url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
  payload = {"chat_id": chat_id, "text": mensagem, "parse_mode": "Markdown"}
  try:
    response = requests.post(url, json=payload)
    if response.status_code != 200:
      print(f"Erro ao enviar Telegram para {chat_id}: {response.text}")
    else:
      print(f"🐝 Alerta enviado com sucesso para o chat {chat_id}!")
  except Exception as e:
    print(f"Erro de conexão com Telegram: {e}")


def calcular_rsi(serie, janela=14):
  delta = serie.diff()
  ganho = delta.where(delta > 0, 0).rolling(window=janela).mean()
  perda = (-delta.where(delta < 0, 0)).rolling(window=janela).mean()
  rs = ganho / perda
  return 100 - (100 / (1 + rs))


def executar_varredura_usuario(chat_id, user_prefs):
  lista_ativos = user_prefs.get("ativos", [])
  limiar_confianca = user_prefs.get("confianca_minima", 62.0) / 100
  stop_loss = 0.015  # 1.5%
  take_profit = 0.03  # 3.0%
  capital_inicial = 10000.0
  risco_por_trade_pct = 0.01
  max_alocacao_pct = 0.20
  horizonte = 5

  if not lista_ativos:
    return

  print(
      f"🐝 [Vespa Trade] Analisando para o chat {chat_id} | Confiança Mínima:"
      f" {limiar_confianca*100:.1f}%"
  )

  sinais_encontrados = []
  resumo_status = []

  for ticker in lista_ativos:
    try:
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

      # Cálculo do Win Rate Histórico
      trades_hist, vitorias_hist, posicao = 0, 0, False
      p_ent, dias_p, p_stop, p_gain = 0.0, 0, 0.0, 0.0

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
            f" trades)\n\n"
            f"⚙️ *Ficha OCO para Corretora:*\n"
            f"• **Capital Alocado:** `R$ / $ {valor_alocado:,.2f}`\n"
            f"• **Stop Loss (-1.5%):** `{valor_stop:,.2f}`\n"
            f"• **Take Profit (+3.0%):** `{valor_gain:,.2f}`\n"
        )
        sinais_encontrados.append(msg)

    except Exception as e:
        print(f"Erro ao processar {ticker} para o chat {chat_id}: {e}")

  # Consolidar relatório para o usuário
  resumo_texto = "\n".join(resumo_status) if resumo_status else "Nenhum ativo."

  if sinais_encontrados:
    mensagem_final = (
        f"🐝 *RELATÓRIO VESPA TRADE - OPORTUNIDADES*\n"
        f"⚙️ *Sua Régua de Confiança:* `{limiar_confianca*100:.1f}%`\n\n"
        f"📊 *Status dos seus ativos:*\n{resumo_texto}\n\n"
        f"-------------------\n"
        + "\n-------------------\n".join(sinais_encontrados)
    )
  else:
    mensagem_final = (
        f"🐝 *RELATÓRIO DIÁRIO - VESPA TRADE*\n"
        f"⚙️ *Sua Régua de Confiança:* `{limiar_confianca*100:.1f}%`\n\n"
        f"📊 *Status dos seus ativos:*\n{resumo_texto}\n\n"
        f"⏳ *Nenhum ativo atingiu o seu limiar hoje.*"
    )

  enviar_alerta_telegram(chat_id, mensagem_final)


def disparar_varreduras_gerais():
  config = carregar_config()
  if not config:
    print("Nenhum usuário configurado no bot ainda.")
    return

  for chat_id, prefs in config.items():
    executar_varredura_usuario(chat_id, prefs)


# --- INICIALIZAÇÃO DO BOT ---
async def main():
  app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

  app.add_handler(CommandHandler("ativos", set_ativos))
  app.add_handler(CommandHandler("confianca", set_confianca))
  app.add_handler(CommandHandler("status", status_comando))

  print("🤖 Bot do Telegram rodando e escutando comandos...")
  await app.initialize()
  await app.start()
  await app.updater.start_polling()

  # Mantém o bot rodando em background e executa varreduras periódicas se quiser
  while True:
    await asyncio.sleep(3600)  # Mantém o loop ativo


if __name__ == "__main__":
  asyncio.run(main())