from datetime import datetime
import os
import sqlite3
import asyncio
import threading
from bot_alerta import main as iniciar_bot_telegram

# Adicionado 'flash', 'redirect' e 'url_for' aqui embaixo:
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from werkzeug.security import check_password_hash, generate_password_hash
import yfinance as yf

app = Flask(__name__)
app.config['SECRET_KEY'] = 'chave-secreta-vespa-trade-2026'

# Pega a URL do Render e corrige o prefixo se necessário
db_url = os.environ.get('DATABASE_URL', 'sqlite:///dados_robo.db')
if db_url:
  if db_url.startswith('postgres://'):
    db_url = db_url.replace('postgres://', 'postgresql+psycopg2://', 1)
  elif db_url.startswith('postgresql://'):
    db_url = db_url.replace('postgresql://', 'postgresql+psycopg2://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# Modelos de Usuário e Ativos
class User(UserMixin, db.Model):
  __tablename__ = 'usuarios'
  id = db.Column(db.Integer, primary_key=True)
  username = db.Column(db.String(150), unique=True, nullable=False)
  password = db.Column(db.String(200), nullable=False)
  is_approved = db.Column(db.Boolean, default=False)


class AtivoFavorito(db.Model):
  __tablename__ = 'ativos_favoritos'
  id = db.Column(db.Integer, primary_key=True)
  ticker = db.Column(db.String(50), unique=True, nullable=False)


@login_manager.user_loader
def load_user(user_id):
  return User.query.get(int(user_id))


# Bloco de criação automática das tabelas e do admin padrão (colocar antes do if __name__ == '__main__')
with app.app_context():
  db.create_all()
  if not User.query.filter_by(username='admin').first():
    hashed_pw = generate_password_hash('admin123', method='pbkdf2:sha256')
    admin_user = User(
        username='admin', password=hashed_pw, is_approved=True
    )
    db.session.add(admin_user)
    db.session.commit()

  if AtivoFavorito.query.count() == 0:
    padroes = [
        'PETR4.SA',
        'VALE3.SA',
        'ITUB4.SA',
        'BBDC4.SA',
        'ABEV3.SA',
        'WEGE3.SA',
        'SPY',
        'QQQ',
        'BTC-USD',
        'ETH-USD',
    ]
    for t in padroes:
      db.session.add(AtivoFavorito(ticker=t))
    try:
      db.session.commit()
    except:
      db.session.rollback()


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

  # Insere ativos padrão de fábrica se a tabela estiver vazia
  ativos_padrao = [
      "PETR4.SA",
      "VALE3.SA",
      "ITUB4.SA",
      "BBDC4.SA",
      "ABEV3.SA",
      "WEGE3.SA",
      "SPY",
      "QQQ",
      "BTC-USD",
      "ETH-USD",
  ]
  for ticker in ativos_padrao:
    try:
      cursor.execute(
          "INSERT OR IGNORE INTO ativos_favoritos (ticker) VALUES (?)",
          (ticker,),
      )
    except Exception:
      pass

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


def calcular_rsi(serie, janela=14):
  delta = serie.diff()
  ganho = delta.where(delta > 0, 0).rolling(window=janela).mean()
  perda = (-delta.where(delta < 0, 0)).rolling(window=janela).mean()
  rs = ganho / perda
  return 100 - (100 / (1 + rs))


def encontrar_melhor_configuracao_historica(
    df_test,
    confianca,
    horizonte,
    capital_ini,
    risco_pct,
    max_alocacao,
    taxa_corretagem,
):
  sl_options = [0.01, 0.015, 0.02, 0.025]
  tp_options = [0.02, 0.03, 0.04, 0.05]

  best_wr = -1
  best_sl_val = 1.5
  best_tp_val = 3.0

  for sl_val in sl_options:
    for tp_val in tp_options:
      capital = capital_ini
      posicao = False
      p_ent = 0.0
      dias_p = 0
      p_stop = 0.0
      p_gain = 0.0
      trades = 0
      vitorias = 0

      for i in range(len(df_test)):
        h_at = df_test["High"].iloc[i]
        l_at = df_test["Low"].iloc[i]
        c_at = df_test["Close"].iloc[i]
        prob = df_test["Prob_Alta"].iloc[i]

        if not posicao:
          if prob >= confianca:
            posicao = True
            p_ent = c_at
            dias_p = 0
            p_stop = p_ent * (1 - sl_val)
            p_gain = p_ent * (1 + tp_val)
            tamanho_posicao = min(
                capital,
                (capital * risco_pct) / sl_val,
                capital * max_alocacao,
            )
            capital -= tamanho_posicao * taxa_corretagem
        else:
          dias_p += 1
          b_stop = l_at <= p_stop
          b_gain = h_at >= p_gain

          if b_stop and b_gain:
            var = -sl_val
            posicao = False
          elif b_stop:
            var = -sl_val
            posicao = False
          elif b_gain:
            var = tp_val
            posicao = False
          elif dias_p >= horizonte:
            var = (c_at - p_ent) / p_ent
            posicao = False

          if not posicao:
            trades += 1
            if var > 0:
              vitorias += 1

      wr = (vitorias / trades * 100) if trades >= 2 else 0
      if wr > best_wr and trades >= 2:
        best_wr = wr
        best_sl_val = sl_val * 100
        best_tp_val = tp_val * 100

  return best_wr, best_sl_val, best_tp_val


def gerar_comentario_ia(
    prob, limiar, win_rate, total_trades, best_wr, best_sl, best_tp
):
  pct = prob * 100
  if pct < 40:
    base_txt = (
        "🔴 **Risco Alto:** Baixa probabilidade estatística atual."
        " Recomenda-se cautela."
    )
  elif 40 <= pct < (limiar * 100):
    base_txt = (
        "🟡 **Zona Neutra:** Abaixo da régua de confiança exigida para"
        " entrada."
    )
  elif (limiar * 100) <= pct < 75:
    base_txt = (
        "🟢 **Boas Probabilidades:** Padrões técnicos favoráveis detectados."
    )
  else:
    base_txt = (
        "⭐ **Alta Convicção:** Alinhamento robusto de indicadores para alta."
    )

  if total_trades >= 3:
    hist_txt = (
        f" O histórico atual registra uma taxa de acerto de {win_rate:.1f}%"
        f" ({total_trades} operações)."
    )
  else:
    hist_txt = " Poucos trades históricos registrados para este ativo."

  sugestao_otimizacao = (
      f"<br><br>💡 <strong>Sugestão baseada no histórico:</strong> O melhor Win"
      f" Rate ({best_wr:.1f}%) para este ativo foi alcançado utilizando um"
      f" <strong>Stop Loss de {best_sl:.1f}%</strong> e um <strong>Take Profit"
      f" de {best_tp:.1f}%</strong>. Esta é uma sugestão gerada de acordo com o"
      f" comportamento histórico do ativo."
  )

  return base_txt + hist_txt + sugestao_otimizacao


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

    lim_val = confianca if confianca <= 1 else confianca / 100
    best_wr, best_sl, best_tp = encontrar_melhor_configuracao_historica(
        df_test,
        lim_val,
        horizonte,
        capital_ini,
        risco_pct,
        max_alocacao,
        taxa_corretagem,
    )

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
        if prob >= lim_val:
          posicao = True
          preco_entrada = close_atual
          data_entrada = data_atual
          dias_na_posicao = 0
          preco_stop_atual = preco_entrada * (1 - sl)
          preco_gain_atual = preco_entrada * (1 + tp)

          risco_fin = capital * risco_pct
          tam_por_risco = risco_fin / sl
          tam_max = capital * max_alocacao
          tamanho_posicao = min(capital, tam_por_risco, tam_max)

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

    comentario = gerar_comentario_ia(
        prob_atual, lim_val, win_rate, total_trades, best_wr, best_sl, best_tp
    )

    datas_str = [str(idx)[:10] for idx in df_test.index]
    precos_float = [float(p) for p in df_test["Close"].values]

    return {
        "ticker": ticker,
        "capital_final": round(capital, 2),
        "lucro_liquido": round(lucro_liq, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": total_trades,
        "max_drawdown": round(max_drawdown, 2),
        "historico_trades": historico_trades,
        "comentario_ia": comentario,
        "datas": datas_str,
        "capital_historico": capital_historico,
        "precos": precos_float,
        "sinal_atual": {
            "data": data_atual_mercado,
            "preco": preco_atual_mercado,
            "probabilidade": round(prob_atual * 100, 1),
            "ativo_recomendado": bool(prob_atual >= lim_val),
        },
    }
  except Exception as e:
    print(f"Erro no backtest: {e}")
    return None


@app.route('/')
@login_required
def index():
  if not current_user.is_approved:
    return redirect(url_for('logout'))

  return render_template('index.html')

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
    ativos = ["SPY", "BTC-USD", "PETR4.SA"]
  return jsonify(ativos)


@app.route("/api/analisar", methods=["POST"])
def analisar():
  params = request.json
  ticker = params.get("ticker", "SPY")
  capital = float(params.get("capital", 10000.0))
  risco_pct = float(params.get("risco", 1.0)) / 100
  max_aloc = float(params.get("max_alocacao", 20.0)) / 100
  confianca = float(params.get("confianca", 62.0))
  sl = float(params.get("stop_loss", 1.5)) / 100
  tp = float(params.get("take_profit", 3.0)) / 100
  taxa = float(params.get("taxa", 0.05)) / 100
  timeframe = params.get("timeframe", "Diário (1D)")

  # --- SALVA AUTOMATICAMENTE NO BANCO PARA O BOT DO TELEGRAM ---
  salvar_config_db("capital_inicial", capital)
  salvar_config_db("risco_por_trade", float(params.get("risco", 1.0)))
  salvar_config_db("max_alocacao", float(params.get("max_alocacao", 20.0)))
  salvar_config_db("confianca", confianca)
  salvar_config_db("stop_loss", float(params.get("stop_loss", 1.5)))
  salvar_config_db("take_pers", float(params.get("take_profit", 3.0)))
  salvar_config_db("take_profit", float(params.get("take_profit", 3.0)))

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

@app.route('/login', methods=['GET', 'POST'])
def login():
  if request.method == 'POST':
    username = request.form.get('username')
    password = request.form.get('password')

    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
      if not user.is_approved:
        flash('Sua conta aguarda aprovação do administrador.', 'warning')
        return render_template('login.html')

      login_user(user)
      return redirect(url_for('index'))
    else:
      flash('Usuário ou senha inválidos.', 'danger')

  return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
  logout_user()
  return redirect(url_for('login'))

if __name__ == "__main__":
  import os

  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)

  def rodar_telegram_em_background():
    # Cria um loop assíncrono para o Telegram rodar isolado em uma thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(iniciar_bot_telegram())

# Inicia o bot automaticamente quando a aplicação sobe
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or "gunicorn" in os.environ.get("SERVER_SOFTWARE", ""):
    t = threading.Thread(target=rodar_telegram_em_background, daemon=True)
    t.start()
    print("🧵 Thread do Bot do Telegram iniciada com sucesso!")