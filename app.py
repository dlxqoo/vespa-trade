import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yfinance as yf

# Configuração da Página
st.set_page_config(
    page_title='Robô de Análise & Swing Trade Pro', layout='wide'
)
st.title('🤖 Sistema Profissional de Análise & Backtest com IA')


# --- BANCO DE DADOS LOCAL (SQLite) ---
def init_db():
  conn = sqlite3.connect('dados_robo.db')
  cursor = conn.cursor()
  cursor.execute('''
        CREATE TABLE IF NOT EXISTS ativos_favoritos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE
        )
    ''')
  conn.commit()
  conn.close()


init_db()


def adicionar_ativo_db(ticker):
  conn = sqlite3.connect('dados_robo.db')
  cursor = conn.cursor()
  try:
    cursor.execute(
        'INSERT OR IGNORE INTO ativos_favoritos (ticker) VALUES (?)',
        (ticker.upper(),),
    )
    conn.commit()
  except Exception as e:
    st.error(f'Erro ao salvar: {e}')
  finally:
    conn.close()


def carregar_ativos_db():
  conn = sqlite3.connect('dados_robo.db')
  df_ativos = pd.read_sql('SELECT ticker FROM ativos_favoritos', conn)
  conn.close()
  return df_ativos['ticker'].tolist()


# --- BARRA LATERAL (CONTROLES DE SIMULAÇÃO) ---
st.sidebar.header('⚙️ Parâmetros de Simulação')

capital_inicial = st.sidebar.number_input(
    'Capital Inicial ($ / R$)',
    min_value=100.0,
    max_value=1000000.0,
    value=10000.0,
    step=500.0,
)

novo_ticker = st.sidebar.text_input(
    'Adicionar Ticker para Monitorar', value='SPY'
)
if st.sidebar.button('Salvar Ativo'):
  adicionar_ativo_db(novo_ticker)
  st.sidebar.success(f'{novo_ticker.upper()} salvo!')

ativos_salvos = carregar_ativos_db()
if not ativos_salvos:
  ativos_salvos = ['SPY']

ticker_selecionado = st.sidebar.selectbox(
    'Escolher Ativo Salvo', ativos_salvos
)

st.sidebar.markdown('---')
limiar_confianca = st.sidebar.slider('Confiança Mínima da IA (%)', 50, 85, 62) / 100
stop_loss = st.sidebar.slider('Stop Loss (%)', 0.5, 5.0, 1.5) / 100
take_profit = st.sidebar.slider('Take Profit (%)', 1.0, 10.0, 3.0) / 100

# Novo parâmetro: Taxa de corretagem por ordem
taxa_corretagem_pct = (
    st.sidebar.slider('Taxa de Corretagem / Custos por Ordem (%)', 0.0, 0.5, 0.05)
    / 100
)


def calcular_rsi(serie, janela=14):
  delta = serie.diff()
  ganho = delta.where(delta > 0, 0).rolling(window=janela).mean()
  perda = (-delta.where(delta < 0, 0)).rolling(window=janela).mean()
  rs = ganho / perda
  return 100 - (100 / (1 + rs))


# --- EXECUTAR BACKTEST ---
if st.sidebar.button('Rodar Simulação & IA'):
  with st.spinner(f'Processando dados históricos para {ticker_selecionado}...'):
    data = yf.download(ticker_selecionado, period='7y', interval='1d', progress=False)
    if isinstance(data.columns, pd.MultiIndex):
      data.columns = data.columns.droplevel(1)

    data['Retorno_1d'] = data['Close'].pct_change(1)
    data['Retorno_5d'] = data['Close'].pct_change(5)
    data['Retorno_20d'] = data['Close'].pct_change(20)
    data['Volatilidade'] = data['Retorno_1d'].rolling(20).std()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['Dist_SMA50'] = (data['Close'] - data['SMA_50']) / data['SMA_50']
    data['RSI'] = calcular_rsi(data['Close'], 14)
    data['Volume_Relativo'] = data['Volume'] / data['Volume'].rolling(20).mean()

    horizonte = 5
    alvo_ganho = take_profit
    data['Preco_Futuro'] = data['Close'].shift(-horizonte)
    data['Target'] = np.where(
        (data['Preco_Futuro'] - data['Close']) / data['Close'] >= alvo_ganho,
        1,
        0,
    )
    data.dropna(inplace=True)

    features = [
        'Retorno_1d',
        'Retorno_5d',
        'Retorno_20d',
        'Volatilidade',
        'Dist_SMA50',
        'RSI',
        'Volume_Relativo',
    ]
    X = data[features]
    y = data['Target']

    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    df_test = data.iloc[split_idx:].copy()

    model = RandomForestClassifier(
        n_estimators=150, max_depth=6, random_state=42, class_weight='balanced'
    )
    model.fit(X_train, y_train)
    df_test['Prob_Alta'] = model.predict_proba(X_test)[:, 1]

    # Simulação de Trades com Taxas de Corretagem
    capital = capital_inicial
    posicao = False
    preco_entrada = 0.0
    dias_na_posicao = 0
    historico_trades = []
    capital_historico = []

    for i in range(len(df_test)):
      preco_atual = df_test['Close'].iloc[i]
      data_atual = df_test.index[i]
      prob = df_test['Prob_Alta'].iloc[i]

      if not posicao:
        if prob >= limiar_confianca:
          posicao = True
          preco_entrada = preco_atual
          dias_na_posicao = 0
          # Desconta taxa de corretagem na ENTRADA (Compra)
          capital -= capital * taxa_corretagem_pct
          historico_trades.append({
              'Data_Entrada': data_atual,
              'Tipo': 'COMPRA',
              'Preco': preco_entrada,
          })
      else:
        dias_na_posicao += 1
        variacao = (preco_atual - preco_entrada) / preco_entrada
        if (
            variacao <= -stop_loss
            or variacao >= take_profit
            or dias_na_posicao >= horizonte
        ):
          lucro_op = capital * variacao
          capital += lucro_op
          # Desconta taxa de corretagem na SAÍDA (Venda)
          capital -= capital * taxa_corretagem_pct
          posicao = False
          historico_trades.append({
              'Data_Saida': data_atual,
              'Tipo': 'VENDA',
              'Preco': preco_atual,
              'Lucro_Pct': variacao * 100,
              'Lucro_Financeiro': lucro_op,
          })
      capital_historico.append(capital)

    df_test['Capital_Estrategia'] = capital_historico

    # Métricas e Estatísticas
    vendas = [t for t in historico_trades if t['Tipo'] == 'VENDA']
    total_trades = len(vendas)
    vitorias = len([t for t in vendas if t['Lucro_Pct'] > 0])
    win_rate = (vitorias / total_trades * 100) if total_trades > 0 else 0
    lucro_liq = capital - capital_inicial

    serie_cap = pd.Series(capital_historico)
    pico = serie_cap.cummax()
    drawdown = (serie_cap - pico) / pico
    max_drawdown = drawdown.min() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        'Capital Final',
        f'$ {capital:,.2f}',
        f'${lucro_liq:+,.2f}',
        delta_color='normal',
    )
    col2.metric('Taxa de Acerto (Win Rate)', f'{win_rate:.1f}%')
    col3.metric('Total de Operações', total_trades)
    col4.metric('Max Drawdown (Pior Queda)', f'{max_drawdown:.2f}%')

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f'Backtest com IA & Custos ({ticker_selecionado})',
            'Evolução do Capital Líquido',
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=df_test.index,
            y=df_test['Close'],
            name='Preço',
            line=dict(color='royalblue'),
        ),
        row=1,
        col=1,
    )

    compras_x = [
        t['Data_Entrada'] for t in historico_trades if t['Tipo'] == 'COMPRA'
    ]
    compras_y = [t['Preco'] for t in historico_trades if t['Tipo'] == 'COMPRA']
    vendas_x = [t['Data_Saida'] for t in historico_trades if t['Tipo'] == 'VENDA']
    vendas_y = [t['Preco'] for t in historico_trades if t['Tipo'] == 'VENDA']

    fig.add_trace(
        go.Scatter(
            x=compras_x,
            y=compras_y,
            mode='markers',
            name='Compra',
            marker=dict(color='limegreen', size=10, symbol='triangle-up'),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=vendas_x,
            y=vendas_y,
            mode='markers',
            name='Venda',
            marker=dict(color='crimson', size=10, symbol='triangle-down'),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_test.index,
            y=df_test['Capital_Estrategia'],
            name='Capital',
            line=dict(color='mediumseagreen', width=2),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=700, template='plotly_dark', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('📋 Histórico Detalhado de Operações')
    if vendas:
      df_historico_trades = pd.DataFrame(vendas)
      st.dataframe(df_historico_trades, use_container_width=True)
    else:
      st.info('Nenhuma operação de venda concluída no período.')