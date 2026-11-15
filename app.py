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


# --- BARRA LATERAL (CONTROLES GLOBAIS) ---
st.sidebar.header('⚙️ Parâmetros Globais de Risco')

capital_inicial = st.sidebar.number_input(
    'Capital Inicial ($ / R$)',
    min_value=100.0,
    max_value=1000000.0,
    value=10000.0,
    step=500.0,
)

# Position Sizing: Risco máximo do capital por operação (%)
risco_por_trade_pct = (
    st.sidebar.slider('Risco por Operação (Position Sizing %)', 0.5, 5.0, 1.0)
    / 100
)

novo_ticker = st.sidebar.text_input(
    'Adicionar Ticker (Ex: PETR4.SA, BTC-USD, AAPL)', value='SPY'
)
if st.sidebar.button('Salvar Ativo'):
  adicionar_ativo_db(novo_ticker)
  st.sidebar.success(f'{novo_ticker.upper()} salvo!')

ativos_salvos = carregar_ativos_db()
if not ativos_salvos:
  ativos_salvos = ['SPY']

ticker_selecionado = st.sidebar.selectbox(
    'Escolher Ativo para Análise Individual', ativos_salvos
)

st.sidebar.markdown('---')
limiar_confianca = st.sidebar.slider('Confiança Mínima da IA (%)', 50, 85, 62) / 100
stop_loss = st.sidebar.slider('Stop Loss (%)', 0.5, 5.0, 1.5) / 100
take_profit = st.sidebar.slider('Take Profit (%)', 1.0, 10.0, 3.0) / 100
taxa_corretagem_pct = (
    st.sidebar.slider('Taxa de Corretagem por Ordem (%)', 0.0, 0.5, 0.05) / 100
)


def calcular_rsi(serie, janela=14):
  delta = serie.diff()
  ganho = delta.where(delta > 0, 0).rolling(window=janela).mean()
  perda = (-delta.where(delta < 0, 0)).rolling(window=janela).mean()
  rs = ganho / perda
  return 100 - (100 / (1 + rs))


# Função unificada de motor de backtest (usada tanto na análise individual quanto no scanner)
def executar_backtest(
    ticker,
    capital_ini,
    risco_pct,
    sl,
    tp,
    taxa_corretagem,
    confianca,
    horizonte=5,
):
  try:
    data = yf.download(ticker, period='7y', interval='1d', progress=False)
    if data.empty or len(data) < 200:
      return None

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

    data['Preco_Futuro'] = data['Close'].shift(-horizonte)
    data['Target'] = np.where(
        (data['Preco_Futuro'] - data['Close']) / data['Close'] >= tp, 1, 0
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

    if len(X_test) < 10:
      return None

    model = RandomForestClassifier(
        n_estimators=150, max_depth=6, random_state=42, class_weight='balanced'
    )
    model.fit(X_train, y_train)
    df_test['Prob_Alta'] = model.predict_proba(X_test)[:, 1]

    capital = capital_ini
    posicao = False
    preco_entrada = 0.0
    data_entrada = None
    dias_na_posicao = 0
    historico_trades = []
    capital_historico = []

    for i in range(len(df_test)):
      preco_atual = df_test['Close'].iloc[i]
      data_atual = df_test.index[i]
      prob = df_test['Prob_Alta'].iloc[i]

      if not posicao:
        if prob >= confianca:
          posicao = True
          preco_entrada = preco_atual
          data_entrada = data_atual
          dias_na_posicao = 0

          # Position Sizing: Dinheiro alocado na operação baseado no risco máximo permitido
          risco_financeiro_desejado = capital * risco_pct
          tamanho_posicao = min(
              capital, risco_financeiro_desejado / sl
          )  # Alocação calculada pelo Stop Loss

          # Desconta corretagem na entrada sobre o montante alocado
          capital -= tamanho_posicao * taxa_corretagem
          capital_alocado = tamanho_posicao
      else:
        dias_na_posicao += 1
        variacao = (preco_atual - preco_entrada) / preco_entrada
        if variacao <= -sl or variacao >= tp or dias_na_posicao >= horizonte:
          lucro_op = capital_alocado * variacao
          capital += lucro_op
          capital -= capital_alocado * taxa_corretagem  # Corretagem na saída
          posicao = False

          historico_trades.append({
              'Data_Entrada': data_entrada,
              'Preco_Entrada': preco_entrada,
              'Data_Saida': data_atual,
              'Preco_Saida': preco_atual,
              'Lucro_Pct': variacao * 100,
              'Lucro_Financeiro': lucro_op,
          })

      capital_historico.append(capital)

    df_test['Capital_Estrategia'] = capital_historico

    total_trades = len(historico_trades)
    vitorias = len([t for t in historico_trades if t['Lucro_Pct'] > 0])
    win_rate = (vitorias / total_trades * 100) if total_trades > 0 else 0
    lucro_liq = capital - capital_ini

    serie_cap = pd.Series(capital_historico)
    pico = serie_cap.cummax()
    drawdown = (serie_cap - pico) / pico
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0

    return {
        'ticker': ticker,
        'capital_final': capital,
        'lucro_liquido': lucro_liq,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'max_drawdown': max_drawdown,
        'df_test': df_test,
        'historico_trades': historico_trades,
    }
  except Exception as e:
    return None


# --- ABAS DA INTERFACE ---
aba1, aba2 = st.tabs(
    ['📈 Análise Individual & Gráfico', '🏆 Scanner Multiativos (Ranking)']
)

with aba1:
  st.subheader(f'Análise Detalhada: {ticker_selecionado}')
  if st.button('Rodar Simulação Individual'):
    with st.spinner(f'Processando modelo de IA para {ticker_selecionado}...'):
      resultado = executar_backtest(
          ticker_selecionado,
          capital_inicial,
          risco_por_trade_pct,
          stop_loss,
          take_profit,
          taxa_corretagem_pct,
          limiar_confianca,
      )

      if resultado:
        st.session_state['res_individual'] = resultado
      else:
        st.error(
            'Não foi possível processar este ativo (dados insuficientes ou erro'
            ' de conexão).'
        )

  if 'res_individual' in st.session_state:
    res = st.session_state['res_individual']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        'Capital Final',
        f'$ {res["capital_final"]:,.2f}',
        f'${res["lucro_liquido"]:+,.2f}',
    )
    col2.metric('Taxa de Acerto (Win Rate)', f'{res["win_rate"]:.1f}%')
    col3.metric('Total de Operações', res['total_trades'])
    col4.metric('Max Drawdown', f'{res["max_drawdown"]:.2f}%')

    # Gráfico Plotly
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f'Backtest com Position Sizing & IA ({res["ticker"]})',
            'Evolução do Capital Líquido',
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=res['df_test'].index,
            y=res['df_test']['Close'],
            name='Preço',
            line=dict(color='royalblue'),
        ),
        row=1,
        col=1,
    )

    compras_x = [t['Data_Entrada'] for t in res['historico_trades']]
    compras_y = [t['Preco_Entrada'] for t in res['historico_trades']]
    vendas_x = [t['Data_Saida'] for t in res['historico_trades']]
    vendas_y = [t['Preco_Saida'] for t in res['historico_trades']]

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
            x=res['df_test'].index,
            y=res['df_test']['Capital_Estrategia'],
            name='Capital',
            line=dict(color='mediumseagreen', width=2),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=700, template='plotly_dark', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('📋 Histórico Detalhado de Operações')
    if res['historico_trades']:
      st.dataframe(
          pd.DataFrame(res['historico_trades']), use_container_width=True
      )
    else:
      st.info('Nenhuma operação concluída no período.')

with aba2:
  st.subheader('🏆 Scanner Global & Ranking de Ativos')
  st.write(
      'Varre automaticamente todos os ativos salvos na base de dados local,'
      ' aplicando os parâmetros atuais de IA e risco.'
  )

  if st.button('Executar Scanner em Lote'):
    lista_ativos = carregar_ativos_db()
    if not lista_ativos:
      lista_ativos = ['SPY', 'BTC-USD', 'PETR4.SA']

    resultados_lote = []
    barra_progresso = st.progress(0)

    for idx, tk in enumerate(lista_ativos):
      res_lote = executar_backtest(
          tk,
          capital_inicial,
          risco_por_trade_pct,
          stop_loss,
          take_profit,
          taxa_corretagem_pct,
          limiar_confianca,
      )
      if res_lote:
        resultados_lote.append({
            'Ativo': res_lote['ticker'],
            'Capital Final ($/R$)': round(res_lote['capital_final'], 2),
            'Lucro Líquido ($/R$)': round(res_lote['lucro_liquido'], 2),
            'Win Rate (%)': round(res_lote['win_rate'], 1),
            'Total Trades': res_lote['total_trades'],
            'Max Drawdown (%)': round(res_lote['max_drawdown'], 2),
        })
      barra_progresso.progress((idx + 1) / len(lista_ativos))

    if resultados_lote:
      df_ranking = pd.DataFrame(resultados_lote)
      df_ranking = df_ranking.sort_values(
          by='Capital Final ($/R$)', ascending=False
      ).reset_index(drop=True)
      st.success('Scanner concluído com sucesso!')
      st.dataframe(df_ranking, use_container_width=True)
    else:
      st.warning('Nenhum ativo retornou dados válidos no scanner.')