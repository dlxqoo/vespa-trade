import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yfinance as yf

# Configuração da Página
st.set_page_config(page_title="Robô de Análise & Swing Trade Pro", layout="wide")
st.title("🤖 Sistema Profissional de Análise & Backtest com IA")


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
    conn.commit()
    conn.close()


init_db()


def adicionar_ativo_db(ticker):
    conn = sqlite3.connect("dados_robo.db")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR IGNORE INTO ativos_favoritos (ticker) VALUES (?)", (ticker.upper(),))
        conn.commit()
    except Exception as e:
        st.error(f"Erro ao salvar: {e}")
    finally:
        conn.close()


def carregar_ativos_db():
    conn = sqlite3.connect("dados_robo.db")
    df_ativos = pd.read_sql("SELECT ticker FROM ativos_favoritos", conn)
    conn.close()
    return df_ativos["ticker"].tolist()


# --- BARRA LATERAL (ENTRADAS MANUAIS E PARÂMETROS) ---
st.sidebar.header("⚙️ Parâmetros & Gestão de Risco")

capital_inicial = st.sidebar.number_input("Capital Inicial ($ / R$)", min_value=100.0, max_value=1000000.0, value=1000.0, step=100.0)

# Escolha do Timeframe
timeframe_opcao = st.sidebar.selectbox("⏱️ Timeframe (Período Gráfico)", ["Diário (1D)", "Semanal (1W)"])
yf_interval = "1d" if "Diário" in timeframe_opcao else "1wk"

# Entradas Manuais via digitação
risco_por_trade_input = st.sidebar.number_input("Risco por Operação (% da Banca no Stop)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
risco_por_trade_pct = risco_por_trade_input / 100

max_alocacao_input = st.sidebar.number_input("Alocação Máxima por Trade (% da Banca)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
max_alocacao_pct = max_alocacao_input / 100

novo_ticker = st.sidebar.text_input("Adicionar Ticker (Ex: PETR4.SA, BTC-USD, AAPL)", value="SPY")
if st.sidebar.button("Salvar Ativo"):
    adicionar_ativo_db(novo_ticker)
    st.sidebar.success(f"{novo_ticker.upper()} salvo!")

ativos_salvos = carregar_ativos_db()
if not ativos_salvos:
    ativos_salvos = ["SPY"]

ticker_selecionado = st.sidebar.selectbox("Escolher Ativo para Análise Individual", ativos_salvos)

st.sidebar.markdown("---")

limiar_confianca_input = st.sidebar.number_input("Confiança Mínima da IA para Entrada (%)", min_value=40.0, max_value=90.0, value=62.0, step=1.0)
limiar_confianca = limiar_confianca_input / 100

stop_loss_input = st.sidebar.number_input("Stop Loss - Prejuízo Máximo (%)", min_value=0.1, max_value=20.0, value=1.5, step=0.1)
stop_loss = stop_loss_input / 100

take_profit_input = st.sidebar.number_input("Stop Gain / Take Profit - Alvo (%)", min_value=0.1, max_value=50.0, value=3.0, step=0.1)
take_profit = take_profit_input / 100

taxa_corretagem_input = st.sidebar.number_input("Taxa de Corretagem por Ordem (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
taxa_corretagem_pct = taxa_corretagem_input / 100


def calcular_rsi(serie, janela=14):
    delta = serie.diff()
    ganho = delta.where(delta > 0, 0).rolling(window=janela).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(window=janela).mean()
    rs = ganho / perda
    return 100 - (100 / (1 + rs))


# Função unificada de backtest com verificação via Máxima/Mínima do candle
def executar_backtest(ticker, capital_ini, risco_pct, max_alocacao, sl, tp, taxa_corretagem, confianca, interval="1d", horizonte=5):
    try:
        period = "10y" if interval == "1wk" else "7y"
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty or len(data) < 100:
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
        data['Target'] = np.where((data['Preco_Futuro'] - data['Close']) / data['Close'] >= tp, 1, 0)
        
        data_features = data.copy()
        data.dropna(inplace=True)

        features = ['Retorno_1d', 'Retorno_5d', 'Retorno_20d', 'Volatilidade', 'Dist_SMA50', 'RSI', 'Volume_Relativo']
        X = data[features]
        y = data['Target']

        split_idx = int(len(data) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        df_test = data.iloc[split_idx:].copy()

        if len(X_test) < 10:
            return None

        model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        df_test['Prob_Alta'] = model.predict_proba(X_test)[:, 1]

        # Predição do SINAL ATUAL
        X_ultimo = data_features[features].iloc[[-1]]
        prob_atual = model.predict_proba(X_ultimo)[0][1]
        data_atual_mercado = data_features.index[-1]
        preco_atual_mercado = data_features['Close'].iloc[-1]

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
            data_atual = df_test.index[i]
            high_atual = df_test['High'].iloc[i]
            low_atual = df_test['Low'].iloc[i]
            close_atual = df_test['Close'].iloc[i]
            prob = df_test['Prob_Alta'].iloc[i]

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
                        'Data_Entrada': data_entrada,
                        'Preco_Entrada': round(preco_entrada, 2),
                        'Valor_Alocado': round(capital_alocado, 2),
                        'Data_Saida': data_atual,
                        'Preco_Saida': round(preco_saida, 2),
                        'Lucro_Pct': round(variacao * 100, 2),
                        'Lucro_Financeiro': round(lucro_op, 2)
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

        timeframe_nome = "Diário (1D)" if interval == "1d" else "Semanal (1W)"

        return {
            'ticker': ticker,
            'timeframe': timeframe_nome,
            'capital_final': capital,
            'lucro_liquido': lucro_liq,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'max_drawdown': max_drawdown,
            'df_test': df_test,
            'historico_trades': historico_trades,
            'sinal_atual': {
                'data': data_atual_mercado,
                'preco': preco_atual_mercado,
                'probabilidade': prob_atual,
                'ativo_recomendado': prob_atual >= confianca,
                'timeframe': timeframe_nome
            }
        }
    except Exception as e:
        return None


# --- GERADOR DE COMENTÁRIO INTELIGENTE E ANÁLISE DE PARAMETRIZAÇÃO ---
def gerar_comentario_ia(prob, limiar, win_rate, total_trades):
    pct = prob * 100
    
    # Avaliação cruzada entre a probabilidade da IA e a taxa de acerto histórica do backtest
    if pct < 40:
        base_txt = "🔴 **Risco Alto de Prejuízo:** O modelo aponta baixa probabilidade estatística de alta. **Evite operar.**"
    elif 40 <= pct < limiar:
        base_txt = "🟡 **Zona Neutra / Alerta Amarelo:** O ativo está abaixo da exigência mínima de confiança. **Aguardar oportunidade.**"
    elif limiar <= pct < 75:
        base_txt = "🟢 **Boas Probabilidades:** O algoritmo identificou padrões favoráveis consistentes com o filtro de risco."
    else:
        base_txt = "⭐ **Alta Convicção Estatística:** O modelo detectou um alinhamento técnico robusto."

    # Adicionando validação histórica do backtest
    if total_trades >= 3:
        if win_rate >= 60.0:
            hist_txt = f" Além disso, o **histórico de testes valida positivamente** esta parametrização neste ativo, apresentando uma excelente taxa de acerto de **{win_rate:.1f}%** ({total_trades} trades testados)."
        elif win_rate >= 45.0:
            hist_txt = f" O histórico para estes parâmetros apresenta desempenho moderado, com taxa de acerto de **{win_rate:.1f}%**."
        else:
            hist_txt = f" ⚠️ *Atenção:* O histórico deste ativo com esses parâmetros específicos mostrou baixa efetividade (**{win_rate:.1f}%** de acerto), exigindo cautela extra."
    else:
        hist_txt = " *Nota:* Poucos trades históricos registrados para este ativo com os filtros atuais."

    return base_txt + hist_txt


# --- ABAS DA INTERFACE (Apenas 2 abas essenciais) ---
aba1, aba2 = st.tabs(["🎯 Sinal Real (Hoje)", "📊 Backtest Histórico & Desempenho"])

with aba1:
    st.subheader(f"Painel de Sinal em Tempo Real: {ticker_selecionado} ({timeframe_opcao})")
    st.write("Análise exclusiva do **último fechamento de mercado** cruzada com o desempenho histórico dos parâmetros.")

    if st.button("🔍 Verificar Sinal Atual para Este Ativo"):
        with st.spinner(f"Analisando dados atuais de {ticker_selecionado}..."):
            resultado = executar_backtest(
                ticker_selecionado, 
                capital_inicial, 
                risco_por_trade_pct, 
                max_alocacao_pct, 
                stop_loss, 
                take_profit, 
                taxa_corretagem_pct, 
                limiar_confianca,
                interval=yf_interval
            )
            if resultado:
                st.session_state['res_individual'] = resultado
            else:
                st.error("Erro ao carregar dados do ativo.")

    if 'res_individual' in st.session_state:
        res = st.session_state['res_individual']
        sinal = res['sinal_atual']
        prob_pct = sinal['probabilidade'] * 100
        win_rate_hist = res['win_rate']
        total_trd = res['total_trades']

        # Cálculo da Segunda Confiabilidade (Score Operacional Conjunto: IA + Histórico do Backtest)
        if total_trd > 0:
            score_operacional = (prob_pct * 0.4) + (min(win_rate_hist, 100) * 0.6)
        else:
            score_operacional = prob_pct * 0.8

        st.markdown("---")
        st.markdown("### 📋 Parecer do Robô para o Momento Atual")
        
        if sinal['ativo_recomendado']:
            st.success(f"🚀 **SINAL DE COMPRA CONFIRMADO ({sinal['timeframe']})**")
        else:
            st.warning(f"⏳ **FORA DOS PARÂMETROS DE ENTRADA ({sinal['timeframe']})**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Preço Atual", f"{sinal['preco']:,.2f}")
        c2.metric("Confiança da IA (Preditiva)", f"{prob_pct:.1f}%", f"Mínimo: {limiar_confianca*100:.0f}%")
        c3.metric("Taxa de Acerto Histórica", f"{win_rate_hist:.1f}%", f"{total_trd} trades passados")
        c4.metric("Score Operacional Conjunto", f"{score_operacional:.1f}%", "IA + Histórico")

        # Caixa com o comentário pré-programado inteligente com validação cruzada
        st.info(gerar_comentario_ia(sinal['probabilidade'], limiar_confianca, win_rate_hist, total_trd))

        # Guia prático de Ordem OCO com Position Sizing calculado exatamente
        if sinal['ativo_recomendado']:
            preco_compra = sinal['preco']
            valor_stop_sugerido = preco_compra * (1 - stop_loss)
            valor_gain_sugerido = preco_compra * (1 + take_profit)
            
            # Cálculo exato do valor alocado para esta operação conforme as travas configuradas
            risco_fin = capital_inicial * risco_por_trade_pct
            tam_por_risco = risco_fin / stop_loss
            tam_max = capital_inicial * max_alocacao_pct
            valor_alocado_atual = min(capital_inicial, tam_por_risco, tam_max)
            
            st.markdown("#### ⚙️ Ficha de Ordens para a Corretora (Ordem OCO)")
            st.markdown(f"""
            * **Ativo:** `{res['ticker']}`
            * **Tipo de Ordem:** Compra a Mercado / Limite em `{preco_compra:,.2f}`
            * **💰 Valor a ser Operado (Alocado):** `R$ / $ {valor_alocado_atual:,.2f}` (Respeitando o teto de {max_alocacao_input}% da banca e risco de {risco_por_trade_input}%)
            * **Preço de Stop Loss (Proteção):** `{valor_stop_sugerido:,.2f}` (-{stop_loss_input}%)
            * **Preço de Stop Gain (Alvo):** `{valor_gain_sugerido:,.2f}` (+{take_profit_input}%)
            """)

with aba2:
    st.subheader(f"📊 Validação Histórica & Backtest: {ticker_selecionado}")
    st.write("Resultados obtidos testando a estratégia no passado do ativo (Simulação pura para auditoria de desempenho).")

    if 'res_individual' in st.session_state:
        res = st.session_state['res_individual']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Capital Final", f"$ {res['capital_final']:,.2f}", f"${res['lucro_liquido']:+,.2f}")
        col2.metric("Taxa de Acerto (Win Rate)", f"{res['win_rate']:.1f}%")
        col3.metric("Total de Operações", res['total_trades'])
        col4.metric("Max Drawdown (Pior Queda)", f"{res['max_drawdown']:.2f}%")

        # Gráfico Plotly
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"Histórico de Trades no Passado ({res['ticker']})", "Evolução do Capital Líquido")
        )
        fig.add_trace(go.Scatter(x=res['df_test'].index, y=res['df_test']['Close'], name='Preço', line=dict(color='royalblue')), row=1, col=1)

        compras_x = [t['Data_Entrada'] for t in res['historico_trades']]
        compras_y = [t['Preco_Entrada'] for t in res['historico_trades']]
        vendas_x = [t['Data_Saida'] for t in res['historico_trades']]
        vendas_y = [t['Preco_Saida'] for t in res['historico_trades']]

        fig.add_trace(go.Scatter(x=compras_x, y=compras_y, mode='markers', name='Compra', marker=dict(color='limegreen', size=10, symbol='triangle-up')), row=1, col=1)
        fig.add_trace(go.Scatter(x=vendas_x, y=vendas_y, mode='markers', name='Venda', marker=dict(color='crimson', size=10, symbol='triangle-down')), row=1, col=1)
        fig.add_trace(go.Scatter(x=res['df_test'].index, y=res['df_test']['Capital_Estrategia'], name='Capital', line=dict(color='mediumseagreen', width=2)), row=2, col=1)

        fig.update_layout(height=700, template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Histórico Detalhado de Operações Passadas")
        if res['historico_trades']:
            df_trades = pd.DataFrame(res['historico_trades'])
            df_trades['Data_Entrada'] = pd.to_datetime(df_trades['Data_Entrada']).dt.strftime('%Y-%m-%d')
            df_trades['Data_Saida'] = pd.to_datetime(df_trades['Data_Saida']).dt.strftime('%Y-%m-%d')
            st.dataframe(df_trades, use_container_width=True)
        else:
            st.info("Nenhuma operação concluída no período de teste.")
    else:
        st.info("Clique em 'Verificar Sinal Atual' na aba 1 para carregar também os dados de backtest deste ativo.")