import streamlit as st
import pandas as pd
import numpy as np # 用來做基礎運算
from fugle_marketdata import RestClient
import google.generativeai as genai
import plotly.graph_objects as go
import json

# --- 頁面設定 ---
st.set_page_config(page_title="AI 股市戰情室 (Fugle 輕量版)", layout="wide", page_icon="🦅")

# --- 0. 核心：手寫技術指標 (不依賴 pandas_ta，避免報錯) ---
def calculate_indicators_manual(df):
    """
    使用純 Pandas 計算指標，避開 Numba/Pandas_TA 的相容性地獄
    """
    # 1. MA (移動平均)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # 2. RSI (相對強弱 - 參數 6)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=6).mean()
    avg_loss = loss.rolling(window=6).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. KD (隨機指標 - 9,3,3)
    # RSV = (今日收盤 - 最近9天最低) / (最近9天最高 - 最近9天最低) * 100
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    # K = 2/3 * 昨日K + 1/3 * 今日RSV (使用 ewm 模擬遞迴運算, com=2 等同於 alpha=1/3)
    df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()

    # 4. 布林通道 (20, 2)
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (std * 2)
    df['BB_Lower'] = df['MA20'] - (std * 2)
    
    return df

# --- 1. 資料抓取 ---
def fetch_fugle_data(api_key, symbol, timeframe):
    try:
        client = RestClient(api_key=api_key)
        stock = client.stock
        
        # 抓取盤中 K 棒
        candles = stock.intraday.candles(symbol=symbol)
        
        if 'data' not in candles or not candles['data']:
            return None, "❌ 抓不到資料，請確認股票代號 (富果代號如 2330)"

        # 轉成 DataFrame
        df = pd.DataFrame(candles['data'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # 欄位重新命名
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

        # 重取樣 (Resample)
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        df_resampled = df.resample(timeframe).apply(ohlc_dict).dropna()

        if len(df_resampled) < 20:
            return None, "⚠️ 資料筆數不足 (<20筆)，無法計算均線，請稍晚再試"

        # --- 使用手寫函數計算指標 ---
        df_resampled = calculate_indicators_manual(df_resampled)

        return df_resampled, None

    except Exception as e:
        return None, f"Fugle API 連線錯誤: {str(e)}"

# --- 2. 本地快速訊號 ---
def local_signal_scan(df):
    if df is None or len(df) < 1: return "等待數據...", "grey", []
    last = df.iloc[-1]
    signals = []
    score = 0
    
    # KD
    if pd.notna(last['K']) and pd.notna(last['D']):
        if last['K'] > last['D']:
            signals.append(f"🔸 KD 金叉 (K:{last['K']:.1f} > D:{last['D']:.1f})")
            score += 1
        else:
            signals.append(f"🔹 KD 死叉 (K:{last['K']:.1f} < D:{last['D']:.1f})")
            score -= 1
        if last['K'] < 20: signals.append("💎 KD 超賣 (<20)")

    # RSI
    if pd.notna(last['RSI']):
        if last['RSI'] < 25: signals.append("💎 RSI 超賣 (<25)")
        elif last['RSI'] > 75: signals.append("🔥 RSI 過熱 (>75)")

    # MA & 布林
    if pd.notna(last['MA20']):
        if last['Close'] > last['MA20']:
            signals.append("✅ 站上月線")
            score += 1
        else:
            signals.append("🔻 跌破月線")
            score -= 1
    
    if pd.notna(last['BB_Upper']) and last['Close'] > last['BB_Upper']:
        signals.append("🚀 衝破布林上軌")
        score += 1

    if score >= 2: return "🚀 強力多頭訊號", "success", signals
    elif score >= 1: return "📈 偏多震盪", "info", signals
    elif score <= -2: return "🐻 強力空頭訊號", "error", signals
    elif score <= -1: return "📉 偏空震盪", "warning", signals
    else: return "⚖️ 盤整 / 訊號不明", "secondary", signals

# --- 3. Gemini AI 分析 ---
def ask_gemini(stock_symbol, df):
    if "GEMINI_API_KEY" not in st.secrets:
        return "❌ 錯誤：找不到 Gemini Key"
    
    api_key = st.secrets["GEMINI_API_KEY"]
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 取最後 5 筆
        recent = df.tail(5)[['Open', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'K', 'D', 'BB_Upper', 'BB_Lower']]
        recent.index = recent.index.strftime('%H:%M')
        json_data = recent.to_json(orient="index")

        prompt = f"""
        你是一位專業的台股當沖教練。
        股票代號：{stock_symbol}。
        數據 (最後5根K棒)：{json_data}
        
        請給出「快狠準」的診斷：
        1. **多空判斷**：目前趨勢？
        2. **操作建議**：現在該買、賣還是觀望？(給出價位)
        3. **風險提示**：注意什麼？
        """
        
        with st.spinner("🤖 AI 教練正在分析..."):
            response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Gemini 連線錯誤: {str(e)}"

# --- 主程式 ---
def main():
    st.title("🦅 AI 股市戰情室 (Fugle 直連版)")
    
    # 檢查 Keys
    if "FUGLE_API_KEY" in st.secrets and "GEMINI_API_KEY" in st.secrets:
        st.sidebar.success("✅ 雙鑰匙已載入")
    else:
        st.sidebar.error("❌ 缺少 API Key，請檢查 secrets.toml")
        return

    # 側邊欄
    with st.sidebar:
        st.header("⚙️ 參數設定")
        symbol = st.text_input("股票代號", value="6274").upper()
        timeframe = st.selectbox("K線週期", ["1T", "5T", "15T", "30T", "60T"], index=1)
        
        if st.button("🚀 啟動掃描", type="primary"):
            st.session_state['run_scan'] = True

    if st.session_state.get('run_scan'):
        df, error = fetch_fugle_data(st.secrets["FUGLE_API_KEY"], symbol, timeframe)
        
        if error:
            st.error(error)
        else:
            col_chart, col_ai = st.columns([2, 1])
            
            with col_chart:
                last_bar = df.iloc[-1]
                st.subheader(f"📊 {symbol} ({timeframe}) 走勢")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("收盤價", f"{last_bar['Close']}")
                m2.metric("RSI", f"{last_bar['RSI']:.1f}")
                m3.metric("KD (K)", f"{last_bar['K']:.1f}")
                m4.metric("成交量", f"{int(last_bar['Volume'])}")

                fig = go.Figure(data=[go.Candlestick(x=df.index,
                                open=df['Open'], high=df['High'],
                                low=df['Low'], close=df['Close'], name='K線')])
                
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'))
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='上軌'))
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='下軌'))

                fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            with col_ai:
                st.subheader("🤖 戰情判讀")
                summary, color, signals = local_signal_scan(df)
                if color == "success": st.success(summary)
                elif color == "error": st.error(summary)
                elif color == "warning": st.warning(summary)
                else: st.info(summary)
                
                with st.expander("訊號細節"):
                    for s in signals: st.write(s)

                st.divider()

                if st.button("🧠 呼叫 AI 教練", type="primary"):
                    analysis = ask_gemini(symbol, df)
                    st.markdown(analysis)

if __name__ == "__main__":
    main()
