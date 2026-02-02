import streamlit as st
import pandas as pd
import numpy as np
from fugle_marketdata import RestClient
import google.generativeai as genai
import plotly.graph_objects as go
import json

# --- 頁面設定 ---
st.set_page_config(page_title="AI 股市戰情室 (Gemini 3 Flash)", layout="wide", page_icon="⚡")

# --- 0. 核心：手寫技術指標 (極速運算) ---
def calculate_indicators_manual(df):
    # MA
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # RSI (6)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=6).mean()
    avg_loss = loss.rolling(window=6).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # KD (9,3,3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()

    # 布林通道 (20, 2)
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (std * 2)
    df['BB_Lower'] = df['MA20'] - (std * 2)
    
    return df

# --- 1. 資料抓取 (Fugle) ---
def fetch_fugle_data(api_key, symbol, timeframe):
    try:
        client = RestClient(api_key=api_key)
        stock = client.stock
        
        candles = stock.intraday.candles(symbol=symbol)
        
        if 'data' not in candles or not candles['data']:
            return None, "❌ 抓不到資料 (請確認代號或是否開盤)"

        df = pd.DataFrame(candles['data'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

        ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        df_resampled = df.resample(timeframe).apply(ohlc_dict).dropna()

        if len(df_resampled) < 20:
            return None, "⚠️ 資料筆數不足 (<20筆)，無法計算指標"

        df_resampled = calculate_indicators_manual(df_resampled)
        return df_resampled, None

    except Exception as e:
        return None, f"Fugle API 錯誤: {str(e)}"

# --- 2. 本地訊號掃描 ---
def local_signal_scan(df):
    if df is None or len(df) < 1: return "等待數據...", "grey", []
    last = df.iloc[-1]
    signals = []
    score = 0
    
    # KD
    if pd.notna(last['K']) and pd.notna(last['D']):
        if last['K'] > last['D']:
            signals.append(f"🔸 KD 金叉 ({last['K']:.1f} > {last['D']:.1f})")
            score += 1
        else:
            signals.append(f"🔹 KD 死叉 ({last['K']:.1f} < {last['D']:.1f})")
            score -= 1
        if last['K'] < 20: signals.append("💎 KD 超賣 (<20)")

    # RSI
    if pd.notna(last['RSI']):
        if last['RSI'] < 25: signals.append("💎 RSI 超賣 (<25)")
        elif last['RSI'] > 75: signals.append("🔥 RSI 過熱 (>75)")

    # MA
    if pd.notna(last['MA20']):
        if last['Close'] > last['MA20']:
            signals.append("✅ 站上月線")
            score += 1
        else:
            signals.append("🔻 跌破月線")
            score -= 1

    if score >= 2: return "🚀 強力多頭訊號", "success", signals
    elif score >= 1: return "📈 偏多震盪", "info", signals
    elif score <= -2: return "🐻 強力空頭訊號", "error", signals
    elif score <= -1: return "📉 偏空震盪", "warning", signals
    else: return "⚖️ 盤整 / 訊號不明", "secondary", signals

# --- 3. Gemini 3.0 智能引擎 (核心升級) ---
def ask_gemini(stock_symbol, df):
    if "GEMINI_API_KEY" not in st.secrets:
        return "❌ 錯誤：找不到 GEMINI_API_KEY", "Unknown"
    
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    
    # 🔥 2026年最新模型優先順序
    model_candidates = [
        "gemini-3-flash-preview",  # 2026 主力：博士級推論 + 極速
        "gemini-2.5-flash",        # 2025 穩定版備援
        "gemini-2.0-flash"         # 最後防線
    ]
    
    used_model_name = ""
    response_text = ""

    # 自動尋找可用模型 (Auto-Fallback)
    for model_name in model_candidates:
        try:
            model = genai.GenerativeModel(model_name)
            
            recent = df.tail(5)[['Open', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'K', 'D', 'BB_Upper', 'BB_Lower']]
            recent.index = recent.index.strftime('%H:%M')
            json_data = recent.to_json(orient="index")

            prompt = f"""
            你是一位使用 Gemini 3 技術的頂尖台股教練。
            
            【戰情資料】
            標的：{stock_symbol}
            數據 (最新5根K棒)：{json_data}
            
            【分析指令】
            請利用你強大的邏輯推論能力，給出一個「快、狠、準」的交易決策：
            1. **多空定調**：一句話講完 (例如：多頭回檔守月線)。
            2. **關鍵攻防**：明確指出下檔支撐與上檔壓力價位。
            3. **操作建議**：
               - 如果空手：哪裡買？
               - 如果持有：續抱還是跑？
            4. **風險雷達**：有無背離或主力騙線跡象？

            (請用繁體中文，不需要客套，像戰場指揮官一樣直接下令)
            """
            
            response = model.generate_content(prompt)
            response_text = response.text
            used_model_name = model_name
            break # 成功就跳出
            
        except Exception:
            continue # 失敗就試下一個
    
    if not response_text:
        return "❌ 系統忙碌中，Gemini 所有模型暫時無法連線。", "None"
        
    return response_text, used_model_name

# --- 主程式 ---
def main():
    st.title("⚡ AI 股市戰情室 (Gemini 3 Flash)")
    st.caption("🚀 Powered by Google Gemini 3.0 Technology")
    
    if "FUGLE_API_KEY" in st.secrets and "GEMINI_API_KEY" in st.secrets:
        st.sidebar.success("✅ 雙鑰匙已載入")
    else:
        st.sidebar.error("❌ 缺少 API Key，請檢查 secrets.toml")
        return

    with st.sidebar:
        st.header("⚙️ 參數設定")
        symbol = st.text_input("股票代號", value="6274").upper()
        timeframe = st.selectbox("K線週期", ["1T", "5T", "15T", "30T", "60T"], index=1)
        
        if st.button("🚀 啟動 AI 掃描", type="primary"):
            st.session_state['run_scan'] = True

    if st.session_state.get('run_scan'):
        df, error = fetch_fugle_data(st.secrets["FUGLE_API_KEY"], symbol, timeframe)
        
        if error:
            st.error(error)
        else:
            col_chart, col_ai = st.columns([2, 1])
            
            with col_chart:
                last_bar = df.iloc[-1]
                st.subheader(f"📊 {symbol} ({timeframe}) K線圖")
                
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
                st.subheader("🤖 AI 戰情判讀")
                summary, color, signals = local_signal_scan(df)
                if color == "success": st.success(summary)
                elif color == "error": st.error(summary)
                elif color == "warning": st.warning(summary)
                else: st.info(summary)
                
                with st.expander("訊號細節"):
                    for s in signals: st.write(s)

                st.divider()

                if st.button("🧠 呼叫 Gemini 3.0", type="primary"):
                    with st.spinner("⚡ Gemini 3 Flash 正在高速推理中..."):
                        analysis, model_used = ask_gemini(symbol, df)
                        
                        # 顯示目前使用的引擎版本
                        if "gemini-3" in model_used:
                            st.caption(f"🚀 引擎：**{model_used}** (最新 V12 引擎)")
                        else:
                            st.caption(f"🛡️ 引擎：**{model_used}** (備援系統啟動)")
                            
                        st.markdown(analysis)

if __name__ == "__main__":
    main()
