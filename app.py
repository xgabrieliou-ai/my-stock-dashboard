import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import json

# --- 頁面設定 ---
st.set_page_config(page_title="AI 交易戰情室", layout="wide", page_icon="📈")

# --- 1. 技術指標計算核心 (不依賴外部 TA 套件，減少錯誤) ---
def calculate_indicators(df):
    if df is None or len(df) < 20:
        return df
    
    # MA (移動平均)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean() # 月線/布林中軌

    # RSI (相對強弱指標)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']

    # KD (隨機指標) - 使用 9,3,3
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2).mean() # 1/3權重約等於 com=2
    df['D'] = df['K'].ewm(com=2).mean()

    # Bollinger Bands (布林通道)
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (std * 2)
    df['BB_Lower'] = df['MA20'] - (std * 2)

    return df

# --- 2. 第一層：Python 規則基礎掃描 (快速) ---
def analyze_technical_signals_rule_based(df):
    if df is None or len(df) < 1:
        return "資料不足", [], "grey"

    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    score = 0  

    # MA 判斷
    if last['Close'] > last['MA20']:
        signals.append("✅ 股價站上月線 (短多)")
        score += 1
    else:
        signals.append("🔻 股價跌破月線 (短空)")
        score -= 1

    # KD 判斷
    if last['K'] > last['D']:
        signals.append("🔸 KD 黃金交叉 (轉強)")
        score += 1
    elif last['K'] < last['D']:
        signals.append("🔹 KD 死亡交叉 (轉弱)")
        score -= 1
    
    # RSI 判斷
    if last['RSI'] > 75:
        signals.append("🔥 RSI 過熱 (>75)")
        score += 0.5
    elif last['RSI'] < 25:
        signals.append("💎 RSI 超賣 (<25)") # 視為機會
        score += 0.5

    # 布林判斷
    if last['Close'] > last['BB_Upper']:
        signals.append("🚀 衝破布林上軌")
        score += 1
    elif last['Close'] < last['BB_Lower']:
        signals.append("💧 跌破布林下軌")
        score -= 1

    # 總結
    if score >= 2: return "🚀 強力多頭訊號", signals, "success"
    elif score >= 1: return "📈 偏多震盪", signals, "info"
    elif score <= -2: return "🐻 強力空頭訊號", signals, "error"
    elif score <= -1: return "📉 偏空震盪", signals, "warning"
    else: return "⚖️ 多空膠著 / 盤整", signals, "secondary"

# --- 3. 第二層：Gemini AI 深度分析 (大腦) ---
def ask_gemini_analysis(df):
    """將最近 5 根 K 棒數據整理成 JSON 餵給 Gemini"""
    
    # 檢查 Secrets 是否存在
    if "GEMINI_API_KEY" not in st.secrets:
        return "❌ 錯誤：找不到 API Key，請檢查 secrets.toml 設定。"
    
    api_key = st.secrets["GEMINI_API_KEY"]
    
    try:
        # 1. 整理數據 (只取最後 5 筆，減少 Token 消耗並聚焦當下)
        recent_data = df.tail(5).copy()
        # 格式化時間索引
        recent_data.index = recent_data.index.strftime('%Y-%m-%d %H:%M')
        # 轉成 JSON 字串
        data_json = recent_data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'RSI', 'K', 'D', 'BB_Upper', 'BB_Lower']].to_json(orient="index")

        # 2. 設定 Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # 3. 戰術指令 (Prompt)
        prompt = f"""
        你是一位身經百戰的台股當沖與波段交易教練。
        請根據以下最新的即時技術指標數據 (JSON 格式)，進行專業判讀。

        數據內容 (最後 5 根 K 棒)：
        {data_json}

        請給我一份簡短有力的「戰情診斷書」，包含以下部分：
        1. **【多空判斷】**：一句話定調 (例如：多頭回檔、空方破線、盤整待變)。
        2. **【關鍵價位】**：根據數據，指出下方的防守支撐價，與上方的壓力目標價。
        3. **【操作建議】**：針對持有者，現在該續抱、加碼還是停損？(請果斷一點)。
        4. **【風險警示】**：是否有背離、乖離過大或主力騙線的跡象？

        要求：使用繁體中文，語氣專業、冷靜、客觀。不要講模稜兩可的廢話。
        """

        # 4. 發送請求
        with st.spinner("🤖 AI 教練正在讀取盤勢..."):
            response = model.generate_content(prompt)
        
        return response.text

    except Exception as e:
        return f"Gemini 連線失敗: {str(e)}"

# --- 主程式 ---
def main():
    st.title("📈 AI 智能股票戰情室")

    # 側邊欄輸入
    with st.sidebar:
        st.header("參數設定")
        ticker_input = st.text_input("股票代號 (台股請加 .TW)", value="6274.TW").upper()
        interval = st.selectbox("K線週期", ["1m", "5m", "15m", "60m", "1d"], index=1)
        period = "5d" # 預設抓 5 天資料
        
        st.info("💡 範例：\n台積電: 2330.TW\n台燿: 6274.TW\n創意: 3443.TW")

    if ticker_input:
        # 1. 抓取資料
        try:
            df = yf.download(ticker_input, period=period, interval=interval, progress=False)
            
            if df.empty:
                st.error("❌ 找不到資料，請檢查代號是否正確 (台股記得加 .TW)")
                return

            # 2. 計算指標
            df = calculate_indicators(df)

            # 3. 畫面佈局
            col_chart, col_analysis = st.columns([2, 1])

            with col_chart:
                st.subheader(f"{ticker_input} - 走勢圖")
                
                # 繪製 K 線圖
                fig = go.Figure(data=[go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='K線')])
                
                # 加上布林通道與 MA
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='月線(MA20)'))
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='布林上軌'))
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='布林下軌'))

                fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                # 顯示最新報價數據
                last_bar = df.iloc[-1]
                cols = st.columns(4)
                cols[0].metric("收盤價", f"{last_bar['Close']:.2f}")
                cols[1].metric("RSI", f"{last_bar['RSI']:.2f}")
                cols[2].metric("KD (K)", f"{last_bar['K']:.2f}")
                cols[3].metric("月線", f"{last_bar['MA20']:.2f}")

            with col_analysis:
                st.subheader("🤖 AI 戰情判讀")
                
                # --- 第一層：Python 快速掃描 ---
                summary, signals, color = analyze_technical_signals_rule_based(df)
                
                st.markdown("### ⚡ 快速訊號掃描")
                if color == "success": st.success(summary)
                elif color == "error": st.error(summary)
                elif color == "warning": st.warning(summary)
                else: st.info(summary)

                with st.expander("查看訊號細節", expanded=True):
                    for s in signals:
                        st.write(s)

                st.divider()

                # --- 第二層：Gemini 深度分析 ---
                st.markdown("### 🧠 深度戰略分析")
                if st.button("呼叫 AI 教練診斷", type="primary", use_container_width=True):
                    analysis_result = ask_gemini_analysis(df)
                    st.markdown(analysis_result)
                    
                    with st.expander("查看傳送給 AI 的原始數據"):
                        st.dataframe(df.tail(5)[['Close', 'RSI', 'K', 'D', 'MA20']])

        except Exception as e:
            st.error(f"發生錯誤: {e}")

if __name__ == "__main__":
    main()
