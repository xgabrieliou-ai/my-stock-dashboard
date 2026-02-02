import streamlit as st
import pandas as pd
import pandas_ta as ta  # 技術指標神器
from fugle_marketdata import RestClient
import google.generativeai as genai
import plotly.graph_objects as go
import json

# --- 頁面設定 ---
st.set_page_config(page_title="AI 股市戰情室 (富果直連版)", layout="wide", page_icon="🦅")

# --- 1. 資料抓取與處理 (您的核心邏輯) ---
def fetch_fugle_data(api_key, symbol, timeframe):
    try:
        client = RestClient(api_key=api_key)
        stock = client.stock
        
        # 抓取盤中 K 棒 (Fugle 回傳的是 1分K)
        candles = stock.intraday.candles(symbol=symbol)
        
        if 'data' not in candles or not candles['data']:
            return None, "❌ 抓不到資料，請確認股票代號或市場是否開盤"

        # 轉成 DataFrame
        df = pd.DataFrame(candles['data'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # 欄位重新命名
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

        # 重取樣 (Resample) - 這段邏輯非常棒，保留！
        # 如果 timeframe 是 '1T' 就不需要重取樣，但為了統一邏輯還是跑一次
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        df_resampled = df.resample(timeframe).apply(ohlc_dict).dropna()

        # 如果資料太少，無法計算指標
        if len(df_resampled) < 20:
            return None, "⚠️ 資料筆數不足以計算均線與指標，請稍晚再試"

        # --- 計算技術指標 (使用 pandas_ta) ---
        # 1. 均線
        df_resampled['MA5'] = ta.sma(df_resampled['Close'], length=5)
        df_resampled['MA20'] = ta.sma(df_resampled['Close'], length=20)
        
        # 2. RSI
        df_resampled['RSI'] = ta.rsi(df_resampled['Close'], length=6)
        
        # 3. KD (Stoch)
        stoch = ta.stoch(df_resampled['High'], df_resampled['Low'], df_resampled['Close'], k=9, d=3, smooth_k=3)
        # pandas_ta 回傳的欄位名稱通常是 STOCHk_9_3_3, STOCHd_9_3_3
        df_resampled = pd.concat([df_resampled, stoch], axis=1)
        # 重新命名方便後續使用
        df_resampled['K'] = df_resampled['STOCHk_9_3_3']
        df_resampled['D'] = df_resampled['STOCHd_9_3_3']

        # 4. 布林通道 (Bollinger Bands)
        bbands = ta.bbands(df_resampled['Close'], length=20, std=2)
        df_resampled = pd.concat([df_resampled, bbands], axis=1)
        df_resampled['BB_Upper'] = df_resampled['BBU_20_2.0']
        df_resampled['BB_Lower'] = df_resampled['BBL_20_2.0']

        return df_resampled, None

    except Exception as e:
        return None, f"Fugle API 連線錯誤: {str(e)}"

# --- 2. 本地快速訊號掃描 ---
def local_signal_scan(df):
    if df is None or len(df) < 1: return "等待數據...", "grey", []
    
    last = df.iloc[-1]
    signals = []
    score = 0
    
    # KD 判斷
    if pd.notna(last['K']) and pd.notna(last['D']):
        if last['K'] > last['D']:
            signals.append(f"🔸 KD 金叉 (K:{last['K']:.1f} > D:{last['D']:.1f})")
            score += 1
        else:
            signals.append(f"🔹 KD 死叉 (K:{last['K']:.1f} < D:{last['D']:.1f})")
            score -= 1
        if last['K'] < 20: signals.append("💎 KD 超賣 (<20)")

    # MA 判斷
    if pd.notna(last['MA20']):
        if last['Close'] > last['MA20']:
            signals.append("✅ 站上月線 (偏多)")
            score += 1
        else:
            signals.append("🔻 跌破月線 (偏空)")
            score -= 1

    # 總結
    if score >= 2: return "🚀 強力多頭訊號", "success", signals
    elif score >= 1: return "📈 偏多震盪", "info", signals
    elif score <= -2: return "🐻 強力空頭訊號", "error", signals
    elif score <= -1: return "📉 偏空震盪", "warning", signals
    else: return "⚖️ 盤整 / 訊號不明", "secondary", signals

# --- 3. Gemini AI 分析 ---
def ask_gemini(stock_symbol, df):
    if "GEMINI_API_KEY" not in st.secrets:
        return "❌ 錯誤：找不到 Gemini Key，請檢查 secrets.toml"
    
    api_key = st.secrets["GEMINI_API_KEY"]
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 只取最後 5 筆給 AI
        recent_data = df.tail(5)[['Open', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'K', 'D', 'BB_Upper', 'BB_Lower']]
        # 時間轉字串
        recent_data.index = recent_data.index.strftime('%H:%M')
        json_data = recent_data.to_json(orient="index")

        prompt = f"""
        你是一位專業的台股當沖教練。
        這是一份來自 Fugle 的即時數據，股票代號：{stock_symbol}。
        
        數據 (最後5根K棒)：
        {json_data}
        
        請給我「快、狠、準」的分析：
        1. **多空判斷**：目前趨勢為何？
        2. **操作指令**：現在該買、該賣還是空手？(給出明確的進出價位建議)
        3. **風險提示**：注意什麼？(如量能不足、指標背離)
        """
        
        with st.spinner("🤖 AI 教練正在連線思考中..."):
            response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Gemini 連線錯誤: {str(e)}"

# --- 主程式 ---
def main():
    st.title("🦅 AI 股市戰情室 (Fugle 直連版)")
    
    # 檢查 Keys
    if "FUGLE_API_KEY" in st.secrets and "GEMINI_API_KEY" in st.secrets:
        st.sidebar.success("✅ 雙鑰匙 (Fugle/Gemini) 已載入")
    else:
        st.sidebar.error("❌ 缺少 API Key，請檢查 secrets.toml")
        return

    # 側邊欄
    with st.sidebar:
        st.header("⚙️ 參數設定")
        symbol = st.text_input("股票代號", value="6274").upper() # 預設台燿
        timeframe = st.selectbox("K線週期", ["1T", "5T", "30T", "60T"], index=1, help="T=分鐘")
        
        if st.button("🚀 啟動掃描", type="primary"):
            st.session_state['run_scan'] = True

    # 執行掃描邏輯
    if st.session_state.get('run_scan'):
        df, error = fetch_fugle_data(st.secrets["FUGLE_API_KEY"], symbol, timeframe)
        
        if error:
            st.error(error)
        else:
            # 版面配置
            col_chart, col_ai = st.columns([2, 1])
            
            with col_chart:
                last_bar = df.iloc[-1]
                st.subheader(f"📊 {symbol} ({timeframe}) 走勢")
                
                # 數據看板
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("收盤價", f"{last_bar['Close']}")
                m2.metric("RSI", f"{last_bar['RSI']:.1f}")
                m3.metric("KD (K)", f"{last_bar['K']:.1f}")
                m4.metric("成交量", f"{int(last_bar['Volume'])}")

                # 繪圖 (Plotly)
                fig = go.Figure(data=[go.Candlestick(x=df.index,
                                open=df['Open'], high=df['High'],
                                low=df['Low'], close=df['Close'], name='K線')])
                
                # 加均線
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20(月線)'))
                # 加布林
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='布林上軌'))
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='布林下軌'))

                fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            with col_ai:
                st.subheader("🤖 戰情判讀")
                
                # 1. 快速掃描
                summary, color, signals = local_signal_scan(df)
                if color == "success": st.success(summary)
                elif color == "error": st.error(summary)
                elif color == "warning": st.warning(summary)
                else: st.info(summary)
                
                with st.expander("訊號細節"):
                    for s in signals: st.write(s)

                st.divider()

                # 2. Gemini 深度分析
                if st.button("🧠 呼叫 AI 教練", type="primary"):
                    analysis = ask_gemini(symbol, df)
                    st.markdown(analysis)

if __name__ == "__main__":
    main()
