import streamlit as st
from fugle_marketdata import RestClient
import pandas as pd
import pandas_ta as ta  # 技術指標計算神器
import json
from datetime import datetime, timedelta

# --- 設定頁面 ---
st.set_page_config(page_title="AI 股市戰情室 Pro", page_icon="🦅", layout="wide")
st.title("🦅 股市全域戰情 (Data to Gemini)")

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定參數")
    api_key = st.text_input("Fugle API Key", type="password")
    symbol = st.text_input("股票代號", value="2383")
    timeframe = st.selectbox("K線週期", ["1T", "5T", "30T", "60T"], index=2, help="T代表分鐘")
    
    st.markdown("---")
    st.markdown("### 📊 指標參數")
    ma_short = st.number_input("短均線 (MA)", value=5)
    ma_long = st.number_input("長均線 (MA)", value=20)
    rsi_len = st.number_input("RSI 週期", value=6)

# --- 核心函數：處理 K 棒與指標 ---
def process_candles(symbol, api_key, timeframe):
    client = RestClient(api_key=api_key)
    stock = client.stock
    
    # 1. 抓取最近的 K 棒 (Intraday Candles)
    # Fugle 回傳的是 1 分鐘 K 棒，我們抓多一點來重取樣
    candles = stock.intraday.candles(symbol=symbol)
    
    if 'data' not in candles or not candles['data']:
        return None, "抓不到 K 棒資料"

    # 2. 轉成 DataFrame
    df = pd.DataFrame(candles['data'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # 欄位重新命名以符合 pandas_ta 習慣
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    # 3. 重取樣 (Resample) - 把 1 分K 合成為 30分K / 60分K
    # 邏輯：開盤價取第一筆，收盤價取最後一筆，高點取最大，低點取最小，成交量加總
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    df_resampled = df.resample(timeframe).apply(ohlc_dict).dropna()

    # 4. 計算技術指標 (使用 pandas_ta)
    # MA (移動平均)
    df_resampled[f'MA{ma_short}'] = ta.sma(df_resampled['Close'], length=ma_short)
    df_resampled[f'MA{ma_long}'] = ta.sma(df_resampled['Close'], length=ma_long)
    
    # RSI (相對強弱)
    df_resampled[f'RSI{rsi_len}'] = ta.rsi(df_resampled['Close'], length=rsi_len)
    
    # MACD
    macd = ta.macd(df_resampled['Close'])
    # 將 MACD 欄位合併進來 (MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9)
    df_resampled = pd.concat([df_resampled, macd], axis=1)

    return df_resampled, None

# --- 主程式 ---
if st.button("🚀 啟動全域掃描"):
    if not api_key:
        st.error("請輸入 API Key")
    else:
        try:
            # 1. 執行運算
            df, error = process_candles(symbol, api_key, timeframe)
            
            if error:
                st.error(error)
            else:
                # 2. 取得現價 (用於確認)
                current_price = df['Close'].iloc[-1]
                st.metric(f"{symbol} 目前 ({timeframe}) 收盤價", current_price)

                # 3. 整理 JSON 給 Gemini
                # 我們只取「最後 5 根」K棒給 Gemini 就好，不然資料太多
                last_n = 5
                output_df = df.tail(last_n).copy()
                
                # 格式化時間變成字串
                output_df.index = output_df.index.strftime('%Y-%m-%d %H:%M:%S')
                
                # 轉成 Dict
                k_data = output_df.to_dict(orient='index')

                gemini_payload = {
                    "stock": symbol,
                    "timeframe": timeframe,
                    "analysis_needed": "請根據 MA 排列、RSI 背離與 MACD 柱狀圖分析趨勢",
                    "technical_data": k_data
                }

                json_str = json.dumps(gemini_payload, indent=2, ensure_ascii=False)

                # 4. 顯示結果
                st.subheader("📋 複製這串 JSON 給教練")
                st.code(json_str, language='json')
                
                # 畫個簡單的圖自己看爽的
                st.line_chart(df[['Close', f'MA{ma_short}', f'MA{ma_long}']].tail(50))

        except Exception as e:
            st.error(f"系統錯誤: {e}")