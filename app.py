import streamlit as st
from fugle_marketdata import RestClient
import pandas as pd
import pandas_ta as ta
import json
from datetime import datetime

# --- 設定頁面 ---
st.set_page_config(page_title="AI 股市指揮所 (Ultimate)", page_icon="🦅", layout="wide")
st.title("🦅 股市全域戰情 (Ultimate Ver.)")

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    
    # --- 智慧鑰匙判斷邏輯 (修改這段) ---
    if 'fugle_api_key' in st.secrets:
        # 如果雲端後台有設定，直接讀取，不顯示輸入框
        api_key = st.secrets['fugle_api_key']
        st.success("✅ API Key 已從雲端載入") 
    else:
        # 如果後台沒設定 (例如在本機跑且沒 secrets.toml)，才顯示輸入框
        api_key = st.text_input("Fugle API Key", type="password")
    # ----------------------------------

    symbol = st.text_input("股票代號", value="3231")
    timeframe = st.selectbox("K線週期", ["1T", "5T", "30T", "60T"], index=1)
    
    st.markdown("### 📊 指標參數")
    ma_short = st.number_input("短均線 (MA)", value=5)
    # 這裡如果不夠長，計算會回傳 null，但不影響程式運行
    ma_long = st.number_input("長均線 (MA)", value=20) 

def get_signal(row):
    # 簡單的訊號判讀，顯示在畫面上給人看
    signal = []
    if row['RSI'] < 20: signal.append("🟢RSI超賣")
    if row['RSI'] > 80: signal.append("🔴RSI過熱")
    if row['k'] < 20 and row['k'] > row['d']: signal.append("⚡KD金叉(低檔)")
    return " ".join(signal) if signal else "觀察中"
    
# --- 新增這個函數用來抓中文名稱 ---
@st.cache_data(ttl=86400)  # 🌟 重點：快取存 24 小時，超級省錢！
def get_stock_name(symbol, api_key):
    try:
        client = RestClient(api_key=api_key)
        # 呼叫 quote API 取得簡介，只為了拿 name
        quote = client.stock.intraday.quote(symbol=symbol)
        return quote.get('name', symbol)  # 如果抓不到就回傳代號
    except Exception:
        return symbol
        
def process_data(symbol, api_key, timeframe):
    client = RestClient(api_key=api_key)
    stock = client.stock
    
    # 抓取 Intraday Candles
    candles = stock.intraday.candles(symbol=symbol)
    if 'data' not in candles or not candles['data']:
        return None, "抓不到資料，請確認開盤中或 Key 正確"

    df = pd.DataFrame(candles['data'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    # 重取樣 (Resample)
    ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    df_res = df.resample(timeframe).apply(ohlc_dict).dropna()

    # --- 1. 計算均線 ---
    df_res[f'MA{ma_short}'] = ta.sma(df_res['Close'], length=ma_short)
    df_res[f'MA{ma_long}'] = ta.sma(df_res['Close'], length=ma_long)

    # --- 2. 計算 RSI ---
    df_res['RSI'] = ta.rsi(df_res['Close'], length=6)

    # --- 3. 計算 MACD ---
    macd = ta.macd(df_res['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df_res = pd.concat([df_res, macd], axis=1)

    # --- 4. 計算 KD (Stochastic) --- 🌟 新增
    # k=9, d=3, smooth_d=3
    stoch = ta.stoch(df_res['High'], df_res['Low'], df_res['Close'], k=9, d=3, smooth_k=3)
    if stoch is not None:
        df_res = pd.concat([df_res, stoch], axis=1)
        # pandas_ta 欄位名稱通常是 STOCHk_9_3_3, STOCHd_9_3_3，我們簡化它
        df_res['k'] = df_res['STOCHk_9_3_3']
        df_res['d'] = df_res['STOCHd_9_3_3']

# --- 5. 計算布林通道 (Bollinger Bands) --- 🌟 修正版
    bbands = ta.bbands(df_res['Close'], length=20, std=2)
    if bbands is not None:
        df_res = pd.concat([df_res, bbands], axis=1)
        
        # 🌟 修正重點：自動抓取欄位名稱 (不用猜是 2.0 還是 2)
        # 直接從產生的欄位裡，找出 BBU (上軌) 和 BBL (下軌) 開頭的
        for col in bbands.columns:
            if col.startswith("BBU"):  # 抓上軌
                df_res['BB_Upper'] = df_res[col]
            elif col.startswith("BBL"): # 抓下軌
                df_res['BB_Lower'] = df_res[col]

    return df_res, None

if st.button("🚀 啟動全域掃描"):
    if not api_key:
        st.error("請輸入 API Key")
    else:
        try:
            # 1. 先抓名字 (這會用快取，不扣次數)
            stock_name = get_stock_name(symbol, api_key)
            
            # 2. 再顯示大標題
            st.subheader(f"📊 {stock_name} ({symbol}) - {timeframe}")

            # 3. 接著才是原本的運算 (這會用 60秒快取)
            df, error = process_data(symbol, api_key, timeframe)
            if error:
                st.error(error)
            else:
                # 取得最新一筆資料
                latest = df.iloc[-1]
                
                # 畫面顯示即時重點
                col1, col2, col3 = st.columns(3)
                col1.metric("現價", f"{latest['Close']}", f"{latest['Volume']:.0f} 張")
                col2.metric("RSI (6)", f"{latest['RSI']:.2f}")
                
                # 處理 KD 顯示 (如果資料不足會是 NaN)
                k_val = f"{latest.get('k', 0):.2f}" if pd.notna(latest.get('k')) else "N/A"
                col3.metric("KD (K值)", k_val)

                st.info(f"AI 訊號掃描: {get_signal(latest)}")

                # 準備 JSON
                output_df = df.tail(5).copy()
                output_df.index = output_df.index.strftime('%H:%M')
                
                # 清理 NaN (JSON 不支援 NaN)
                output_df = output_df.fillna("資料不足")
                
                technical_data = output_df.to_dict(orient='index')

                payload = {
                    "stock": f"{stock_name} ({symbol})",
                    "timeframe": timeframe,
                    "indicators": {
                        "MA": f"MA{ma_short} vs MA{ma_long}",
                        "RSI": "RSI(6)",
                        "MACD": "12,26,9",
                        "KD": "9,3,3 (Slow)",
                        "Bollinger": "20, 2"
                    },
                    "data": technical_data
                }
                
                json_str = json.dumps(payload, indent=2, ensure_ascii=False)
                
                st.subheader("📋 複製這串給 Gemini")
                st.code(json_str, language='json')
                
                # 簡單畫圖：K值與 D值
                if 'k' in df.columns:
                    st.line_chart(df[['k', 'd']].tail(50))
                
        except Exception as e:
            st.error(f"發生錯誤: {e}")





