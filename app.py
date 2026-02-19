import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks

# 嘗試導入 ssqueezepy
try:
    from ssqueezepy import ssq_cwt
    HAS_SSQ = True
except ImportError:
    HAS_SSQ = False

# ==========================================
# 1. 核心分析函式 (SST + 優化版 Peak Finding)
# ==========================================
def perform_clean_multiridge_sst(data, fps, wavelet, nv, y_min, y_max, show_ridge, ridge_thresh_percent, min_dist):
    """
    執行 SST 並使用優化過的參數找出乾淨的諧波路徑
    """
    st.write(f"➡️ 計算 SST (Wavelet: {wavelet}, Voices: {nv})...")

    try:
        # 1. 計算 SST
        Tx, Wx, ssq_freqs, scales = ssq_cwt(data, wavelet=wavelet, fs=fps, nv=nv)
    except Exception as e:
        st.error(f"SST 計算錯誤: {e}")
        return go.Figure()

    # 2. 取能量幅度
    magnitude = np.abs(Tx)
    
    # 3. 座標轉換
    with np.errstate(divide='ignore'): 
        periods = 1 / ssq_freqs
    
    # 4. 建立時間軸
    time_axis = np.arange(len(data)) / fps
    
    # 5. 建立 Plotly 圖表
    fig = go.Figure()
    
    # 過濾顯示範圍
    valid_mask = np.isfinite(periods)
    plot_periods = periods[valid_mask]
    plot_magnitude = magnitude[valid_mask, :]

    # 畫熱圖
    fig.add_trace(go.Heatmap(
        z=plot_magnitude, 
        x=time_axis, 
        y=plot_periods, 
        colorscale='Jet', # 建議改用 Jet 或 Turbo 對比度較高
        colorbar=dict(title='能量幅度'),
        name='SST 能量譜',
        hovertemplate='時間: %{x:.2f}s<br>週期: %{y:.4f}s<br>能量: %{z:.2f}<extra></extra>'
    ))

    # ==========================================
    # 優化版：多重脊線偵測
    # ==========================================
    if show_ridge:
        st.caption("🔍 正在進行特徵提取 (Peak Peaking)...")
        
        ridge_x = []
        ridge_y = []
        
        # 1. 設定能量門檻 (過濾背景雜訊)
        global_max_energy = np.max(magnitude)
        abs_threshold = global_max_energy * ridge_thresh_percent
        
        # 2. 針對每一個時間點掃描
        num_time_steps = magnitude.shape[1]
        
        for t_idx in range(num_time_steps):
            spectrum_slice = magnitude[:, t_idx]
            
            # === 關鍵修改在這裡 ===
            # distance: 設定兩個峰值之間至少要隔多少個 index
            # 這能避免同一條粗線上出現兩個點，強迫只抓最高點
            peaks, _ = find_peaks(
                spectrum_slice, 
                height=abs_threshold, 
                distance=min_dist  # <--- 這行是讓線條變乾淨的關鍵
            )
            
            if len(peaks) > 0:
                current_periods = periods[peaks]
                current_time = time_axis[t_idx]
                
                ridge_x.extend([current_time] * len(peaks))
                ridge_y.extend(current_periods)

        # 畫出偵測點 (改小一點的白點)
        fig.add_trace(go.Scatter(
            x=ridge_x,
            y=ridge_y,
            mode='markers',
            marker=dict(symbol='circle', color='white', size=2, opacity=0.8), # 點縮小到 size=2
            name='提取的諧波 (Clean Peaks)',
            hoverinfo='skip'
        ))

    # 6.圖表設定
    fig.update_layout(
        title=f'SST 高精度諧波分析', 
        xaxis_title='時間 (秒)', 
        yaxis_title='週期 (秒)', 
        height=700,
        yaxis_type="log", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    if y_min > 0 and y_max > 0:
        fig.update_yaxes(range=[np.log10(y_min), np.log10(y_max)])
    
    return fig

# ==========================================
# 2. 資料讀取函式 (不變)
# ==========================================
def load_uploaded_npy(uploaded_file):
    try:
        data = np.load(uploaded_file, allow_pickle=True)
        if data.ndim == 1:
            return data.astype(float)
        elif data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 1].astype(float)
        else:
            return None
    except:
        return None

# ==========================================
# 3. Streamlit 介面配置
# ==========================================
st.set_page_config(page_title="SST 諧波優化版", layout="wide")
st.title("📊 SST 諧波分析 (優化抗噪版)")

if not HAS_SSQ:
    st.error("請安裝套件: pip install ssqueezepy scipy")
    st.stop()

with st.sidebar:
    st.header("⚙️ 參數設定")
    fps = st.number_input("取樣率 (FPS)", value=30.0, min_value=1.0)

    st.subheader("1. SST 參數")
    sst_wavelet = st.selectbox("小波基底", ['morlet', 'bump'], index=0)
    nv = st.select_slider("頻率解析度 (Voices)", options=[16, 32, 64], value=32)

    st.subheader("2. 諧波提取 (重點調整區)")
    show_ridge = st.checkbox("顯示提取結果", value=True)
    
    # --- [關鍵參數 1] 能量過濾 ---
    ridge_thresh = st.slider(
        "⚡ 能量過濾門檻 (%)", 
        min_value=1, 
        max_value=30, 
        value=5, 
        step=1,
        help="數值越大，只有越紅(能量越強)的線才會被標示。若背景雜訊很多，請調大此值。"
    )

    # --- [關鍵參數 2] 最小間距 ---
    min_dist = st.slider(
        "↔️ 最小峰值間距 (Pixel Distance)", 
        min_value=1, 
        max_value=50, 
        value=10, 
        step=1,
        help="數值越大，線條越乾淨(不會有重影)，但如果兩條諧波靠太近可能會被合併成一條。建議值 10~20。"
    )

    st.subheader("3. 顯示範圍")
    col1, col2 = st.columns(2)
    with col1:
        y_axis_min = st.number_input("Min 週期(s)", value=0.1, format="%.2f")
    with col2:
        y_axis_max = st.number_input("Max 週期(s)", value=10.0, format="%.1f")

# --- 主畫面 ---
uploaded_file = st.file_uploader("上傳 .npy 數據檔案", type=["npy"])

if uploaded_file is not None:
    signal_data = load_uploaded_npy(uploaded_file)
    if signal_data is not None:
        signal_data = signal_data - np.mean(signal_data)
        st.line_chart(signal_data, height=150)

        fig_sst = perform_clean_multiridge_sst(
            data=signal_data, 
            fps=fps, 
            wavelet=sst_wavelet, 
            nv=nv,
            y_min=y_axis_min, 
            y_max=y_axis_max,
            show_ridge=show_ridge,
            ridge_thresh_percent=ridge_thresh/100.0,
            min_dist=min_dist # 傳入新參數
        )
        st.plotly_chart(fig_sst, use_container_width=True)
        
        st.info(f"""
        **調校指南：**
        1. 目前能量過濾門檻：**{ridge_thresh}%** (去除背景雜點)
        2. 目前最小間距：**{min_dist}** (去除線條重影/變細)
        - 如果圖上還有藍色區域的雜點 -> **調高** 「能量過濾門檻」。
        - 如果線條看起來很粗、很多點擠在一起 -> **調高** 「最小峰值間距」。
        """)
