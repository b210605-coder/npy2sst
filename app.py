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
# 1. 核心分析函式 (SST + 多重脊線偵測)
# ==========================================
def perform_multiridge_sst(data, fps, wavelet, nv, y_min, y_max, show_ridge, ridge_thresh_percent):
    """
    執行 SST 並找出每個時間點的所有諧波峰值 (Local Maxima)
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
    
    # 3. 座標轉換 (頻率 -> 週期)
    with np.errstate(divide='ignore'): 
        periods = 1 / ssq_freqs
    
    # 4. 建立時間軸
    time_axis = np.arange(len(data)) / fps
    
    # 5. 建立 Plotly 圖表 (底層熱圖)
    fig = go.Figure()
    
    # 過濾顯示範圍 (為了讓熱圖顏色更準確，先把範圍外的拿掉)
    valid_mask = np.isfinite(periods)
    plot_periods = periods[valid_mask]
    plot_magnitude = magnitude[valid_mask, :]

    fig.add_trace(go.Heatmap(
        z=plot_magnitude, 
        x=time_axis, 
        y=plot_periods, 
        colorscale='Jet',
        colorbar=dict(title='能量幅度'),
        name='SST 能量譜',
        hovertemplate='時間: %{x:.2f}s<br>週期: %{y:.4f}s<br>能量: %{z:.2f}<extra></extra>'
    ))

    # ==========================================
    # 多重脊線偵測 (Multi-Ridge Detection)
    # ==========================================
    if show_ridge:
        st.caption("正在提取所有諧波路徑...")
        
        ridge_x = []
        ridge_y = []
        
        # 設定絕對閾值：只抓出能量超過 "最大能量 * 百分比" 的峰值
        # 這樣可以過濾掉背景雜訊
        global_max_energy = np.max(magnitude)
        abs_threshold = global_max_energy * ridge_thresh_percent
        
        # 針對每一個時間點 (column) 進行 Peak Finding
        num_time_steps = magnitude.shape[1]
        
        for t_idx in range(num_time_steps):
            # 取得當下這一秒的頻譜切片 (1D array)
            spectrum_slice = magnitude[:, t_idx]
            
            # 使用 scipy.signal.find_peaks 找局部高點
            # height: 設定最小高度，過濾雜訊
            peaks, _ = find_peaks(spectrum_slice, height=abs_threshold)
            
            if len(peaks) > 0:
                # 找到峰值對應的週期
                current_periods = periods[peaks]
                current_time = time_axis[t_idx]
                
                # 收集座標用於繪圖
                # 這裡把同一個時間點的多個頻率都加進去
                ridge_x.extend([current_time] * len(peaks))
                ridge_y.extend(current_periods)

        # 畫出所有偵測到的脊線點 (黑點或白點)
        fig.add_trace(go.Scatter(
            x=ridge_x,
            y=ridge_y,
            mode='markers', # 使用點模式，因為多條線在數據結構上是不連續的
            marker=dict(symbol='circle', color='white', size=3, opacity=0.7),
            name='偵測到的諧波峰值 (Peaks)',
            hoverinfo='skip' 
        ))

    # 6.圖表設定
    fig.update_layout(
        title=f'SST 多重諧波偵測 (Multi-Ridge)', 
        xaxis_title='時間 (秒)', 
        yaxis_title='週期 (秒)', 
        height=700,
        yaxis_type="log", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # 強制設定 Y 軸顯示範圍
    if y_min > 0 and y_max > 0:
        fig.update_yaxes(range=[np.log10(y_min), np.log10(y_max)])
    
    return fig

# ==========================================
# 2. 資料讀取函式
# ==========================================
def load_uploaded_npy(uploaded_file):
    try:
        data = np.load(uploaded_file, allow_pickle=True)
        if data.ndim == 1:
            return data.astype(float)
        elif data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 1].astype(float)
        else:
            st.error(f"資料格式錯誤：形狀為 {data.shape}")
            return None
    except Exception as e:
        st.error(f"讀取檔案失敗: {e}")
        return None

# ==========================================
# 3. Streamlit 介面配置
# ==========================================
st.set_page_config(page_title="SST 多諧波分析", layout="wide")
st.title("📊 SST 多重諧波分析儀表板")

if not HAS_SSQ:
    st.warning("⚠️ 未安裝 ssqueezepy。請執行 `pip install ssqueezepy scipy`")
    st.stop()

with st.sidebar:
    st.header("⚙️ 參數設定")
    fps = st.number_input("取樣率 (FPS)", value=30.0, min_value=1.0)

    st.subheader("1. SST 參數")
    sst_wavelet = st.selectbox("小波基底", ['morlet', 'bump'], index=0)
    nv = st.select_slider("頻率解析度 (Voices)", options=[16, 32, 64], value=32)

    st.subheader("2. 諧波提取 (Peak Finding)")
    show_ridge = st.checkbox("顯示諧波峰值點", value=True)
    
    # 重要參數：閾值
    ridge_thresh = st.slider(
        "能量過濾閾值 (%)", 
        min_value=1, 
        max_value=50, 
        value=5, 
        step=1,
        help="只有能量強度超過「最大值 x 此百分比」的點才會被標示出來。調高此數值可過濾背景雜訊。"
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

        # 執行分析
        fig_sst = perform_multiridge_sst(
            data=signal_data, 
            fps=fps, 
            wavelet=sst_wavelet, 
            nv=nv,
            y_min=y_axis_min, 
            y_max=y_axis_max,
            show_ridge=show_ridge,
            ridge_thresh_percent=ridge_thresh/100.0 # 轉為小數
        )
        
        st.plotly_chart(fig_sst, use_container_width=True)
        
        if show_ridge:
            st.info("""
            **判讀說明：**
            圖上的**白點**代表電腦偵測到的能量峰值。
            - 如果白點太多太雜：請調高左側的「能量過濾閾值」。
            - 如果諧波沒顯示出來：請調低「能量過濾閾值」。
            這樣你就可以看到多條平行的諧波軌跡，而不是單一跳動的線。
            """)
else:
    st.info("請上傳檔案開始分析。")
