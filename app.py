import streamlit as st
import numpy as np
import plotly.graph_objects as go

# 嘗試導入 ssqueezepy
try:
    from ssqueezepy import ssq_cwt
    HAS_SSQ = True
except ImportError:
    HAS_SSQ = False

# ==========================================
# 1. 核心邏輯：脊線提取與躍遷偵測
# ==========================================
def extract_ridge_and_transitions(magnitude, freqs, threshold_percent=0.1):
    """
    演算法說明：
    1. Ridge (脊線): 在每個時間點 t，找出能量最強的頻率 f_max。
    2. Transition (躍遷): 計算 f_max 的變化率，若超過閾值則視為躍遷。
    """
    # --- A. 抓 Ridge (脊線) ---
    # axis=0 代表沿著頻率軸找最大值的索引 (因為 magnitude 是 [頻率, 時間])
    max_indices = np.argmax(magnitude, axis=0)
    
    # 將索引映射回實際的頻率值
    ridge_freqs = freqs[max_indices]
    
    # 轉成週期 (T = 1/f) 方便繪圖
    with np.errstate(divide='ignore'):
        ridge_periods = 1 / ridge_freqs

    # --- B. 定義躍遷 (Transition) ---
    # 計算相鄰時間點的頻率變化量 (微分概念)
    # diff[i] = freq[i+1] - freq[i]
    diffs = np.diff(ridge_freqs)
    
    # 計算相對變化率: |Delta_f| / f_current
    # 使用 eps 避免除以 0
    eps = 1e-8
    relative_change = np.abs(diffs) / (ridge_freqs[:-1] + eps)
    
    # 找出變化率超過設定百分比 (例如 10% = 0.1) 的時間點索引
    transition_indices = np.where(relative_change > threshold_percent)[0]
    
    return ridge_periods, transition_indices

# ==========================================
# 2. 核心分析函式 (SST + 繪圖)
# ==========================================
def perform_sst_analysis(data, fps, wavelet, nv, y_min, y_max, show_ridge, trans_thresh):
    """
    執行 SST 並繪製疊加了脊線的圖表
    """
    st.write(f"➡️ 正在計算 SST (Wavelet: {wavelet}, Voices: {nv})...")

    try:
        # 1. 計算 SST
        # Tx: SST 複數矩陣, ssq_freqs: 對應的頻率軸
        Tx, Wx, ssq_freqs, scales = ssq_cwt(data, wavelet=wavelet, fs=fps, nv=nv)
    except Exception as e:
        st.error(f"SST 計算錯誤: {e}")
        return go.Figure()

    # 2. 取能量幅度
    magnitude = np.abs(Tx)
    
    # 3. 處理座標軸 (頻率 -> 週期)
    with np.errstate(divide='ignore'): 
        periods = 1 / ssq_freqs
    
    # 過濾無效值 (直流分量無限大週期)
    valid_mask = np.isfinite(periods)
    periods = periods[valid_mask]
    magnitude = magnitude[valid_mask, :]
    valid_freqs = ssq_freqs[valid_mask] # 用於 Ridge 計算的頻率軸

    time_axis = np.arange(len(data)) / fps

    # 4. 建立 Plotly 圖表
    fig = go.Figure()
    
    # --- Layer 1: SST 熱圖 ---
    fig.add_trace(go.Heatmap(
        z=magnitude, 
        x=time_axis, 
        y=periods, 
        colorscale='Jet',
        colorbar=dict(title='能量幅度'),
        name='SST 能量譜',
        hovertemplate='時間: %{x:.2f}s<br>週期: %{y:.4f}s<br>能量: %{z:.2f}<extra></extra>'
    ))

    # --- Layer 2: 脊線 (Ridge) 與 躍遷 (Transition) ---
    ridge_info = ""
    if show_ridge:
        # 計算脊線
        ridge_periods, trans_idx = extract_ridge_and_transitions(
            magnitude, valid_freqs, threshold_percent=trans_thresh
        )
        
        # 畫白線 (脊線)
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=ridge_periods,
            mode='lines',
            line=dict(color='white', width=2),
            name='最大能量路徑 (Ridge)',
            hoverinfo='skip'
        ))
        
        # 畫紅叉 (躍遷點)
        if len(trans_idx) > 0:
            # 為了對齊，取 trans_idx 對應的時間
            t_trans = time_axis[trans_idx]
            p_trans = ridge_periods[trans_idx]
            
            fig.add_trace(go.Scatter(
                x=t_trans,
                y=p_trans,
                mode='markers',
                marker=dict(symbol='x', color='red', size=12, line=dict(width=2, color='red')),
                name='諧波躍遷點 (Transition)',
                hovertemplate='躍遷發生!<br>時間: %{x:.2f}s<br>週期: %{y:.4f}s<extra></extra>'
            ))
            ridge_info = f" | 偵測到 {len(trans_idx)} 個躍遷點 (閾值: {trans_thresh:.0%})"

    # 5.圖表美化設定
    fig.update_layout(
        title=f'SST 同步壓縮轉換 + 脊線追蹤 {ridge_info}', 
        xaxis_title='時間 (秒)', 
        yaxis_title='週期 (秒)', 
        height=700,
        yaxis_type="log", # 使用對數座標方便觀察
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # 強制設定 Y 軸範圍
    if y_min > 0 and y_max > 0:
        fig.update_yaxes(range=[np.log10(y_min), np.log10(y_max)])
    
    return fig

# ==========================================
# 3. 資料讀取函式
# ==========================================
def load_uploaded_npy(uploaded_file):
    try:
        data = np.load(uploaded_file, allow_pickle=True)
        if data.ndim == 1:
            return data.astype(float)
        elif data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 1].astype(float)
        else:
            st.error(f"資料格式錯誤：形狀為 {data.shape}，需為 (N,) 或 (N, 2)")
            return None
    except Exception as e:
        st.error(f"讀取檔案失敗: {e}")
        return None

# ==========================================
# 4. Streamlit 介面配置
# ==========================================
st.set_page_config(page_title="SST 諧波分析儀表板", layout="wide")
st.title("📊 進階諧波分析 (SST + Ridge Detection)")

# 檢查庫是否存在
if not HAS_SSQ:
    st.warning("⚠️ 系統檢測到未安裝 `ssqueezepy`，無法執行 SST。")
    st.code("pip install ssqueezepy", language="bash")
    st.stop()

with st.sidebar:
    st.header("⚙️ 參數設定")
    
    fps = st.number_input("取樣率 (FPS)", value=30.0, min_value=1.0, step=1.0)

    st.subheader("1. SST 轉換參數")
    sst_wavelet = st.selectbox("小波基底", ['morlet', 'bump'], index=0)
    nv = st.select_slider("頻率解析度 (Voices)", options=[16, 32, 64], value=32)

    st.subheader("2. 諧波脊線分析")
    show_ridge = st.checkbox("顯示 Ridge (脊線) 與 躍遷", value=True)
    
    trans_thresh = st.slider(
        "躍遷判定閾值 (變化率)", 
        min_value=0.01, 
        max_value=0.50, 
        value=0.10, 
        step=0.01,
        help="數值越小越敏感。例如 0.10 代表頻率變化超過 10% 即視為躍遷。"
    )

    st.subheader("3. 顯示範圍")
    col1, col2 = st.columns(2)
    with col1:
        y_axis_min = st.number_input("Min 週期(s)", value=0.1, format="%.2f")
    with col2:
        y_axis_max = st.number_input("Max 週期(s)", value=10.0, format="%.1f")
    
    st.divider()
    st.caption("說明：\n- **白線**：能量最強的頻率路徑。\n- **紅叉**：頻率發生突變的時間點。")

# --- 主畫面 ---
uploaded_file = st.file_uploader("上傳 .npy 數據檔案", type=["npy"])

if uploaded_file is not None:
    signal_data = load_uploaded_npy(uploaded_file)
    
    if signal_data is not None:
        # 去除直流分量
        signal_data = signal_data - np.mean(signal_data)
        
        st.success(f"檔案讀取成功 ({len(signal_data)} 點)")
        st.line_chart(signal_data, height=150)

        # 執行分析
        fig_sst = perform_sst_analysis(
            data=signal_data, 
            fps=fps, 
            wavelet=sst_wavelet, 
            nv=nv,
            y_min=y_axis_min, 
            y_max=y_axis_max,
            show_ridge=show_ridge,
            trans_thresh=trans_thresh
        )
        
        st.plotly_chart(fig_sst, use_container_width=True)
        
        if show_ridge:
            st.info("""
            **判讀指南：**
            1. **Ridge (白線)**：代表該時刻訊號的「主旋律」或「主頻率」。
            2. **Transition (紅叉)**：代表訊號特性發生了改變（例如：姿態變換、轉速改變、新的外力介入）。
            3. 如果紅叉太多，請嘗試調高「躍遷判定閾值」。
            """)

else:
    st.info("請上傳 .npy 檔案以開始分析。")
