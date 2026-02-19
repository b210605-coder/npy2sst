import streamlit as st
import numpy as np
import plotly.graph_objects as go

# 嘗試導入 ssqueezepy，如果沒有安裝則提示
try:
    from ssqueezepy import ssq_cwt
    HAS_SSQ = True
except ImportError:
    HAS_SSQ = False

# ==========================================
# 1. 核心分析函式 (SST)
# ==========================================
def perform_sst(data, fps, wavelet, nv, y_min, y_max):
    """
    執行 SST (同步壓縮轉換) 並回傳 Plotly Figure
    
    Parameters:
    - wavelet: 'morlet' 或 'bump' (SST 最常用的兩種)
    - nv: Number of Voices (每階聲音數，決定頻率解析度，通常 32 或 64)
    """
    st.write(f"➡️ 開始進行 SST 分析 (小波: {wavelet}, Voices: {nv})...")
    st.caption("提示：SST 計算量較大，請耐心等待...")

    try:
        # ssqueezepy 的 ssq_cwt 直接回傳：
        # Tx: SST 轉換後的矩陣 (Complex)
        # Wx: 原始 CWT 矩陣
        # ssq_freqs: SST 的頻率軸 (Hz)
        # scales: 使用的尺度
        Tx, Wx, ssq_freqs, scales = ssq_cwt(data, wavelet=wavelet, fs=fps, nv=nv)
    except Exception as e:
        st.error(f"SST 計算錯誤: {e}")
        return go.Figure().update_layout(title='SST 分析失敗')

    # 取絕對值獲得能量/幅度
    magnitude = np.abs(Tx)
    
    # 頻率轉週期 (T = 1/f)
    # 注意：ssq_freqs 包含了從高頻到低頻的數值
    with np.errstate(divide='ignore'): 
        periods = 1 / ssq_freqs
    
    # 處理無限大或無效值 (直流分量)
    valid_mask = np.isfinite(periods)
    periods = periods[valid_mask]
    magnitude = magnitude[valid_mask, :] # 對應的矩陣也要切片
    
    # 建立時間軸
    time_axis = np.arange(len(data)) / fps

    # 繪製 Plotly 熱圖
    # SST 的特點是線條非常銳利，我們使用 Jet 或 Turbo 配色
    fig = go.Figure(data=go.Heatmap(
        z=magnitude, 
        x=time_axis, 
        y=periods, 
        colorscale='Jet',
        colorbar=dict(title='幅度 (Magnitude)'),
        hovertemplate='時間: %{x:.2f} s<br>週期: %{y:.3f} s<br>幅度: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'同步壓縮轉換 (SST) - {wavelet}', 
        xaxis_title='時間 (秒)', 
        yaxis_title='週期 (秒)', 
        height=650,
        yaxis_type="log" # Y 軸維持對數座標，方便觀察寬範圍頻率
    )
    
    # 設定 Y 軸顯示範圍
    if y_min > 0 and y_max > 0:
        fig.update_yaxes(range=[np.log10(y_min), np.log10(y_max)])
    
    return fig

# ==========================================
# 2. 資料讀取函式 (保持不變)
# ==========================================
def load_uploaded_npy(uploaded_file):
    try:
        data = np.load(uploaded_file, allow_pickle=True)
        # 簡單判斷形狀，兼容 (N,) 或 (N, 2)
        if data.ndim == 1:
            return data.astype(float)
        elif data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 1].astype(float) # 假設第二欄是訊號
        else:
            st.error(f"資料格式不符：形狀為 {data.shape}，預期為 (N,) 或 (N, 2)")
            return None
    except Exception as e:
        st.error(f"讀取檔案失敗: {e}")
        return None

# ==========================================
# 3. Streamlit 介面配置
# ==========================================
st.set_page_config(page_title="SST 訊號分析儀表板", layout="wide")
st.title("📊 進階訊號分析儀表板 (SST 專用版)")

# 檢查依賴庫
if not HAS_SSQ:
    st.warning("⚠️ 系統檢測到未安裝 `ssqueezepy`。SST 功能無法使用。")
    st.code("pip install ssqueezepy", language="bash")
    st.stop()

with st.sidebar:
    st.header("⚙️ 參數設定")
    
    fps = st.number_input("取樣率 (FPS)", value=30.0, min_value=1.0, step=1.0)

    st.subheader("SST 參數")
    # SST 在 ssqueezepy 中主要支援 morlet 和 bump
    sst_wavelet = st.selectbox(
        "小波選擇 (Wavelet)", 
        ['morlet', 'bump'], 
        index=0,
        help="Morlet 適合一般震盪訊號；Bump 頻率定位性更好但時間解析度稍差。"
    )

    # NV (Number of Voices) 取代了 Scales 的概念
    nv = st.number_input(
        "Voices (每階層級數)", 
        value=32, 
        min_value=16, 
        max_value=64, 
        step=8,
        help="��值越高，頻率解析度越細緻，但計算越慢。通常設為 32 或 64。"
    )

# --- 顯示範圍控制 ---
    st.markdown("**圖表顯示範圍 (週期)**")
    col1, col2 = st.columns(2)
    with col1:
        y_axis_min = st.number_input("Min (秒)", value=0.1, format="%.2f")
    with col2:
        y_axis_max = st.number_input("Max (秒)", value=10.0, format="%.1f")

    st.divider()
    st.caption("ℹ️ 已移除 0-1 Test 與 GAF 模組，專注於時頻分析。")

# --- 主畫面 ---
uploaded_file = st.file_uploader("上傳 .npy 數據檔案", type=["npy"])

if uploaded_file is not None:
    signal_data = load_uploaded_npy(uploaded_file)
    
    if signal_data is not None:
        # 去除直流分量 (DC Offset)
        signal_data = signal_data - np.mean(signal_data)
        
        st.success(f"檔案讀取成功！ 數據長度: {len(signal_data)} 點")
        st.line_chart(signal_data, height=150)

        st.markdown("### 🔍 同步壓縮轉換 (Synchrosqueezing Transform)")
        st.markdown("""
        SST 是 CWT 的後處理技術，能將原本模糊的時頻能量「重新分配 (Reassign)」到瞬時頻率中心。
        **觀察重點：** 尋找圖中**清晰、銳利的亮線**，這代表訊號中真實存在的週期性成分（諧波）。
        """
        )
        
        fig_sst = perform_sst(
            data=signal_data, 
            fps=fps, 
            wavelet=sst_wavelet, 
            nv=int(nv),
            y_min=y_axis_min, 
            y_max=y_axis_max
        )
        
        st.plotly_chart(fig_sst, use_container_width=True)

else:
    st.info("請從左側上傳 .npy 檔案以開始分析。")
