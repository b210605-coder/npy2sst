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
# 核心分析函式 (SST + 諧波提取 + 躍遷偵測)
# ==========================================
def analyze_sst_and_ridges(
    data, fps, wavelet, nv, y_min, y_max, 
    ridge_thresh_percent, min_dist, 
    top_k_ridges,          # 新增: 每個時間點只保留能量最強的 K 個點 (降噪關鍵)
    jump_duration_sec      # 新增: 躍遷必須維持的秒數
):
    """
    執行 SST，提取帶有能量顏色的脊線，並偵測諧波能量躍遷 (3rd > 2nd)
    """
    st.write(f"🔄 計算 SST (Wavelet: {wavelet}, Voices: {nv})...")

    try:
        # 1. 計算 SST
        Tx, Wx, ssq_freqs, scales = ssq_cwt(data, wavelet=wavelet, fs=fps, nv=nv)
    except Exception as e:
        st.error(f"SST 計算錯誤: {e}")
        return go.Figure(), go.Figure(), []

    # 2. 處理數據
    magnitude = np.abs(Tx)
    with np.errstate(divide='ignore'): 
        periods = 1 / ssq_freqs
    time_axis = np.arange(len(data)) / fps
    
    # 3. 準備儲存脊線數據
    ridge_x = []
    ridge_y = []
    ridge_z = [] # 儲存能量值以供上色
    
    # 躍遷偵測變數
    jump_events = []
    consecutive_frames = 0
    required_frames = int(jump_duration_sec * fps)
    current_jump_start_time = None
    is_jumping = False

    # 4. 逐時掃描與特徵提取
    num_time_steps = magnitude.shape[1]
    global_max_energy = np.max(magnitude)
    abs_threshold = global_max_energy * ridge_thresh_percent

    for t_idx in range(num_time_steps):
        spectrum_slice = magnitude[:, t_idx]
        
        # --- A. 找峰值 ---
        peaks, properties = find_peaks(
            spectrum_slice, 
            height=abs_threshold, 
            distance=min_dist
        )
        
        if len(peaks) > 0:
            peak_periods = periods[peaks]
            peak_energies = properties['peak_heights']
            
            # --- B. Top-K 過濾 (讓線條更乾淨) ---
            # 將峰值按能量大小排序 (大 -> 小)
            sorted_indices = np.argsort(peak_energies)[::-1]
            # 只取前 K 個最強的點 (去除背景微弱雜訊)
            keep_indices = sorted_indices[:top_k_ridges]
            
            final_peaks = peaks[keep_indices]
            final_periods = peak_periods[keep_indices]
            final_energies = peak_energies[keep_indices]

            # 存入繪圖數據
            ridge_x.extend([time_axis[t_idx]] * len(final_peaks))
            ridge_y.extend(final_periods)
            ridge_z.extend(final_energies)

            # --- C. 諧波躍遷偵測邏輯 (3rd > 2nd) ---
            # 為了區分第幾諧波，我們需要按「週期」從大到小排序 (基頻週期最大)
            # Index 0: 基頻 (1st), Index 1: 2nd, Index 2: 3rd...
            harmonic_sort_idx = np.argsort(final_periods)[::-1]
            
            # 只有當偵測到至少 3 個明顯諧波時才進行比較
            if len(harmonic_sort_idx) >= 3:
                idx_2nd = harmonic_sort_idx[1] # 第二諧波的 index
                idx_3rd = harmonic_sort_idx[2] # 第三諧波的 index
                
                energy_2nd = final_energies[np.where(keep_indices == idx_2nd)[0][0]]
                energy_3rd = final_energies[np.where(keep_indices == idx_3rd)[0][0]]

                # 判定條件: 第三諧波能量 > 第二諧波
                if energy_3rd > energy_2nd:
                    if not is_jumping:
                        current_jump_start_time = time_axis[t_idx]
                        is_jumping = True
                    consecutive_frames += 1
                else:
                    # 中斷了，檢查之前是否滿足持續時間
                    if is_jumping and consecutive_frames >= required_frames:
                        jump_events.append(current_jump_start_time)
                    
                    # 重置
                    is_jumping = False
                    consecutive_frames = 0
            else:
                # 諧波不足，視為中斷
                if is_jumping and consecutive_frames >= required_frames:
                    jump_events.append(current_jump_start_time)
                is_jumping = False
                consecutive_frames = 0

    # 迴圈結束後，如果還在 jumping 狀態且滿足時間，也要紀錄
    if is_jumping and consecutive_frames >= required_frames:
        jump_events.append(current_jump_start_time)

    # ==========================================
    # 5. 繪製圖表 1: 原始 SST 熱圖
    # ==========================================
    fig_sst = go.Figure()
    
    # 過濾顯示範圍
    valid_mask = np.isfinite(periods)
    plot_periods = periods[valid_mask]
    plot_magnitude = magnitude[valid_mask, :]

    fig_sst.add_trace(go.Heatmap(
        z=plot_magnitude, 
        x=time_axis, 
        y=plot_periods, 
        colorscale='Jet', 
        colorbar=dict(title='Energy'),
        name='SST Spectrum'
    ))

    # 在熱圖上也標示躍遷點 (垂直線)
    for jump_t in jump_events:
        fig_sst.add_vline(x=jump_t, line_width=2, line_dash="dash", line_color="white", opacity=0.8)

    fig_sst.update_layout(
        title='1. SST 時頻能量熱圖 (含躍遷標記)',
        xaxis_title='時間 (s)', yaxis_title='週期 (s)',
        height=500, yaxis_type="log"
    )
    if y_min > 0 and y_max > 0:
        fig_sst.update_yaxes(range=[np.log10(y_min), np.log10(y_max)])

    # ==========================================
    # 6. 繪製圖表 2: 獨立脊線圖 (保留熱值顏色)
    # ==========================================
    fig_ridge = go.Figure()

    # 繪製彩色散點 (模擬連續線條)
    fig_ridge.add_trace(go.Scatter(
        x=ridge_x,
        y=ridge_y,
        mode='markers',
        marker=dict(
            size=4,           # 點稍微大一點以呈現連續感
            color=ridge_z,    # 關鍵：顏色對應能量
            colorscale='Jet', # 保持跟熱圖一樣的色階
            showscale=True,
            colorbar=dict(title='Energy')
        ),
        name='Harmonic Ridges',
        hoverinfo='x+y+text',
        text=[f"Energy: {z:.2f}" for z in ridge_z]
    ))

    # 標示躍遷點 (加上文字標註)
    for i, jump_t in enumerate(jump_events):
        fig_ridge.add_vline(x=jump_t, line_width=2, line_dash="dash", line_color="red")
        fig_ridge.add_annotation(
            x=jump_t, y=np.log10(y_max) if y_max>0 else 0,
            text=f"Jump {i+1}", showarrow=False, yshift=10, font=dict(color="red")
        )

    fig_ridge.update_layout(
        title=f'2. 諧波特徵提取 (Ridges) & 躍遷偵測 (共發現 {len(jump_events)} 次)',
        xaxis_title='時間 (s)', 
        yaxis_title='週期 (s)',
        height=500, 
        yaxis_type="log",
        plot_bgcolor='rgba(0,0,0,0.05)' # 淺灰背景凸顯線條
    )
    
    if y_min > 0 and y_max > 0:
        fig_ridge.update_yaxes(range=[np.log10(y_min), np.log10(y_max)])

    return fig_sst, fig_ridge, jump_events

# ==========================================
# 3. Streamlit 介面
# ==========================================
st.set_page_config(page_title="SST 諧波分析 Pro", layout="wide")
st.title("📊 SST 諧波分析 Pro (分離顯示 + 躍遷偵測)")

if not HAS_SSQ:
    st.error("請先安裝必要套件: pip install ssqueezepy scipy plotly")
    st.stop()

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    fps = st.number_input("取樣率 (FPS)", value=30.0, min_value=1.0)
    
    with st.expander("1. SST 基礎參數", expanded=False):
        sst_wavelet = st.selectbox("小波基底", ['morlet', 'bump'], index=0)
        nv = st.select_slider("頻率解析度 (Voices)", options=[16, 32, 64], value=32)

    st.subheader("2. 脊線提取 (去噪與連續性)")
    # [優化] 能量過濾
    ridge_thresh = st.slider("⚡ 能量過濾門檻 (%)", 1, 40, 5, help="過濾背景雜訊，數值越大越乾淨")
    # [優化] 最小間距
    min_dist = st.slider("↔️ 峰值最小間距 (Px)", 1, 50, 15, help="避免同一條線出現雙重影子")
    # [新增] Top-K
    top_k = st.slider("🔝 每個時刻只留 Top K 強點", 1, 10, 5, help="強制只抓能量最強的前幾條線，能極大程度讓圖變乾淨")

    st.subheader("3. 諧波躍遷 (Jump Detection)")
    st.caption("定義：當第 3 諧波能量 > 第 2 諧波能量")
    jump_dur = st.number_input("⏱️ 觸發需持續 (秒)", value=0.1, step=0.05, min_value=0.0)

    st.subheader("4. 顯示範圍")
    c1, c2 = st.columns(2)
    y_axis_min = c1.number_input("Min 週期(s)", value=0.1)
    y_axis_max = c2.number_input("Max 週期(s)", value=10.0)

# --- 主程式 ---
def load_uploaded_npy(uploaded_file):
    try:
        data = np.load(uploaded_file, allow_pickle=True)
        if data.ndim == 1: return data.astype(float)
        elif data.ndim == 2 and data.shape[1] >= 2: return data[:, 1].astype(float)
        return None
    except: return None

uploaded_file = st.file_uploader("上傳 .npy 數據檔案", type=["npy"])

if uploaded_file is not None:
    signal_data = load_uploaded_npy(uploaded_file)
    if signal_data is not None:
        # 去除直流分量
        signal_data = signal_data - np.mean(signal_data)
        
        st.subheader("原始訊號")
        st.line_chart(signal_data[:1000] if len(signal_data)>1000 else signal_data, height=120)

        # 執行分析
        fig1, fig2, jumps = analyze_sst_and_ridges(
            data=signal_data, 
            fps=fps, 
            wavelet=sst_wavelet, 
            nv=nv,
            y_min=y_axis_min, 
            y_max=y_axis_max,
            ridge_thresh_percent=ridge_thresh/100.0,
            min_dist=min_dist,
            top_k_ridges=top_k,
            jump_duration_sec=jump_dur
        )
        
        # 顯示結果
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        
        # 顯示躍遷資訊
        if jumps:
            st.success(f"✅ 偵測到 {len(jumps)} 次諧波躍遷 (3rd > 2nd)！")
            st.write("躍遷發生時間點 (秒):", [round(t, 3) for t in jumps])
        else:
            st.warning("在此設定下未偵測到諧波躍遷事件。")
            
        st.info("""
        **💡 調校技巧：**
        1. **想讓線條更連續？** 
           - 降低「能量過濾門檻」。
           - 減少「Top K」數量 (例如設為 3 或 4)，強迫只顯示主要諧波。
        2. **雜訊太多？** 
           - 調高「能量過濾門檻」。
           - 調大「峰值最小間距」。
        3. **躍遷抓不到？**
           - 可能是「持續時間」設太長，試著縮短。
           - 或是第 3 諧波確實沒有比第 2 諧波強。
        """)
