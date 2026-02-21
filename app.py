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
# 核心分析函式 (SST + 諧波分層標記 + 躍遷偵測)
# ==========================================
def analyze_sst_and_ridges(
    data, fps, wavelet, nv, y_min, y_max, 
    ridge_thresh_percent, min_dist, 
    top_k_ridges,          
    jump_duration_sec      
):
    """
    執行 SST，提取脊線並按諧波順序分類 (1st, 2nd, 3rd, Others)
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
    total_duration = time_axis[-1]
    
    # 3. 準備儲存分層數據
    harmonic_data = {
        1: {'x': [], 'y': [], 'z': []},
        2: {'x': [], 'y': [], 'z': []},
        3: {'x': [], 'y': [], 'z': []},
        0: {'x': [], 'y': [], 'z': []} 
    }
    
    jump_events = []
    consecutive_frames = 0
    required_frames = int(jump_duration_sec * fps)
    current_jump_start_time = None
    is_jumping = False

    # ★ 關鍵修正：建立有效範圍遮罩 ★
    # 只允許在使用者設定的 y_min ~ y_max 範圍內的訊號參與峰值排名
    valid_period_mask = (periods >= y_min) & (periods <= y_max)

    # 4. 逐時掃描與特徵提取
    num_time_steps = magnitude.shape[1]
    
    # 計算全域最大值時，也只考慮有效範圍內的值，讓門檻更準確
    valid_magnitude = np.where(valid_period_mask[:, None], magnitude, 0)
    global_max_energy = np.max(valid_magnitude)
    abs_threshold = global_max_energy * ridge_thresh_percent

    for t_idx in range(num_time_steps):
        # 取出當下時間點的頻譜，並將不在顯示範圍內的能量直接歸零
        spectrum_slice = np.copy(magnitude[:, t_idx])
        spectrum_slice[~valid_period_mask] = 0 
        
        # 尋找峰值
        peaks, properties = find_peaks(spectrum_slice, height=abs_threshold, distance=min_dist)
        
        if len(peaks) > 0:
            peak_periods = periods[peaks]
            peak_energies = properties['peak_heights']
            
            # Top-K 過濾
            sorted_indices = np.argsort(peak_energies)[::-1]
            keep_indices = sorted_indices[:top_k_ridges]
            
            final_peaks = peaks[keep_indices]
            final_periods = peak_periods[keep_indices]
            final_energies = peak_energies[keep_indices]

            # 諧波分類：按照週期從大到小排序
            local_sort_idx = np.argsort(final_periods)[::-1]
            
            for rank, idx in enumerate(local_sort_idx):
                h_num = rank + 1
                p_val = final_periods[idx]
                e_val = final_energies[idx]
                t_val = time_axis[t_idx]

                if h_num <= 3:
                    harmonic_data[h_num]['x'].append(t_val)
                    harmonic_data[h_num]['y'].append(p_val)
                    harmonic_data[h_num]['z'].append(e_val)
                else:
                    harmonic_data[0]['x'].append(t_val)
                    harmonic_data[0]['y'].append(p_val)
                    harmonic_data[0]['z'].append(e_val)

            # 躍遷偵測 (3rd > 2nd)
            if len(local_sort_idx) >= 3:
                idx_2nd = local_sort_idx[1]
                idx_3rd = local_sort_idx[2]
                
                energy_2nd = final_energies[idx_2nd]
                energy_3rd = final_energies[idx_3rd]

                if energy_3rd > energy_2nd:
                    if not is_jumping:
                        current_jump_start_time = time_axis[t_idx]
                        is_jumping = True
                    consecutive_frames += 1
                else:
                    if is_jumping and consecutive_frames >= required_frames:
                        jump_events.append(current_jump_start_time)
                    is_jumping = False
                    consecutive_frames = 0
            else:
                if is_jumping and consecutive_frames >= required_frames:
                    jump_events.append(current_jump_start_time)
                is_jumping = False
                consecutive_frames = 0

    if is_jumping and consecutive_frames >= required_frames:
        jump_events.append(current_jump_start_time)

    # ==========================================
    # 全白主題 Layout 基礎設定
    # ==========================================
    white_layout_settings = dict(
        template="plotly_white", 
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=12), 
        uirevision='constant'
    )
    
    y_range = [np.log10(y_min), np.log10(y_max)] if (y_min > 0 and y_max > 0) else None

    # ==========================================
    # 5. 繪製圖表 1: SST 熱圖
    # ==========================================
    fig_sst = go.Figure()
    
    # 熱圖顯示時也套用過濾 (選用)
    plot_periods = periods[valid_period_mask]
    plot_magnitude = magnitude[valid_period_mask, :]

    fig_sst.add_trace(go.Heatmap(
        z=plot_magnitude, x=time_axis, y=plot_periods, 
        coloraxis="coloraxis", 
        name='SST Spectrum'
    ))

    for jump_t in jump_events:
        fig_sst.add_vline(x=jump_t, line_width=2, line_dash="dash", line_color="white", opacity=0.8)

    fig_sst.update_layout(
        title=dict(text='1. SST 時頻能量熱圖', font=dict(color="black", size=18)),
        height=500,
        coloraxis=dict(
            colorscale='Jet',
            colorbar=dict(
                title=dict(text='Energy', font=dict(color="black")),
                tickfont=dict(color="black")
            )
        ),
        **white_layout_settings
    )
    
    fig_sst.update_xaxes(
        title_text='時間 (s)', title_font=dict(color="black", size=14),
        showgrid=True, gridcolor='lightgray',
        zeroline=True, zerolinecolor='black', linecolor='black', 
        ticks='outside', tickfont=dict(color="black"),
        range=[0, total_duration]
    )
    fig_sst.update_yaxes(
        title_text='週期 (s)', title_font=dict(color="black", size=14),
        showgrid=True, gridcolor='lightgray',
        zeroline=False, linecolor='black',
        ticks='outside', tickfont=dict(color="black"),
        type="log", range=y_range
    )

    # ==========================================
    # 6. 繪製圖表 2: 分層諧波脊線圖
    # ==========================================
    fig_ridge = go.Figure()

    labels = {1: "1st Harmonic (基頻)", 2: "2nd Harmonic", 3: "3rd Harmonic", 0: "Others"}
    markers = {1: "circle", 2: "diamond", 3: "cross", 0: "x"} 
    
    all_z = []
    for k in harmonic_data:
        all_z.extend(harmonic_data[k]['z'])
    cmin, cmax = (min(all_z), max(all_z)) if all_z else (0, 1)

    for k in [1, 2, 3, 0]:
        d = harmonic_data[k]
        if len(d['x']) > 0:
            fig_ridge.add_trace(go.Scatter(
                x=d['x'],
                y=d['y'],
                mode='markers',
                name=labels[k],
                marker=dict(
                    symbol=markers.get(k, "circle"),
                    size=6 if k==1 else 5, 
                    color=d['z'],
                    coloraxis="coloraxis"
                ),
                hovertemplate=f"<b>{labels[k]}</b><br>Time: %{{x:.2f}}s<br>Period: %{{y:.4f}}s<br>Energy: %{{marker.color:.2f}}<extra></extra>"
            ))

    for i, jump_t in enumerate(jump_events):
        fig_ridge.add_vline(x=jump_t, line_width=2, line_dash="dash", line_color="red")
        fig_ridge.add_annotation(
            x=jump_t, y=np.log10(y_max) if y_max>0 else 0,
            text=f"Jump {i+1}", showarrow=False, yshift=10, font=dict(color="red", size=12)
        )

    fig_ridge.update_layout(
        title=dict(text='2. 諧波分類標記 (點擊圖例可開關，畫面不跳動)', font=dict(color="black", size=18)),
        height=500, 
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.8)", font=dict(color="black")
        ),
        coloraxis=dict(
            colorscale='Jet',
            cmin=cmin, cmax=cmax,
            colorbar=dict(
                title=dict(text='Energy', font=dict(color="black")),
                tickfont=dict(color="black")
            )
        ),
        **white_layout_settings
    )
    
    fig_ridge.update_xaxes(
        title_text='時間 (s)', title_font=dict(color="black", size=14),
        showgrid=True, gridcolor='lightgray',
        zeroline=True, zerolinecolor='black', linecolor='black', 
        ticks='outside', tickfont=dict(color="black"),
        range=[0, total_duration], autorange=False
    )
    fig_ridge.update_yaxes(
        title_text='週期 (s)', title_font=dict(color="black", size=14),
        showgrid=True, gridcolor='lightgray',
        zeroline=False, linecolor='black',
        ticks='outside', tickfont=dict(color="black"),
        type="log", range=y_range, autorange=False
    )

    return fig_sst, fig_ridge, jump_events

# ==========================================
# 3. Streamlit 介面
# ==========================================
st.set_page_config(page_title="SST 諧波分析 Pro", layout="wide")
st.title("📊 SST 諧波分析 Pro (白底 + 鎖定視角)")

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

    st.subheader("2. 顯示範圍 (★極為重要)")
    st.caption("程式只會抓取此範圍內的波峰！")
    c1, c2 = st.columns(2)
    y_axis_min = c1.number_input("Min 週期(s)", value=0.1)
    y_axis_max = c2.number_input("Max 週期(s)", value=10.0)

    st.subheader("3. 脊線提取 (去噪與連續性)")
    ridge_thresh = st.slider("⚡ 能量過濾門檻 (%)", 1, 40, 5)
    min_dist = st.slider("↔️ 峰值最小間距 (Px)", 1, 50, 15)
    top_k = st.slider("🔝 每個時刻只留 Top K 強點", 1, 10, 5)

    st.subheader("4. 諧波躍遷 (Jump Detection)")
    jump_dur = st.number_input("⏱️ 觸發需持續 (秒)", value=0.1, step=0.05, min_value=0.0)

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
        signal_data = signal_data - np.mean(signal_data)
        
        st.subheader("原始訊號")
        st.line_chart(signal_data[:1000] if len(signal_data)>1000 else signal_data, height=120)

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
        
        st.plotly_chart(fig1, use_container_width=True, theme=None)
        st.plotly_chart(fig2, use_container_width=True, theme=None)
        
        if jumps:
            st.success(f"✅ 偵測到 {len(jumps)} 次諧波躍遷 (3rd > 2nd)！")
            st.write("躍遷發生時間點 (秒):", [round(t, 3) for t in jumps])
        else:
            st.warning("在此設定下未偵測到諧波躍遷事件。")
