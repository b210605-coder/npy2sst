import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks

# Try importing ssqueezepy
try:
    from ssqueezepy import ssq_cwt
    HAS_SSQ = True
except ImportError:
    HAS_SSQ = False

# ==========================================
# Core Analysis Function (SSWT + Period Anchoring + Extrema Tracking)
# ==========================================
def analyze_sst_and_ridges(
    data, fps, wavelet, nv, y_min, y_max, 
    ridge_thresh_percent, min_dist, 
    top_k_ridges,          
    jump_duration_sec,
    jump_ratio 
):
    st.write(f"🔄 Computing SSWT (Wavelet: {wavelet}, Voices: {nv})...")

    try:
        Tx, Wx, ssq_freqs, scales = ssq_cwt(data, wavelet=wavelet, fs=fps, nv=nv)
    except Exception as e:
        st.error(f"SSWT Computation Error: {e}")
        return go.Figure(), go.Figure(), [], {}

    magnitude = np.abs(Tx)
    with np.errstate(divide='ignore'): 
        periods = 1 / ssq_freqs
    time_axis = np.arange(len(data)) / fps
    total_duration = time_axis[-1]
    
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

    # Create valid range mask
    valid_period_mask = (periods >= y_min) & (periods <= y_max)
    num_time_steps = magnitude.shape[1]
    
    valid_magnitude = np.where(valid_period_mask[:, None], magnitude, 0)
    global_max_energy = np.max(valid_magnitude)
    abs_threshold = global_max_energy * ridge_thresh_percent

    for t_idx in range(num_time_steps):
        spectrum_slice = np.copy(magnitude[:, t_idx])
        spectrum_slice[~valid_period_mask] = 0 
        
        # Find all valid peaks
        peaks, properties = find_peaks(spectrum_slice, height=abs_threshold, distance=min_dist)
        
        if len(peaks) > 0:
            peak_periods = periods[peaks]
            peak_energies = properties['peak_heights']
            
            # Filter and keep the Top K strongest points
            sorted_indices = np.argsort(peak_energies)[::-1]
            keep_indices = sorted_indices[:top_k_ridges]
            
            final_peaks = peaks[keep_indices]
            final_periods = peak_periods[keep_indices]
            final_energies = peak_energies[keep_indices]

            # ==========================================
            # Anchoring Logic: 1st Harmonic = Longest Period
            # ==========================================
            base_idx = np.argmax(final_periods) 
            T_base = final_periods[base_idx]
            E_base = final_energies[base_idx]

            # Classify based on the true fundamental period (T_base)
            for p_val, e_val in zip(final_periods, final_energies):
                ratio = T_base / p_val  
                t_val = time_axis[t_idx]

                if 0.85 <= ratio <= 1.15:     
                    h_num = 1
                elif 1.8 <= ratio <= 2.2:     
                    h_num = 2
                elif 2.8 <= ratio <= 3.2:     
                    h_num = 3
                else:
                    h_num = 0                 

                harmonic_data[h_num]['x'].append(t_val)
                harmonic_data[h_num]['y'].append(p_val)
                harmonic_data[h_num]['z'].append(e_val)

            # Jump detection logic
            mask_2nd = (periods >= T_base/2.2) & (periods <= T_base/1.8)
            mask_3rd = (periods >= T_base/3.2) & (periods <= T_base/2.8)

            E_2_real = np.max(spectrum_slice[mask_2nd]) if np.any(mask_2nd) else 0
            E_3_real = np.max(spectrum_slice[mask_3rd]) if np.any(mask_3rd) else 0

            min_required_energy = E_base * 0.05 

            if (E_3_real > E_2_real * jump_ratio) and (E_3_real > min_required_energy):
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
    # Calculate Extrema for the 1st Harmonic (Fundamental)
    # ==========================================
    stats = {
        'base_min_t': None, 'base_min_p': None,
        'base_max_t': None, 'base_max_p': None
    }
    
    if len(harmonic_data[1]['y']) > 0:
        y_array = np.array(harmonic_data[1]['y'])
        x_array = np.array(harmonic_data[1]['x'])
        
        # Find lowest point (minimum period)
        min_idx = np.argmin(y_array)
        stats['base_min_t'] = x_array[min_idx]
        stats['base_min_p'] = y_array[min_idx]
        
        # Find highest point (maximum period)
        max_idx = np.argmax(y_array)
        stats['base_max_t'] = x_array[max_idx]
        stats['base_max_p'] = y_array[max_idx]

    # ==========================================
    # White Theme Layout Base Settings
    # ==========================================
    white_layout_settings = dict(
        template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black", size=12), uirevision='constant'
    )
    y_range = [np.log10(y_min), np.log10(y_max)] if (y_min > 0 and y_max > 0) else None

    # ==========================================
    # 5. Plot 1: SSWT Heatmap
    # ==========================================
    fig_sst = go.Figure()
    plot_periods = periods[valid_period_mask]
    plot_magnitude = magnitude[valid_period_mask, :]

    fig_sst.add_trace(go.Heatmap(z=plot_magnitude, x=time_axis, y=plot_periods, coloraxis="coloraxis", name='SSWT Spectrum'))

    # Only mark the FIRST jump
    if len(jump_events) > 0:
        first_jump = jump_events[0]
        fig_sst.add_vline(x=first_jump, line_width=2, line_dash="dash", line_color="white", opacity=0.8)

    fig_sst.update_layout(
        title=dict(text='SSWT Energy Heatmap', font=dict(color="black", size=18)), # 數字移除了
        height=500,
        coloraxis=dict(colorscale='Jet', colorbar=dict(title=dict(text='Energy', font=dict(color="black")), tickfont=dict(color="black"))),
        **white_layout_settings
    )
    fig_sst.update_xaxes(title_text='Time (s)', title_font=dict(color="black", size=14), showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black', linecolor='black', ticks='outside', tickfont=dict(color="black"), range=[0, total_duration])
    fig_sst.update_yaxes(title_text='Period (s)', title_font=dict(color="black", size=14), showgrid=True, gridcolor='lightgray', zeroline=False, linecolor='black', ticks='outside', tickfont=dict(color="black"), type="log", range=y_range)

    # ==========================================
    # 6. Plot 2: SSWT Ridge Extraction
    # ==========================================
    fig_ridge = go.Figure()
    labels = {1: "1st Harmonic (Fundamental)", 2: "2nd Harmonic", 3: "3rd Harmonic", 0: "Others"}
    markers = {1: "circle", 2: "diamond", 3: "cross", 0: "x"} 
    
    all_z = []
    for k in harmonic_data: all_z.extend(harmonic_data[k]['z'])
    cmin, cmax = (min(all_z), max(all_z)) if all_z else (0, 1)

    for k in [1, 2, 3, 0]:
        d = harmonic_data[k]
        if len(d['x']) > 0:
            fig_ridge.add_trace(go.Scatter(
                x=d['x'], y=d['y'], mode='markers', name=labels[k],
                marker=dict(symbol=markers.get(k, "circle"), size=6 if k==1 else 5, color=d['z'], coloraxis="coloraxis"),
                hovertemplate=f"<b>{labels[k]}</b><br>Time: %{{x:.2f}}s<br>Period: %{{y:.4f}}s<br>Energy: %{{marker.color:.2f}}<extra></extra>"
            ))

    # ★ Clean & professional marker for the Min Period point
    if stats['base_min_t'] is not None:
        fig_ridge.add_trace(go.Scatter(
            x=[stats['base_min_t']], y=[stats['base_min_p']],
            mode='markers', name='Min Period Point',
            marker=dict(symbol='circle', size=14, color='crimson', line=dict(width=2, color='black')),
            hovertemplate="<b>Min Fundamental Period</b><br>Time: %{x:.2f}s<br>Period: %{y:.4f}s<extra></extra>"
        ))
        
        fig_ridge.add_annotation(
            x=stats['base_min_t'], y=np.log10(stats['base_min_p']),
            text="Min Period", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="crimson",
            ax=0, ay=35, font=dict(color="crimson", size=13, weight="bold")
        )

    # Only mark the FIRST jump on the ridge plot
    if len(jump_events) > 0:
        first_jump = jump_events[0]
        fig_ridge.add_vline(x=first_jump, line_width=2, line_dash="dash", line_color="red")
        fig_ridge.add_annotation(
            x=first_jump, y=np.log10(y_max) if y_max > 0 else 0, 
            text="First Jump", showarrow=False, yshift=10, 
            font=dict(color="red", size=12)
        )

    fig_ridge.update_layout(
        title=dict(text='SSWT Ridge Extraction', font=dict(color="black", size=18)), # 數字移除了
        height=500, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.8)", font=dict(color="black")),
        coloraxis=dict(colorscale='Jet', cmin=cmin, cmax=cmax, colorbar=dict(title=dict(text='Energy', font=dict(color="black")), tickfont=dict(color="black"))),
        **white_layout_settings
    )
    
    fig_ridge.update_xaxes(title_text='Time (s)', title_font=dict(color="black", size=14), showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black', linecolor='black', ticks='outside', tickfont=dict(color="black"), range=[0, total_duration], autorange=False)
    fig_ridge.update_yaxes(title_text='Period (s)', title_font=dict(color="black", size=14), showgrid=True, gridcolor='lightgray', zeroline=False, linecolor='black', ticks='outside', tickfont=dict(color="black"), type="log", range=y_range, autorange=False)

    return fig_sst, fig_ridge, jump_events, stats

# ==========================================
# 3. Streamlit Interface
# ==========================================
st.set_page_config(page_title="SSWT Ridge Extraction", layout="wide")
st.title("📊 SSWT Ridge Extraction")

if not HAS_SSQ:
    st.error("Please install required packages first: pip install ssqueezepy scipy plotly")
    st.stop()

# --- Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ Parameter Settings")
    fps = st.number_input("Sampling Rate (FPS)", value=30.0, min_value=1.0)
    
    with st.expander("1. Basic SSWT Parameters", expanded=False):
        sst_wavelet = st.selectbox("Wavelet Basis", ['morlet', 'bump'], index=0)
        nv = st.select_slider("Frequency Resolution (Voices)", options=[16, 32, 64], value=32)

    st.subheader("2. Display Range")
    st.caption("The algorithm limits fundamental frequency search to this range.")
    c1, c2 = st.columns(2)
    y_axis_min = c1.number_input("Min Period (s)", value=0.1)
    y_axis_max = c2.number_input("Max Period (s)", value=10.0)

    st.subheader("3. Ridge Extraction")
    ridge_thresh = st.slider("⚡ Energy Filter Threshold (%)", 1, 40, 5)
    min_dist = st.slider("↔️ Min Peak Distance (px)", 1, 50, 15)
    top_k = st.slider("🔝 Keep Top K Peaks", 1, 10, 5)

    st.subheader("4. Harmonic Jump Detection")
    jump_dur = st.number_input("⏱️ Trigger Duration (s)", value=0.1, step=0.05, min_value=0.0)
    jump_multiplier = st.slider("🚀 Jump Energy Threshold (E3 > E2 multiplier)", 1.0, 3.0, 1.0, 0.1)

# --- Main Program ---
def load_uploaded_npy(uploaded_file):
    try:
        data = np.load(uploaded_file, allow_pickle=True)
        if data.ndim == 1: return data.astype(float)
        elif data.ndim == 2 and data.shape[1] >= 2: return data[:, 1].astype(float)
        return None
    except: return None

uploaded_file = st.file_uploader("Upload .npy Data File", type=["npy"])

if uploaded_file is not None:
    signal_data = load_uploaded_npy(uploaded_file)
    if signal_data is not None:
        signal_data = signal_data - np.mean(signal_data)
        
        # Plot full original signal
        time_axis_orig = np.arange(len(signal_data)) / fps 
        fig_orig = go.Figure()
        fig_orig.add_trace(go.Scatter(x=time_axis_orig, y=signal_data, mode='lines', name='Original Signal', line=dict(color='royalblue', width=1)))
        fig_orig.update_layout(title=dict(text='Original Signal (DC offset removed)', font=dict(color="black", size=16)), xaxis_title='Time (s)', yaxis_title='Amplitude', height=250, margin=dict(l=0, r=0, t=40, b=0), template="plotly_white", plot_bgcolor="white", paper_bgcolor="white")
        fig_orig.update_xaxes(title_font=dict(color="black", size=12), tickfont=dict(color="black"), showgrid=True, gridcolor='lightgray', linecolor='black')
        fig_orig.update_yaxes(title_font=dict(color="black", size=12), tickfont=dict(color="black"), showgrid=True, gridcolor='lightgray', linecolor='black')
        st.plotly_chart(fig_orig, use_container_width=True, theme=None)

        # Get analysis results
        fig1, fig2, jumps, stats = analyze_sst_and_ridges(
            data=signal_data, fps=fps, wavelet=sst_wavelet, nv=nv,
            y_min=y_axis_min, y_max=y_axis_max, ridge_thresh_percent=ridge_thresh/100.0,
            min_dist=min_dist, top_k_ridges=top_k, jump_duration_sec=jump_dur,
            jump_ratio=jump_multiplier 
        )
        
        st.plotly_chart(fig1, use_container_width=True, theme=None)
        st.plotly_chart(fig2, use_container_width=True, theme=None)
        
        # Display extreme values and jump info
        st.markdown("### 📊 Analysis Summary")
        c1, c2 = st.columns(2)
        
        with c1:
            if stats['base_min_t'] is not None:
                st.info(
                    f"🔴 **Lowest Fundamental Period**\n\n"
                    f"⏱️ Time: **{stats['base_min_t']:.2f} s**\n\n"
                    f"📉 Min Period: **{stats['base_min_p']:.4f} s** (~ {1/stats['base_min_p']:.2f} Hz)"
                )
            else:
                st.info("No fundamental frequency data detected yet.")
                
        with c2:
            if stats['base_max_t'] is not None:
                st.success(
                    f"📈 **Highest Fundamental Period**\n\n"
                    f"⏱️ Time: **{stats['base_max_t']:.2f} s**\n\n"
                    f"📈 Max Period: **{stats['base_max_p']:.4f} s** (~ {1/stats['base_max_p']:.2f} Hz)"
                )

        if jumps:
            st.warning(f"🚀 **First Harmonic Jump Detected (3rd > 2nd)** at **{jumps[0]:.2f} s**\n\n" + 
                       f"Total jumps detected in sequence: {len(jumps)}")
        else:
            st.markdown("No qualifying harmonic jumps detected.")
