import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

# Try importing ssqueezepy
try:
    from ssqueezepy import ssq_cwt
    HAS_SSQ = True
except ImportError:
    HAS_SSQ = False


# ==========================================
# Core Analysis Function (SSWT + CWT Panel + Period Anchoring + Extrema Tracking)
# ==========================================
def analyze_sst_and_ridges(
    data, fps, wavelet, nv, y_min, y_max,
    ridge_thresh_percent, min_dist,
    top_k_ridges,
    transition_duration_sec,
    transition_ratio
):
    st.write(f"🔄 Computing CWT + SSWT (Wavelet: {wavelet}, Voices: {nv})...")

    try:
        # Tx: synchrosqueezed CWT (SSWT/SSQ-CWT output)
        # Wx: regular CWT
        Tx, Wx, ssq_freqs, scales = ssq_cwt(data, wavelet=wavelet, fs=fps, nv=nv)
    except Exception as e:
        st.error(f"SSWT Computation Error: {e}")
        return go.Figure(), go.Figure(), [], {}, go.Figure()

    magnitude_sswt = np.abs(Tx)
    magnitude_cwt = np.abs(Wx)

    with np.errstate(divide='ignore', invalid='ignore'):
        periods = 1 / ssq_freqs

    time_axis = np.arange(len(data)) / fps
    total_duration = float(time_axis[-1]) if len(time_axis) else 0.0

    # ---- Ridge storage (for optional ridge plot) ----
    harmonic_data = {
        1: {'x': [], 'y': [], 'z': []},
        2: {'x': [], 'y': [], 'z': []},
        3: {'x': [], 'y': [], 'z': []},
        0: {'x': [], 'y': [], 'z': []}
    }

    # ---- Transition detection bookkeeping ----
    transition_events = []
    consecutive_frames = 0
    required_frames = int(max(0.0, transition_duration_sec) * fps)
    current_transition_start_time = None
    in_transition = False

    # ---- Valid period mask ----
    valid_period_mask = (periods >= y_min) & (periods <= y_max)
    num_time_steps = magnitude_sswt.shape[1]

    # Use SSWT valid-energy max for threshold baseline
    valid_magnitude = np.where(valid_period_mask[:, None], magnitude_sswt, 0)
    global_max_energy = float(np.max(valid_magnitude)) if valid_magnitude.size else 0.0
    abs_threshold = global_max_energy * float(ridge_thresh_percent)

    # ---- Per-time ridge peaks on SSWT magnitude ----
    for t_idx in range(num_time_steps):
        spectrum_slice = np.array(magnitude_sswt[:, t_idx], copy=True)
        spectrum_slice[~valid_period_mask] = 0

        peaks, properties = find_peaks(
            spectrum_slice,
            height=abs_threshold,
            distance=int(min_dist)
        )

        if len(peaks) > 0:
            peak_periods = periods[peaks]
            peak_energies = properties.get('peak_heights', np.array([]))

            # Keep top-K by energy
            sorted_indices = np.argsort(peak_energies)[::-1]
            keep_indices = sorted_indices[:int(top_k_ridges)]

            final_peaks = peaks[keep_indices]
            final_periods = peak_periods[keep_indices]
            final_energies = peak_energies[keep_indices]

            # ---- Anchoring: fundamental = longest period among kept peaks ----
            base_idx = int(np.argmax(final_periods))
            T_base = float(final_periods[base_idx])
            E_base = float(final_energies[base_idx])

            t_val = float(time_axis[t_idx])

            # Classify peaks into harmonics by period ratio
            for p_val, e_val in zip(final_periods, final_energies):
                p_val = float(p_val)
                e_val = float(e_val)

                if p_val <= 0 or not np.isfinite(p_val):
                    h_num = 0
                else:
                    ratio = T_base / p_val  # ~1,2,3 for 1st/2nd/3rd harmonic
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

            # ---- Transition detection: E3 > E2 * ratio, plus minimum energy guard ----
            mask_2nd = (periods >= T_base / 2.2) & (periods <= T_base / 1.8)
            mask_3rd = (periods >= T_base / 3.2) & (periods <= T_base / 2.8)

            E_2_real = float(np.max(spectrum_slice[mask_2nd])) if np.any(mask_2nd) else 0.0
            E_3_real = float(np.max(spectrum_slice[mask_3rd])) if np.any(mask_3rd) else 0.0

            min_required_energy = E_base * 0.05  # prevent tiny noise

            if (E_3_real > E_2_real * float(transition_ratio)) and (E_3_real > min_required_energy):
                if not in_transition:
                    current_transition_start_time = t_val
                    in_transition = True
                consecutive_frames += 1
            else:
                if in_transition and consecutive_frames >= required_frames and current_transition_start_time is not None:
                    transition_events.append(current_transition_start_time)
                in_transition = False
                consecutive_frames = 0
        else:
            if in_transition and consecutive_frames >= required_frames and current_transition_start_time is not None:
                transition_events.append(current_transition_start_time)
            in_transition = False
            consecutive_frames = 0

    if in_transition and consecutive_frames >= required_frames and current_transition_start_time is not None:
        transition_events.append(current_transition_start_time)

    # ---- Fundamental extrema (global min/max period among 1st harmonic points) ----
    stats = {
        'base_min_t': None, 'base_min_p': None,
        'base_max_t': None, 'base_max_p': None
    }

    if len(harmonic_data[1]['y']) > 0:
        y_array = np.array(harmonic_data[1]['y'], dtype=float)
        x_array = np.array(harmonic_data[1]['x'], dtype=float)

        valid = np.isfinite(y_array) & (y_array > 0) & np.isfinite(x_array)
        y_array = y_array[valid]
        x_array = x_array[valid]

        if y_array.size:
            min_idx = int(np.argmin(y_array))
            max_idx = int(np.argmax(y_array))
            stats['base_min_t'] = float(x_array[min_idx])
            stats['base_min_p'] = float(y_array[min_idx])
            stats['base_max_t'] = float(x_array[max_idx])
            stats['base_max_p'] = float(y_array[max_idx])

    # ==========================================
    # Journal-style layout settings
    # ==========================================
    journal_layout = dict(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", color="black", size=14),
        margin=dict(t=70, b=60, l=70, r=30),
        uirevision='constant'
    )

    axis_settings = dict(
        showline=True, linecolor='black', linewidth=1.5,
        mirror=True, ticks='inside', tickcolor='black', tickwidth=1.5, ticklen=6,
        showgrid=False, zeroline=False
    )

    y_range = [np.log10(y_min), np.log10(y_max)] if (y_min > 0 and y_max > 0) else None

    # ==========================================
    # Two-panel figure: (a) CWT, (b) SSWT
    # ==========================================
    fig_twocol = make_subplots(
        rows=1, cols=2,
        subplot_titles=("(a) CWT Energy Scalogram", "(b) SSWT Energy Heatmap"),
        horizontal_spacing=0.08
    )

    plot_periods = periods[valid_period_mask]
    plot_cwt = magnitude_cwt[valid_period_mask, :]
    plot_sswt = magnitude_sswt[valid_period_mask, :]

    # Share a common color scale for side-by-side comparability
    combined = np.concatenate([plot_cwt.ravel(), plot_sswt.ravel()]) if plot_cwt.size and plot_sswt.size else np.array([0, 1])
    cmin = float(np.nanmin(combined))
    cmax = float(np.nanmax(combined)) if float(np.nanmax(combined)) > cmin else (cmin + 1.0)

    fig_twocol.add_trace(
        go.Heatmap(
            z=plot_cwt, x=time_axis, y=plot_periods,
            coloraxis="coloraxis", name="CWT"
        ),
        row=1, col=1
    )

    fig_twocol.add_trace(
        go.Heatmap(
            z=plot_sswt, x=time_axis, y=plot_periods,
            coloraxis="coloraxis", name="SSWT"
        ),
        row=1, col=2
    )

    # Transition line on both panels (use paper coords so it spans each subplot cleanly)
    if len(transition_events) > 0:
        t0 = float(transition_events[0])

        # vline on col 1 and col 2
        fig_twocol.add_vline(x=t0, line_width=2, line_dash="dash", line_color="black", row=1, col=1)
        fig_twocol.add_vline(x=t0, line_width=2, line_dash="dash", line_color="black", row=1, col=2)

        # annotate "Transition" near top in SSWT panel
        if y_max > 0:
            fig_twocol.add_annotation(
                x=t0, y=y_max, xref="x2", yref="y2",
                text="Transition", showarrow=False,
                yshift=18, font=dict(family="Arial", color="black", size=12)
            )

    # Mark "turn" on SSWT panel = lowest fundamental period point
    if stats['base_min_t'] is not None and stats['base_min_p'] is not None:
        fig_twocol.add_trace(
            go.Scatter(
                x=[stats['base_min_t']], y=[stats['base_min_p']],
                mode="markers+text",
                text=["turn"],
                textposition="top center",
                name="turn",
                marker=dict(symbol="circle-open", size=12, line=dict(width=2.5, color="crimson")),
                hovertemplate="<b>turn</b><br>Time: %{x:.2f}s<br>Period: %{y:.4f}s<extra></extra>"
            ),
            row=1, col=2
        )

    fig_twocol.update_layout(
        height=520,
        coloraxis=dict(
            colorscale="Viridis",
            cmin=cmin, cmax=cmax,
            colorbar=dict(
                title=dict(text="Energy", font=dict(family="Arial", size=14, color="black")),
                tickfont=dict(family="Arial", size=12, color="black"),
                outlinewidth=1, outlinecolor="black",
                thickness=18
            )
        ),
        showlegend=False,
        **journal_layout
    )

    # Axes formatting for both panels
    fig_twocol.update_xaxes(title_text="Time (s)", range=[0, total_duration], **axis_settings, row=1, col=1)
    fig_twocol.update_xaxes(title_text="Time (s)", range=[0, total_duration], **axis_settings, row=1, col=2)

    fig_twocol.update_yaxes(title_text="Period (s)", type="log", range=y_range, **axis_settings, row=1, col=1)
    fig_twocol.update_yaxes(title_text="Period (s)", type="log", range=y_range, **axis_settings, row=1, col=2)

    # ==========================================
    # Optional: ridge extraction scatter plot (still useful for debugging / SI)
    # ==========================================
    fig_ridge = go.Figure()
    labels = {1: "1st Harmonic", 2: "2nd Harmonic", 3: "3rd Harmonic", 0: "Others"}
    markers = {1: "circle", 2: "diamond", 3: "cross", 0: "x"}

    all_z = []
    for k in harmonic_data:
        all_z.extend(harmonic_data[k]['z'])
    rzmin, rzmax = (min(all_z), max(all_z)) if all_z else (0, 1)

    for k in [1, 2, 3, 0]:
        d = harmonic_data[k]
        if len(d['x']) > 0:
            fig_ridge.add_trace(go.Scatter(
                x=d['x'], y=d['y'], mode='markers', name=labels[k],
                marker=dict(
                    symbol=markers.get(k, "circle"),
                    size=6 if k == 1 else 5,
                    color=d['z'],
                    coloraxis="coloraxis",
                    line=dict(width=0.5, color='black') if k == 1 else None
                ),
                hovertemplate=f"<b>{labels[k]}</b><br>Time: %{{x:.2f}}s<br>Period: %{{y:.4f}}s<br>Energy: %{{marker.color:.2f}}<extra></extra>"
            ))

    if stats['base_min_t'] is not None and stats['base_min_p'] is not None:
        fig_ridge.add_trace(go.Scatter(
            x=[stats['base_min_t']], y=[stats['base_min_p']],
            mode='markers+text', name='turn',
            text=["turn"], textposition="top center",
            marker=dict(symbol='circle-open', size=12, line=dict(width=2.5, color='crimson'))
        ))

    if len(transition_events) > 0:
        t0 = float(transition_events[0])
        fig_ridge.add_vline(x=t0, line_width=1.5, line_dash="dash", line_color="crimson")
        fig_ridge.add_annotation(
            x=t0, y=y_max if y_max > 0 else 1,
            text="Transition", showarrow=False, yshift=15,
            font=dict(family="Arial", color="crimson", size=12)
        )

    fig_ridge.update_layout(
        title=dict(text='(c) SSWT Ridge Extraction (debug / SI)', font=dict(family="Arial", size=18, color="black"), x=0, xanchor="left"),
        height=480,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="white", bordercolor="black", borderwidth=1,
            font=dict(family="Arial", size=12, color="black")
        ),
        coloraxis=dict(
            colorscale='Viridis', cmin=rzmin, cmax=rzmax,
            colorbar=dict(
                title=dict(text='Energy', font=dict(family="Arial", size=14, color="black")),
                tickfont=dict(family="Arial", size=12, color="black"),
                outlinewidth=1, outlinecolor="black",
                thickness=18
            )
        ),
        **journal_layout
    )
    fig_ridge.update_xaxes(title_text='Time (s)', range=[0, total_duration], **axis_settings)
    fig_ridge.update_yaxes(title_text='Period (s)', type="log", range=y_range, **axis_settings)

    return fig_twocol, transition_events, stats, fig_ridge


# ==========================================
# Streamlit Interface
# ==========================================
st.set_page_config(page_title="CWT + SSWT (Two-Panel Figure)", layout="wide")
st.title("📊 Two-Panel Figure for Journal (a) CWT + (b) SSWT")

if not HAS_SSQ:
    st.error("Please install required packages first: pip install ssqueezepy scipy plotly streamlit numpy")
    st.stop()


# --- Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ Parameter Settings")
    fps = st.number_input("Sampling Rate (FPS)", value=30.0, min_value=1.0)

    with st.expander("1. SSWT/CWT Parameters", expanded=False):
        sst_wavelet = st.selectbox("Wavelet Basis", ['morlet', 'bump'], index=0)
        nv = st.select_slider("Frequency Resolution (Voices)", options=[16, 32, 64], value=32)

    st.subheader("2. Display Range (Period)")
    st.caption("Algorithm limits search to this period range (seconds).")
    c1, c2 = st.columns(2)
    y_axis_min = c1.number_input("Min Period (s)", value=0.1, min_value=1e-6, format="%.6f")
    y_axis_max = c2.number_input("Max Period (s)", value=10.0, min_value=1e-6, format="%.6f")

    st.subheader("3. Ridge Extraction (for turn/transition)")
    ridge_thresh = st.slider("⚡ Energy Filter Threshold (%)", 1, 40, 5)
    min_dist = st.slider("↔️ Min Peak Distance (px)", 1, 50, 15)
    top_k = st.slider("🔝 Keep Top K Peaks", 1, 10, 5)

    st.subheader("4. Transition Detection")
    transition_dur = st.number_input("⏱️ Trigger Duration (s)", value=0.1, step=0.05, min_value=0.0)
    transition_multiplier = st.slider("🚀 Transition Threshold (E3 > E2 multiplier)", 1.0, 3.0, 1.0, 0.1)


def load_uploaded_npy(uploaded_file):
    try:
        data = np.load(uploaded_file, allow_pickle=True)
        if data.ndim == 1:
            return data.astype(float)
        elif data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 1].astype(float)
        return None
    except Exception:
        return None


uploaded_file = st.file_uploader("Upload .npy Data File", type=["npy"])

if uploaded_file is not None:
    signal_data = load_uploaded_npy(uploaded_file)

    if signal_data is None:
        st.error("Could not parse the .npy file. Expect 1D array or 2D array with >=2 columns.")
        st.stop()

    # Demean
    signal_data = signal_data - np.mean(signal_data)

    # Plot original signal (optional, but useful)
    time_axis_orig = np.arange(len(signal_data)) / fps
    fig_orig = go.Figure()
    fig_orig.add_trace(go.Scatter(
        x=time_axis_orig, y=signal_data, mode='lines',
        name='Original Signal', line=dict(color='#1f77b4', width=1.5)
    ))
    journal_axis_settings = dict(
        showline=True, linecolor='black', linewidth=1.5,
        mirror=True, ticks='inside', tickcolor='black', tickwidth=1.5, ticklen=6,
        showgrid=False, zeroline=False
    )
    fig_orig.update_layout(
        title=dict(text='Original Signal (reference)', font=dict(family="Arial", size=18, color="black"), x=0, xanchor="left"),
        height=240,
        margin=dict(t=60, b=50, l=70, r=30),
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", color="black", size=14),
        showlegend=False
    )
    fig_orig.update_xaxes(title_text='Time (s)', **journal_axis_settings)
    fig_orig.update_yaxes(title_text='Amplitude', **journal_axis_settings)
    st.plotly_chart(fig_orig, use_container_width=True, theme=None)

    # Analysis
    fig_twocol, transitions, stats, fig_ridge = analyze_sst_and_ridges(
        data=signal_data,
        fps=fps,
        wavelet=sst_wavelet,
        nv=nv,
        y_min=y_axis_min,
        y_max=y_axis_max,
        ridge_thresh_percent=ridge_thresh / 100.0,
        min_dist=min_dist,
        top_k_ridges=top_k,
        transition_duration_sec=transition_dur,
        transition_ratio=transition_multiplier
    )

    # The journal-ready two-panel figure
    st.plotly_chart(fig_twocol, use_container_width=True, theme=None)

    # Optional ridge figure (debug / SI)
    with st.expander("Show ridge extraction plot (debug / SI)", expanded=False):
        st.plotly_chart(fig_ridge, use_container_width=True, theme=None)

    # Summary
    st.markdown("### 📊 Analysis Summary")
    c1, c2 = st.columns(2)

    with c1:
        if stats['base_min_t'] is not None:
            st.info(
                f"🔴 **turn (Lowest Fundamental Period)**\n\n"
                f"⏱️ Time: **{stats['base_min_t']:.2f} s**\n\n"
                f"📉 Min Period: **{stats['base_min_p']:.4f} s** (~ {1/stats['base_min_p']:.2f} Hz)"
            )
        else:
            st.info("No fundamental frequency data detected yet (so no turn point).")

    with c2:
        if stats['base_max_t'] is not None:
            st.success(
                f"📈 **Highest Fundamental Period**\n\n"
                f"⏱️ Time: **{stats['base_max_t']:.2f} s**\n\n"
                f"📈 Max Period: **{stats['base_max_p']:.4f} s** (~ {1/stats['base_max_p']:.2f} Hz)"
            )

    if transitions:
        st.warning(
            f"🔔 **Transition detected (3rd > 2nd)** at **{transitions[0]:.2f} s**\n\n"
            f"Total transitions detected in sequence: {len(transitions)}"
        )
    else:
        st.markdown("No qualifying transitions detected.")
