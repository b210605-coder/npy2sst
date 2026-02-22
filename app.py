import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import plotly.io as pio

# =========================
# Imports from ssqueezepy
# =========================
try:
    from ssqueezepy import ssq_cwt
    HAS_SSQ = True
except ImportError:
    HAS_SSQ = False

# For true CWT (so panel (a) actually exists and is correct)
try:
    from ssqueezepy import cwt as ssq_cwt_plain
    HAS_CWT = True
except ImportError:
    HAS_CWT = False


# ==========================================
# Helper: convert CWT scales -> frequencies
# ==========================================
def scales_to_freqs_fallback(scales, fs):
    """
    Fallback approximate mapping if ssqueezepy helpers unavailable.
    Not perfect, but better than lying with a mismatched axis.
    """
    scales = np.asarray(scales, dtype=float)
    # crude pseudo-frequency ~ fs / scale
    freqs = fs / np.maximum(scales, 1e-12)
    return freqs


def scales_to_freqs(scales, wavelet, fs):
    """
    Try ssqueezepy utilities; fall back to approximation if unavailable.
    """
    # Try ssqueezepy helper(s)
    try:
        # Some ssqueezepy versions expose wavelets with .center_frequency / .wc
        from ssqueezepy.wavelets import Wavelet
        w = Wavelet(wavelet)
        # Many wavelets provide center frequency wc; pseudo-freq: wc * fs / (2*pi*scale) or wc*fs/scale
        # Different conventions exist; ssqueezepy often uses wc/(2*pi*scale) for radian center freq.
        wc = getattr(w, "wc", None)
        if wc is not None and np.isfinite(wc):
            freqs = (wc * fs) / (2 * np.pi * np.maximum(scales, 1e-12))
            return freqs
    except Exception:
        pass

    # Try experimental / utils if present
    try:
        from ssqueezepy.utils import scale_to_freq  # not guaranteed
        freqs = scale_to_freq(scales, wavelet=wavelet, fs=fs)
        return freqs
    except Exception:
        pass

    # Fallback approximation
    return scales_to_freqs_fallback(scales, fs)


# ==========================================
# Core Analysis (CWT panel + Ridge panel)
# ==========================================
def analyze_for_twocol_figure(
    data, fps, wavelet, nv, y_min, y_max,
    ridge_thresh_percent, min_dist, top_k_ridges,
    transition_duration_sec, transition_ratio
):
    # ----- 1) Compute SSWT (for ridge picking) -----
    try:
        Tx, _, ssq_freqs, _ = ssq_cwt(data, wavelet=wavelet, fs=fps, nv=nv)
    except Exception as e:
        st.error(f"SSWT Computation Error: {e}")
        return None, None, [], {}

    magnitude_sswt = np.abs(Tx)

    with np.errstate(divide='ignore', invalid='ignore'):
        periods_sswt = 1 / ssq_freqs

    time_axis = np.arange(len(data)) / fps
    total_duration = float(time_axis[-1]) if len(time_axis) else 0.0

    # ----- 2) Compute TRUE CWT for panel (a) -----
    # This is the part you complained about: CWT not showing.
    # We compute it explicitly to avoid axis mismatch.
    if not HAS_CWT:
        st.error("你的 ssqueezepy 版本沒有提供 cwt()，請更新：pip install -U ssqueezepy")
        return None, None, [], {}

    try:
        Wx, scales = ssq_cwt_plain(data, wavelet=wavelet, fs=fps, nv=nv)
        mag_cwt = np.abs(Wx)
        freqs_cwt = scales_to_freqs(scales, wavelet=wavelet, fs=fps)
        periods_cwt = 1 / np.maximum(freqs_cwt, 1e-12)
    except Exception as e:
        st.error(f"CWT Computation Error: {e}")
        return None, None, [], {}

    # ----- 3) Valid period masks -----
    valid_sswt = np.isfinite(periods_sswt) & (periods_sswt >= y_min) & (periods_sswt <= y_max)
    valid_cwt = np.isfinite(periods_cwt) & (periods_cwt >= y_min) & (periods_cwt <= y_max)

    # Threshold uses SSWT energy within valid band
    valid_mag = np.where(valid_sswt[:, None], magnitude_sswt, 0)
    global_max = float(np.max(valid_mag)) if valid_mag.size else 0.0
    abs_threshold = global_max * float(ridge_thresh_percent)

    # ----- 4) Ridge extraction + transition detection -----
    harmonic_data = {
        1: {'x': [], 'y': [], 'z': []},
        2: {'x': [], 'y': [], 'z': []},
        3: {'x': [], 'y': [], 'z': []},
        0: {'x': [], 'y': [], 'z': []}
    }

    transition_events = []
    consecutive_frames = 0
    required_frames = int(max(0.0, transition_duration_sec) * fps)
    current_transition_start_time = None
    in_transition = False

    num_time_steps = magnitude_sswt.shape[1]

    for t_idx in range(num_time_steps):
        spectrum_slice = np.array(magnitude_sswt[:, t_idx], copy=True)
        spectrum_slice[~valid_sswt] = 0

        peaks, props = find_peaks(
            spectrum_slice,
            height=abs_threshold,
            distance=int(min_dist)
        )

        if len(peaks) > 0:
            peak_periods = periods_sswt[peaks]
            peak_energies = props.get("peak_heights", np.array([]))

            # keep top-K
            order = np.argsort(peak_energies)[::-1]
            keep = order[:int(top_k_ridges)]
            final_periods = peak_periods[keep]
            final_energies = peak_energies[keep]

            # anchor = longest period = fundamental
            base_idx = int(np.argmax(final_periods))
            T_base = float(final_periods[base_idx])
            E_base = float(final_energies[base_idx])
            t_val = float(time_axis[t_idx])

            # classify harmonics
            for p_val, e_val in zip(final_periods, final_energies):
                p_val = float(p_val)
                e_val = float(e_val)
                if p_val <= 0 or not np.isfinite(p_val):
                    h = 0
                else:
                    ratio = T_base / p_val
                    if 0.85 <= ratio <= 1.15:
                        h = 1
                    elif 1.8 <= ratio <= 2.2:
                        h = 2
                    elif 2.8 <= ratio <= 3.2:
                        h = 3
                    else:
                        h = 0
                harmonic_data[h]['x'].append(t_val)
                harmonic_data[h]['y'].append(p_val)
                harmonic_data[h]['z'].append(e_val)

            # transition detection: E3 > E2 * transition_ratio
            mask_2 = (periods_sswt >= T_base / 2.2) & (periods_sswt <= T_base / 1.8)
            mask_3 = (periods_sswt >= T_base / 3.2) & (periods_sswt <= T_base / 2.8)

            E2 = float(np.max(spectrum_slice[mask_2])) if np.any(mask_2) else 0.0
            E3 = float(np.max(spectrum_slice[mask_3])) if np.any(mask_3) else 0.0
            min_required = E_base * 0.05

            if (E3 > E2 * float(transition_ratio)) and (E3 > min_required):
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

    # ----- 5) turn point = lowest fundamental period (global) -----
    stats = {'turn_t': None, 'turn_p': None}
    if len(harmonic_data[1]['y']) > 0:
        y = np.array(harmonic_data[1]['y'], dtype=float)
        x = np.array(harmonic_data[1]['x'], dtype=float)
        m = np.isfinite(y) & (y > 0) & np.isfinite(x)
        y, x = y[m], x[m]
        if y.size:
            idx = int(np.argmin(y))
            stats['turn_t'] = float(x[idx])
            stats['turn_p'] = float(y[idx])

    # ==========================================
    # Build two-column figure: (a) CWT, (b) Ridge extraction
    # ==========================================
    journal_layout = dict(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", color="black", size=14),
        margin=dict(t=70, b=60, l=70, r=30),
    )

    axis_settings = dict(
        showline=True, linecolor='black', linewidth=1.5,
        mirror=True, ticks='inside', tickcolor='black', tickwidth=1.5, ticklen=6,
        showgrid=False, zeroline=False
    )

    y_range = [np.log10(y_min), np.log10(y_max)] if (y_min > 0 and y_max > 0) else None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("(a) CWT", "(b) Ridge Extraction (SSWT-based)"),
        horizontal_spacing=0.10
    )

    # Panel (a): CWT scalogram
    plot_periods_cwt = periods_cwt[valid_cwt]
    plot_mag_cwt = mag_cwt[valid_cwt, :]

    fig.add_trace(
        go.Heatmap(
            z=plot_mag_cwt,
            x=time_axis,
            y=plot_periods_cwt,
            coloraxis="coloraxis",
            name="CWT"
        ),
        row=1, col=1
    )

    # Panel (b): Ridge extraction points
    labels = {1: "1st Harmonic", 2: "2nd Harmonic", 3: "3rd Harmonic", 0: "Others"}
    markers = {1: "circle", 2: "diamond", 3: "cross", 0: "x"}

    # Ridge colorscale based on energy
    all_z = []
    for k in harmonic_data:
        all_z.extend(harmonic_data[k]['z'])
    rzmin, rzmax = (float(min(all_z)), float(max(all_z))) if all_z else (0.0, 1.0)

    for k in [1, 2, 3, 0]:
        d = harmonic_data[k]
        if len(d['x']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=d['x'], y=d['y'],
                    mode="markers",
                    name=labels[k],
                    marker=dict(
                        symbol=markers.get(k, "circle"),
                        size=6 if k == 1 else 5,
                        color=d['z'],
                        coloraxis="coloraxis2",
                        line=dict(width=0.5, color="black") if k == 1 else None
                    ),
                    hovertemplate=f"<b>{labels[k]}</b><br>Time: %{{x:.2f}} s<br>Period: %{{y:.4f}} s<br>Energy: %{{marker.color:.2f}}<extra></extra>"
                ),
                row=1, col=2
            )

    # Transition line on both panels
    if transition_events:
        t0 = float(transition_events[0])
        fig.add_vline(x=t0, line_width=2, line_dash="dash", line_color="black", row=1, col=1)
        fig.add_vline(x=t0, line_width=2, line_dash="dash", line_color="black", row=1, col=2)

        # annotate in ridge panel
        if y_max > 0:
            fig.add_annotation(
                x=t0, y=y_max,
                xref="x2", yref="y2",
                text="Transition",
                showarrow=False,
                yshift=18,
                font=dict(family="Arial", color="black", size=12)
            )

    # Turn marker on ridge panel
    if stats['turn_t'] is not None and stats['turn_p'] is not None:
        fig.add_trace(
            go.Scatter(
                x=[stats['turn_t']], y=[stats['turn_p']],
                mode="markers+text",
                text=["turn"],
                textposition="top center",
                name="turn",
                marker=dict(symbol="circle-open", size=12, line=dict(width=2.5, color="crimson")),
                hovertemplate="<b>turn</b><br>Time: %{x:.2f} s<br>Period: %{y:.4f} s<extra></extra>"
            ),
            row=1, col=2
        )

    fig.update_layout(
        height=520,
        showlegend=False,  # 期刊圖通常不想要一堆 legend 堵住畫面
        coloraxis=dict(
            colorscale="Viridis",
            colorbar=dict(
                title=dict(text="|CWT|", font=dict(family="Arial", size=13, color="black")),
                tickfont=dict(family="Arial", size=11, color="black"),
                outlinewidth=1, outlinecolor="black",
                thickness=18
            )
        ),
        coloraxis2=dict(
            colorscale="Viridis",
            cmin=rzmin, cmax=rzmax,
            colorbar=dict(
                title=dict(text="Ridge Energy", font=dict(family="Arial", size=13, color="black")),
                tickfont=dict(family="Arial", size=11, color="black"),
                outlinewidth=1, outlinecolor="black",
                thickness=18,
                x=1.02  # 放右側外面一點
            )
        ),
        **journal_layout
    )

    # axes: both log period, same range
    fig.update_xaxes(title_text="Time (s)", range=[0, total_duration], **axis_settings, row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", range=[0, total_duration], **axis_settings, row=1, col=2)

    fig.update_yaxes(title_text="Period (s)", type="log", range=y_range, **axis_settings, row=1, col=1)
    fig.update_yaxes(title_text="Period (s)", type="log", range=y_range, **axis_settings, row=1, col=2)

    return fig, transition_events, stats


# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(page_title="Two-Column Figure: CWT + Ridge", layout="wide")
st.title("🧪 Analytical Chemistry Two-Column Figure: (a) CWT + (b) Ridge Extraction")

if not HAS_SSQ:
    st.error("請先安裝：pip install ssqueezepy scipy plotly streamlit numpy")
    st.stop()

if not HAS_CWT:
    st.error("你的 ssqueezepy 沒有 cwt()。請更新：pip install -U ssqueezepy")
    st.stop()

with st.sidebar:
    st.header("⚙️ Parameters")
    fps = st.number_input("Sampling Rate (FPS)", value=30.0, min_value=1.0)

    with st.expander("1) Wavelet Settings", expanded=False):
        wavelet = st.selectbox("Wavelet", ["morlet", "bump"], index=0)
        nv = st.select_slider("Voices (nv)", options=[16, 32, 64], value=32)

    st.subheader("2) Period Display Range")
    c1, c2 = st.columns(2)
    y_min = c1.number_input("Min Period (s)", value=0.1, min_value=1e-6, format="%.6f")
    y_max = c2.number_input("Max Period (s)", value=10.0, min_value=1e-6, format="%.6f")

    st.subheader("3) Ridge Extraction")
    ridge_thresh = st.slider("Energy Threshold (%)", 1, 40, 5)
    min_dist = st.slider("Min Peak Distance (px)", 1, 50, 15)
    top_k = st.slider("Keep Top K Peaks", 1, 10, 5)

    st.subheader("4) Transition Detection")
    transition_dur = st.number_input("Trigger Duration (s)", value=0.1, step=0.05, min_value=0.0)
    transition_ratio = st.slider("E3 > E2 multiplier", 1.0, 3.0, 1.0, 0.1)

    st.subheader("5) Export")
    export_width = st.number_input("Export width (px)", value=1400, min_value=800, step=100)
    export_height = st.number_input("Export height (px)", value=520, min_value=400, step=20)


def load_uploaded_npy(uploaded_file):
    try:
        arr = np.load(uploaded_file, allow_pickle=True)
        if arr.ndim == 1:
            return arr.astype(float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 1].astype(float)
        return None
    except Exception:
        return None


uploaded_file = st.file_uploader("Upload .npy Data File", type=["npy"])

if uploaded_file is not None:
    x = load_uploaded_npy(uploaded_file)
    if x is None:
        st.error("讀檔失敗：需要 1D array 或 2D array（至少兩欄，取第 2 欄）。")
        st.stop()

    x = x - np.mean(x)

    fig, transitions, stats = analyze_for_twocol_figure(
        data=x,
        fps=fps,
        wavelet=wavelet,
        nv=nv,
        y_min=y_min,
        y_max=y_max,
        ridge_thresh_percent=ridge_thresh / 100.0,
        min_dist=min_dist,
        top_k_ridges=top_k,
        transition_duration_sec=transition_dur,
        transition_ratio=transition_ratio
    )

    if fig is None:
        st.stop()

    st.plotly_chart(fig, use_container_width=True, theme=None)

    # Summary
    st.markdown("### 📌 Summary (for your Methods / caption)")
    if stats.get("turn_t") is not None:
        st.write(
            f"- **turn** at **{stats['turn_t']:.2f} s**, "
            f"fundamental min period = **{stats['turn_p']:.4f} s** (~ {1/stats['turn_p']:.2f} Hz)"
        )
    else:
        st.write("- **turn**: not detected (no valid fundamental ridge points).")

    if transitions:
        st.write(f"- **Transition** (first) at **{transitions[0]:.2f} s**; total transitions = {len(transitions)}")
    else:
        st.write("- **Transition**: none detected with current thresholds.")

    # One-click PDF download
    st.markdown("### ⬇️ Download (PDF)")
    try:
        # Make a copy with fixed size for export
        fig_export = fig.to_dict()
        fig_export = go.Figure(fig_export)
        fig_export.update_layout(width=int(export_width), height=int(export_height))

        pdf_bytes = pio.to_image(fig_export, format="pdf", engine="kaleido")
        st.download_button(
            label="Download two-column figure (PDF)",
            data=pdf_bytes,
            file_name="two_column_CWT_ridge.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(
            "PDF 匯出失敗。通常是因為沒裝 kaleido。\n"
            "請安裝：pip install -U kaleido\n"
            f"錯誤訊息：{e}"
        )
