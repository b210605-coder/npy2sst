import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import plotly.io as pio

# ============== ssqueezepy ==============
try:
    from ssqueezepy import ssq_cwt
    from ssqueezepy import cwt as ssq_cwt_plain
    from ssqueezepy.wavelets import Wavelet
    HAS_SSQ = True
except Exception:
    HAS_SSQ = False


# ==========================================
# Utilities
# ==========================================
def load_uploaded_npy(uploaded_file):
    """Load npy. Support 1D or 2D (use 2nd column)."""
    try:
        arr = np.load(uploaded_file, allow_pickle=True)
        if arr.ndim == 1:
            return arr.astype(float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 1].astype(float)
        return None
    except Exception:
        return None


def morlet_cwt_from_period_band(data, fs, y_min, y_max, n_freqs=256):
    """
    Morlet CWT scalogram restricted to [y_min, y_max] (period),
    with scales sorted ascending to satisfy ssqueezepy.cwt assumptions.
    Returns:
        mag_cwt: (n_freqs, n_time)
        periods: (n_freqs,) increasing (good for plotting)
    """
    from ssqueezepy import cwt as ssq_cwt_plain
    from ssqueezepy.wavelets import Wavelet

    f_min = 1.0 / float(y_max)
    f_max = 1.0 / float(y_min)

    freqs = np.logspace(np.log10(f_min), np.log10(f_max), int(n_freqs))

    w = Wavelet("morlet")
    wc = float(getattr(w, "wc", np.pi * 2))

    # scale = wc*fs / (2π f)
    scales = (wc * float(fs)) / (2 * np.pi * freqs)

    # ✅ 關鍵：把 scales 排成遞增（ssqueezepy 喜歡這樣）
    order = np.argsort(scales)          # ascending scales
    scales_sorted = scales[order]
    freqs_sorted = freqs[order]         # keep mapping consistent
    periods_sorted = 1.0 / freqs_sorted # y-axis should also be consistent

    Wx, _ = ssq_cwt_plain(data, wavelet="morlet", scales=scales_sorted, fs=fs)
    mag_cwt = np.abs(Wx)

    return mag_cwt, periods_sorted

# ==========================================
# Core analysis: SSWT ridges + Transition + turn
# ==========================================
def analyze_twocol_cwt_and_ridges(
    data, fps,
    sswt_wavelet, nv,
    y_min, y_max,
    ridge_thresh_percent,
    min_dist, top_k_ridges,
    transition_duration_sec,
    transition_ratio,
    cwt_nfreqs
):
    # ---- 1) SSWT for ridge picking ----
    try:
        Tx, _, ssq_freqs, _ = ssq_cwt(data, wavelet=sswt_wavelet, fs=fps, nv=nv)
    except Exception as e:
        st.error(f"SSWT 計算失敗：{e}")
        return None, [], {}

    mag_sswt = np.abs(Tx)
    with np.errstate(divide="ignore", invalid="ignore"):
        periods_sswt = 1.0 / ssq_freqs

    time_axis = np.arange(len(data)) / fps
    total_duration = float(time_axis[-1]) if len(time_axis) else 0.0

    # ---- 2) Morlet CWT for panel (a) ----
    try:
        mag_cwt, periods_cwt = morlet_cwt_from_period_band(
            data=data, fs=fps, y_min=y_min, y_max=y_max, n_freqs=cwt_nfreqs
        )
    except Exception as e:
        st.error(f"CWT（Morlet）計算失敗：{e}")
        return None, [], {}

    # ---- masks (SSWT might include outside band, we clamp for ridge) ----
    valid_sswt = np.isfinite(periods_sswt) & (periods_sswt >= y_min) & (periods_sswt <= y_max)

    # ---- ridge threshold (SSWT band max) ----
    band_mag = np.where(valid_sswt[:, None], mag_sswt, 0.0)
    global_max = float(np.max(band_mag)) if band_mag.size else 0.0
    abs_threshold = global_max * float(ridge_thresh_percent)

    # ---- store ridge points ----
    harmonic_data = {
        1: {"x": [], "y": [], "z": []},
        2: {"x": [], "y": [], "z": []},
        3: {"x": [], "y": [], "z": []},
        0: {"x": [], "y": [], "z": []},
    }

    # ---- transition detection ----
    transition_events = []
    consecutive = 0
    required_frames = int(max(0.0, transition_duration_sec) * fps)
    current_start = None
    in_transition = False

    nT = mag_sswt.shape[1]
    for t_idx in range(nT):
        slice_ = np.array(mag_sswt[:, t_idx], copy=True)
        slice_[~valid_sswt] = 0.0

        peaks, props = find_peaks(
            slice_,
            height=abs_threshold,
            distance=int(min_dist),
        )

        if len(peaks) > 0:
            peak_periods = periods_sswt[peaks]
            peak_energies = props.get("peak_heights", np.array([]))

            # keep top K
            order = np.argsort(peak_energies)[::-1]
            keep = order[: int(top_k_ridges)]
            final_periods = peak_periods[keep]
            final_energies = peak_energies[keep]

            # anchor fundamental = longest period
            base_idx = int(np.argmax(final_periods))
            T_base = float(final_periods[base_idx])
            E_base = float(final_energies[base_idx])
            t_val = float(time_axis[t_idx])

            # classify
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

                harmonic_data[h]["x"].append(t_val)
                harmonic_data[h]["y"].append(p_val)
                harmonic_data[h]["z"].append(e_val)

            # transition: E3 > E2 * multiplier
            mask_2 = (periods_sswt >= T_base / 2.2) & (periods_sswt <= T_base / 1.8)
            mask_3 = (periods_sswt >= T_base / 3.2) & (periods_sswt <= T_base / 2.8)

            E2 = float(np.max(slice_[mask_2])) if np.any(mask_2) else 0.0
            E3 = float(np.max(slice_[mask_3])) if np.any(mask_3) else 0.0
            min_required = E_base * 0.05

            if (E3 > E2 * float(transition_ratio)) and (E3 > min_required):
                if not in_transition:
                    current_start = t_val
                    in_transition = True
                consecutive += 1
            else:
                if in_transition and consecutive >= required_frames and current_start is not None:
                    transition_events.append(current_start)
                in_transition = False
                consecutive = 0

        else:
            if in_transition and consecutive >= required_frames and current_start is not None:
                transition_events.append(current_start)
            in_transition = False
            consecutive = 0

    if in_transition and consecutive >= required_frames and current_start is not None:
        transition_events.append(current_start)

    # ---- turn point = lowest fundamental period (global min) ----
    stats = {"turn_t": None, "turn_p": None}
    if len(harmonic_data[1]["y"]) > 0:
        y = np.array(harmonic_data[1]["y"], dtype=float)
        x = np.array(harmonic_data[1]["x"], dtype=float)
        m = np.isfinite(y) & (y > 0) & np.isfinite(x)
        y, x = y[m], x[m]
        if y.size:
            idx = int(np.argmin(y))
            stats["turn_t"] = float(x[idx])
            stats["turn_p"] = float(y[idx])

    # ==========================================
    # Build journal-style two-column figure
    # (a) Morlet CWT scalogram
    # (b) Ridge extraction scatter
    # ==========================================
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("(a) CWT (Morlet)", "(b) Ridge Extraction (SSWT-based)"),
        horizontal_spacing=0.14,
    )

    # Panel (a): CWT heatmap
    fig.add_trace(
        go.Heatmap(
            z=mag_cwt,
            x=time_axis,
            y=periods_cwt,
            coloraxis="coloraxis",
            name="CWT",
        ),
        row=1,
        col=1,
    )

    # Panel (b): ridges scatter
    labels = {1: "1st", 2: "2nd", 3: "3rd", 0: "Others"}
    symbols = {1: "circle", 2: "diamond", 3: "cross", 0: "x"}

    all_z = []
    for k in harmonic_data:
        all_z.extend(harmonic_data[k]["z"])
    zmin, zmax = (float(min(all_z)), float(max(all_z))) if all_z else (0.0, 1.0)

    for k in [1, 2, 3, 0]:
        d = harmonic_data[k]
        if len(d["x"]) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=d["x"],
                y=d["y"],
                mode="markers",
                marker=dict(
                    symbol=symbols[k],
                    size=6 if k == 1 else 5,
                    color=d["z"],
                    coloraxis="coloraxis2",
                    line=dict(width=0.5, color="black") if k == 1 else None,
                ),
                name=labels[k],
                hovertemplate=f"<b>{labels[k]}</b><br>Time: %{{x:.2f}} s<br>Period: %{{y:.4f}} s<br>Energy: %{{marker.color:.2f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    # Transition line on both panels
    if transition_events:
        t0 = float(transition_events[0])
        fig.add_vline(x=t0, line_width=1.5, line_dash="dash", line_color="black", row=1, col=1)
        fig.add_vline(x=t0, line_width=1.5, line_dash="dash", line_color="black", row=1, col=2)

        fig.add_annotation(
            x=t0,
            y=y_max,
            xref="x2",
            yref="y2",
            text="Transition",
            showarrow=False,
            yshift=18,
            font=dict(family="Arial", size=12, color="black"),
        )

    # Turn marker on ridge panel
    if stats["turn_t"] is not None and stats["turn_p"] is not None:
        fig.add_trace(
            go.Scatter(
                x=[stats["turn_t"]],
                y=[stats["turn_p"]],
                mode="markers+text",
                text=["turn"],
                textposition="top left",
                marker=dict(symbol="circle-open", size=12, line=dict(width=2.5, color="crimson")),
                hovertemplate="<b>turn</b><br>Time: %{x:.2f} s<br>Period: %{y:.4f} s<extra></extra>",
                name="turn",
            ),
            row=1,
            col=2,
        )

    # ---- Layout fixes: stop everything from colliding ----
    journal_layout = dict(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=14, color="black"),
        margin=dict(t=100, b=70, l=80, r=100),
        height=520,
        showlegend=False,
    )

    # axis styling
    axis_settings = dict(
        showline=True,
        linecolor="black",
        linewidth=1.5,
        mirror=True,
        ticks="inside",
        tickcolor="black",
        tickwidth=1.5,
        ticklen=6,
        showgrid=False,
        zeroline=False,
    )

    y_range = [np.log10(y_min), np.log10(y_max)]

    fig.update_layout(
        **journal_layout,
        # Two separate colorbars, placed so they don't murder your titles
        coloraxis=dict(
            colorscale="Viridis",
            colorbar=dict(
                title="|CWT|",
                thickness=14,
                len=0.82,
                x=0.46,  # between panels, slightly left
                y=0.5,
                outlinewidth=1,
                outlinecolor="black",
            ),
        ),
        coloraxis2=dict(
            colorscale="Viridis",
            cmin=zmin,
            cmax=zmax if zmax > zmin else (zmin + 1.0),
            colorbar=dict(
                title="Energy",
                thickness=14,
                len=0.82,
                x=1.06,  # outside right
                y=0.5,
                outlinewidth=1,
                outlinecolor="black",
            ),
        ),
    )

    # Subplot titles not glued to the top edge
    fig.update_annotations(font=dict(size=16, family="Times New Roman", color="black"), y=1.06)

    # Axes: both log period, same range
    fig.update_xaxes(title_text="Time (s)", title_standoff=10, range=[0, total_duration], **axis_settings, row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", title_standoff=10, range=[0, total_duration], **axis_settings, row=1, col=2)

    fig.update_yaxes(title_text="Period (s)", title_standoff=12, type="log", range=y_range, **axis_settings, row=1, col=1)
    fig.update_yaxes(title_text="Period (s)", title_standoff=12, type="log", range=y_range, **axis_settings, row=1, col=2)

    return fig, transition_events, stats


# ==========================================
# Streamlit App
# ==========================================
st.set_page_config(page_title="Two-Column Figure: Morlet CWT + Ridge", layout="wide")
st.title("🧪 Two-Column Figure: (a) Morlet CWT + (b) Ridge Extraction")

if not HAS_SSQ:
    st.error("缺套件。請安裝：pip install -U ssqueezepy scipy plotly streamlit numpy kaleido")
    st.stop()

with st.sidebar:
    st.header("⚙️ Parameters")

    fps = st.number_input("Sampling Rate (FPS)", value=30.0, min_value=1.0)

    with st.expander("1) Transform Settings", expanded=False):
        sswt_wavelet = st.selectbox("SSWT Wavelet (for ridges)", ["morlet", "bump"], index=0)
        nv = st.select_slider("SSWT Voices (nv)", options=[16, 32, 64], value=32)
        cwt_nfreqs = st.slider("CWT frequency bins", 64, 512, 256, 64)

    st.subheader("2) Period Range (s)")
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

    st.subheader("5) Export (PDF)")
    export_width = st.number_input("PDF width (px)", value=1600, min_value=800, step=100)
    export_height = st.number_input("PDF height (px)", value=520, min_value=400, step=20)

uploaded_file = st.file_uploader("Upload .npy Data File", type=["npy"])

if uploaded_file is not None:
    x = load_uploaded_npy(uploaded_file)
    if x is None:
        st.error("讀檔失敗：需要 1D array 或 2D array（至少兩欄，取第 2 欄）。")
        st.stop()

    x = x - np.mean(x)

    fig, transitions, stats = analyze_twocol_cwt_and_ridges(
        data=x,
        fps=fps,
        sswt_wavelet=sswt_wavelet,
        nv=nv,
        y_min=y_min,
        y_max=y_max,
        ridge_thresh_percent=ridge_thresh / 100.0,
        min_dist=min_dist,
        top_k_ridges=top_k,
        transition_duration_sec=transition_dur,
        transition_ratio=transition_ratio,
        cwt_nfreqs=cwt_nfreqs,
    )

    if fig is None:
        st.stop()

    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("### ⬇️ Download PDF")
    try:
        fig_export = go.Figure(fig.to_dict())
        fig_export.update_layout(width=int(export_width), height=int(export_height))

        pdf_bytes = pio.to_image(fig_export, format="pdf", engine="kaleido")
        st.download_button(
            label="Download two-column figure (PDF)",
            data=pdf_bytes,
            file_name="two_column_CWT_morlet_ridge.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(
            "PDF 匯出失敗。通常是 kaleido 沒裝或環境不對。\n"
            "請看下方教學。錯誤訊息：\n"
            f"{e}"
        )

    st.markdown("### 📌 Summary")
    if stats.get("turn_t") is not None:
        st.write(f"- **turn**: {stats['turn_t']:.2f} s, period = {stats['turn_p']:.4f} s (~ {1/stats['turn_p']:.2f} Hz)")
    else:
        st.write("- **turn**: not detected")

    if transitions:
        st.write(f"- **Transition** (first): {transitions[0]:.2f} s, total = {len(transitions)}")
    else:
        st.write("- **Transition**: none detected")
