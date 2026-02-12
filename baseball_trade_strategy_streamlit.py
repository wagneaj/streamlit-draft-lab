import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =====================================
# PAGE CONFIG + DASHBOARD AESTHETICS
# =====================================

st.set_page_config(page_title="Draft Optimization Lab", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0A0F14 0%, #0E141B 100%);
}

html, body, [class*="css"]  {
    color: #E6EDF3;
    font-family: Inter, system-ui, sans-serif;
}

h1 { font-weight: 600; letter-spacing: -0.5px; }
h2, h3 { color: #DCE6F2; }

[data-testid="stDataFrame"] {
    background-color: rgba(255,255,255,0.02);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.05);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1016 0%, #0F1620 100%);
}

.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# =====================================
# CONFIG
# =====================================

TEAMS = ["A", "B", "C", "D", "E", "F"]
N_ROUNDS = 11
PICKS_PER_ROUND = len(TEAMS)
TOTAL_PLAYERS = N_ROUNDS * PICKS_PER_ROUND

TEAM_COLORS = {
    "A": "#FFD166",  # Soft Yellow for Team A
    "B": "#61DDAA",
    "C": "#65789B",
    "D": "#F6BD16",
    "E": "#7262FD",
    "F": "#78D3F8",
}

# =====================================
# DRAFT ORDER
# =====================================

def generate_default_draft_order():
    order = {}
    forward = TEAMS.copy()
    reverse = list(reversed(TEAMS))

    for r in range(1, N_ROUNDS + 1):
        order[r] = forward.copy() if r % 2 == 1 else reverse.copy()

    return order

# =====================================
# DISTRIBUTION MODEL
# =====================================

def generate_player_values(dist_type, alpha, normal_sigma, noise_scale):
    ranks = np.arange(1, TOTAL_PLAYERS + 1)

    if dist_type == "Power Law":
        base_curve = 1 / (ranks ** alpha)

    elif dist_type == "Normal":
        x = np.linspace(-2.5, 2.5, TOTAL_PLAYERS)
        base_curve = np.exp(-(x**2) / (2 * normal_sigma**2))
        base_curve = base_curve[::-1]
        base_curve = base_curve / base_curve.max()

    elif dist_type == "Hybrid":
        power = 1 / (ranks ** alpha)
        x = np.linspace(-2.5, 2.5, TOTAL_PLAYERS)
        normal = np.exp(-(x**2) / (2 * normal_sigma**2))
        normal = normal[::-1]
        normal = normal / normal.max()
        base_curve = 0.5 * power + 0.5 * normal

    noise = np.random.normal(0, noise_scale, TOTAL_PLAYERS)
    values = np.clip(base_curve + noise, 0, None)

    return ranks, base_curve, values

# =====================================
# TEAM VALUE
# =====================================

def compute_team_values(order, player_values):
    team_totals = {t: 0 for t in TEAMS}
    team_rank_positions = {t: [] for t in TEAMS}

    rank_idx = 0
    for r in range(1, N_ROUNDS + 1):
        for team in order[r]:
            team_totals[team] += player_values[rank_idx]
            team_rank_positions[team].append(rank_idx + 1)
            rank_idx += 1

    return team_totals, team_rank_positions

# =====================================
# SHAPLEY-LIKE PICK CONTRIBUTION
# =====================================

def compute_pick_contributions(order, base_curve):
    contrib = []
    rank_idx = 0

    for r in range(1, N_ROUNDS + 1):
        for team in order[r]:
            contrib.append({
                "Round": r,
                "Team": team,
                "Rank": rank_idx + 1,
                "Value": base_curve[rank_idx]
            })
            rank_idx += 1

    return pd.DataFrame(contrib)

# =====================================
# UI
# =====================================

st.title("⚾ Draft Optimization Lab")
st.caption("Talent Curve + Expected Value + Pick Contribution")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.header("Talent Distribution")

dist_type = st.sidebar.selectbox(
    "Distribution Type",
    ["Power Law", "Normal", "Hybrid"]
)

alpha = st.sidebar.slider("Power Law Strength", 0.5, 2.5, 1.2, 0.05)
normal_sigma = st.sidebar.slider("Normal Spread", 0.5, 2.5, 1.2, 0.1)
noise_scale = st.sidebar.slider("Evaluation Noise", 0.0, 0.1, 0.02, 0.005)

st.sidebar.header("Monte Carlo")
use_custom_mc = st.sidebar.toggle("Custom Monte Carlo Runs", value=False)

if use_custom_mc:
    mc_runs = st.sidebar.slider("Monte Carlo Runs", 50, 5000, 500, 50)
else:
    mc_runs = 400

show_mc_curves = st.sidebar.toggle("Show Multiple MC Curves", value=False)

# =====================================
# DATA
# =====================================

base_order = generate_default_draft_order()

# =====================================
# LAYOUT
# =====================================

col_left, col_right = st.columns([1.1, 1.4])

# LEFT
with col_left:

    st.subheader("Draft Order")
    order_df = pd.DataFrame(base_order).T
    order_df.columns = [f"Pick {i+1}" for i in range(PICKS_PER_ROUND)]

    edited_order_df = st.data_editor(order_df, use_container_width=True)

    # Convert edited DF back to dict order
    working_order = {
        int(idx): list(row.values)
        for idx, row in edited_order_df.iterrows()
    }

    # Monte Carlo using working order
    sim_results = []
    for _ in range(mc_runs):
        _, base_curve_tmp, vals = generate_player_values(dist_type, alpha, normal_sigma, noise_scale)
        totals, _ = compute_team_values(working_order, vals)
        sim_results.append(totals)

    sim_df = pd.DataFrame(sim_results)

    st.subheader("Expected Team Draft Value")
    st.dataframe(pd.DataFrame({
        "Mean": sim_df.mean(),
        "StdDev": sim_df.std()
    }))

# RIGHT GRAPH
with col_right:

    # Visualization sample based on working order
    ranks, base_curve, vis_values = generate_player_values(dist_type, alpha, normal_sigma, noise_scale)
    _, team_rank_positions = compute_team_values(working_order, vis_values)

    # Contributions
    contrib_df = compute_pick_contributions(working_order, base_curve)

    st.subheader("League Talent Curve")

    fig = go.Figure()

    # Optional MC sample curves
    if show_mc_curves:
        # Draw multiple sampled curves from noisy player value generations
        for _ in range(20):
            r_tmp, base_tmp, noisy_tmp = generate_player_values(dist_type, alpha, normal_sigma, noise_scale)
            fig.add_trace(go.Scatter(
                x=r_tmp,
                y=noisy_tmp,
                line=dict(width=1, color='rgba(180,180,180,0.25)'),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Curve
    fig.add_trace(go.Scatter(
        x=ranks,
        y=base_curve,
        name="Expected Curve",
        line=dict(width=3)
    ))

    # Player Points (Soft White)
    fig.add_trace(go.Scatter(
        x=ranks,
        y=vis_values,
        mode='markers',
        marker=dict(size=10, color='rgba(255,255,255,0.85)', line=dict(width=1, color='rgba(255,255,255,0.3)')),
        name="Players",
        text=[f"Rank {r}<br>Skill {v:.3f}" for r, v in zip(ranks, vis_values)],
        hovertemplate="%{text}<extra></extra>"
    ))

    # Team A Highlights (Soft Yellow)
    team_a_ranks = team_rank_positions["A"]
    team_a_vals = [vis_values[r-1] for r in team_a_ranks]

    fig.add_trace(go.Scatter(
        x=team_a_ranks,
        y=team_a_vals,
        mode='markers',
        marker=dict(size=14, color=TEAM_COLORS["A"]),
        name="Team A Picks"
    ))

    fig.update_layout(
        height=600,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Player Rank",
        yaxis_title="Relative Skill"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================
# PICK CONTRIBUTION (SHAPLEY-LIKE)
# =====================================

st.markdown("---")
st.subheader("Pick Contribution by Rank")

fig2 = go.Figure()

for team in TEAMS:
    tdf = contrib_df[contrib_df.Team == team]
    fig2.add_trace(go.Bar(
        x=tdf.Rank,
        y=tdf.Value,
        name=f"Team {team}",
        marker_color=TEAM_COLORS[team],
        opacity=0.7
    ))

fig2.update_layout(
    barmode='stack',
    height=400,
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis_title="Player Rank",
    yaxis_title="Contribution to Team Value"
)

st.plotly_chart(fig2, use_container_width=True)

# =====================================
# NOTES
# =====================================

st.caption("""
New Features:
• Power Law / Normal / Hybrid distributions
• Soft white player markers
• Pick contribution (Shapley-like visualization)
• Adjustable Monte Carlo run count
""")
