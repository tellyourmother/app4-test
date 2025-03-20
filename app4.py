# Visualization
def plot_performance_graphs(df, player_name, opponent_team, last_n_seasons):
    if df is None or df.empty:
        st.warning("⚠️ No data available for selected filters.")
        return

    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=[
            f"🏀 {player_name} - Points (PTS) vs {opponent_team}",
            f"📊 {player_name} - Rebounds (REB) vs {opponent_team}",
            f"🎯 {player_name} - Assists (AST) vs {opponent_team}",
            f"🔥 {player_name} - PRA (PTS+REB+AST) vs {opponent_team}"
        ],
        horizontal_spacing=0.12, vertical_spacing=0.15
    )

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df[::-1]
    df[['PTS', 'REB', 'AST']] = df[['PTS', 'REB', 'AST']].apply(pd.to_numeric)
    df["Game Label"] = df["GAME_DATE"].dt.strftime('%b %d, %Y') + " (" + df["LOCATION"] + ")"

    avg_pts, avg_reb, avg_ast = df["PTS"].mean(), df["REB"].mean(), df["AST"].mean()
    colors_pts = ["#4CAF50" if pts > avg_pts else "#2196F3" for pts in df["PTS"]]
    colors_reb = ["#FFA726" if reb > avg_reb else "#FFEB3B" for reb in df["REB"]]
    colors_ast = ["#AB47BC" if ast > avg_ast else "#9575CD" for ast in df["AST"]]

    # Points
    fig.add_trace(go.Bar(x=df["Game Label"], y=df["PTS"], marker=dict(color=colors_pts)), row=1, col=1)
    fig.add_hline(y=avg_pts, line_dash="dash", line_color="gray", row=1, col=1, annotation_text=f"Avg PTS: {avg_pts:.1f}")

    # Rebounds
    fig.add_trace(go.Bar(x=df["Game Label"], y=df["REB"], marker=dict(color=colors_reb)), row=1, col=2)
    fig.add_hline(y=avg_reb, line_dash="dash", line_color="gray", row=1, col=2, annotation_text=f"Avg REB: {avg_reb:.1f}")

    # Assists
    fig.add_trace(go.Bar(x=df["Game Label"], y=df["AST"], marker=dict(color=colors_ast)), row=2, col=1)
    fig.add_hline(y=avg_ast, line_dash="dash", line_color="gray", row=2, col=1, annotation_text=f"Avg AST: {avg_ast:.1f}")

    # PRA
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    avg_pra = df["PRA"].mean()
    colors_pra = ["#FF3D00" if pra > avg_pra else "#FF8A65" for pra in df["PRA"]]

    fig.add_trace(go.Bar(x=df["Game Label"], y=df["PRA"], marker=dict(color=colors_pra)), row=2, col=2)
    fig.add_hline(y=avg_pra, line_dash="dash", line_color="gray", row=2, col=2, annotation_text=f"Avg PRA: {avg_pra:.1f}")

    fig.update_layout(title_text=f"{player_name} vs {opponent_team} Performance (Last {last_n_seasons} Seasons)", 
                      template="plotly_dark", height=800, width=1200)
    st.plotly_chart(fig)
