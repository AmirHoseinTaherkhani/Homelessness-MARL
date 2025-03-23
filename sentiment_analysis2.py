import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Create a 'vis' directory if it doesn’t exist
if not os.path.exists("vis"):
    os.makedirs("vis")

# Load the CSV file (adjust the filename as needed)
df = pd.read_csv("agent_emotion_proportions.csv")

# Define agents and emotions based on your data
agents = ['law_enforcement', 'shelter_services', 'city_government', 'residents']
emotions = ['neutral', 'joy', 'surprise', 'disgust', 'sadness', 'anger', 'fear']

# Assign colors to agents (emotions will use these colors consistently)
agent_colors = {
    'law_enforcement': '#1f77b4',  # Blue
    'shelter_services': '#ff7f0e', # Orange
    'city_government': '#2ca02c',  # Green
    'residents': '#d62728'         # Red
}

# Create a 2x2 subplot grid
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[agent.replace('_', ' ').title() for agent in agents]
)

# Add traces for each agent-emotion combination
traces = []
for agent in agents:
    for emotion in emotions:
        column_name = f"{agent}_{emotion}"
        if column_name in df.columns:
            agent_data = df[df['agent'] == agent]
            # Determine subplot position
            i = agents.index(agent) + 1
            trace = go.Scatter(
                x=agent_data['episode'],
                y=agent_data[column_name],
                mode='lines',
                name=f"{agent.replace('_', ' ').title()} - {emotion.capitalize()}",
                line=dict(width=2, color=agent_colors[agent]),
                visible=False,  # Initially hidden
                legendgroup=emotion  # Group by emotion for toggling
            )
            fig.add_trace(
                trace,
                row=(i-1)//2 + 1,
                col=(i-1)%2 + 1
            )
            traces.append(trace)

# Create toggle buttons for each emotion
buttons = []
for emotion in emotions:
    # Visibility: toggle traces for this emotion across all agents
    visibility = [False] * len(traces)
    for i, trace in enumerate(traces):
        if emotion.capitalize() in trace.name:
            visibility[i] = True
    
    buttons.append(
        dict(
            label=emotion.capitalize(),
            method="update",
            args=[{"visible": visibility}],
        )
    )

# Update layout with buttons, font, and background
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.2,
            yanchor="top",
            buttons=buttons
        )
    ],
    title="Interactive Emotion Analysis: Toggle Emotions to View Trends",
    height=800,
    width=1200,
    font=dict(family="Times New Roman", size=12),
    plot_bgcolor='#f0f0f0',  # Light gray plot background
    paper_bgcolor='#f0f0f0', # Light gray paper background
    legend_title="Agent - Emotion",
    showlegend=True
)

# Set axis labels and ranges
for row in range(1, 3):
    for col in range(1, 3):
        fig.update_yaxes(title_text="Proportion", range=[0, 1], row=row, col=col)
        fig.update_xaxes(title_text="Episode", row=row, col=col)

# Save as interactive HTML
fig.write_html("vis/interactive_emotion_analysis.html")

# Optional: Display the plot
fig.show()