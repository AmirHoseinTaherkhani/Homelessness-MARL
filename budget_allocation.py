import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Create 'vis' directory if it doesn’t exist
if not os.path.exists("vis"):
    os.makedirs("vis")

# Load data
df = pd.read_json("negotiation_logs.json", lines=True)

# Function to apply moving average smoothing
def smooth_data(series, window_size=5):
    return series.rolling(window=window_size, min_periods=1).mean()

# Function to compute preference score from proposal text
def compute_preference_score(proposal):
    if not proposal:
        return 0.5  # Neutral if no proposal
    law_keywords = ["police", "enforcement", "security", "crime", "officers", "patrols", "arrests", "law", "safety"]
    shelter_keywords = ["shelter", "housing", "assistance", "support", "services", "homeless", "families", "residents", "community", "programs"]
    words = proposal.lower().split()
    law_count = sum(word in law_keywords for word in words)
    shelter_count = sum(word in shelter_keywords for word in words)
    total = law_count + shelter_count
    if total == 0:
        return 0.5  # Neutral if no relevant keywords
    return law_count / total  # Score: 1 = fully law enforcement, 0 = fully shelter services

# Compute preference score for each proposal
df['preference_score'] = df['proposal'].apply(compute_preference_score)

# Compute average preference score per agent per episode
agent_preferences = df.groupby(['agent', 'episode'])['preference_score'].mean().reset_index()

# Get the last step per episode for final budget allocation
last_steps = df.groupby('episode')['step'].max().reset_index()
final_budget = pd.merge(df, last_steps, on=['episode', 'step'])
final_budget = final_budget.groupby('episode').first().reset_index()

# Smooth the final budget values
final_budget['budget_law_enforcement'] = smooth_data(final_budget['budget_law_enforcement'])
final_budget['budget_shelter_services'] = smooth_data(final_budget['budget_shelter_services'])
final_budget['homelessness_rate'] = smooth_data(final_budget['homelessness_rate'])

# Create figure with subplots: 3 rows, 2 columns
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=["Main Budget Allocation", "Law Enforcement", "Shelter Services", "City Government", "Residents"],
    specs=[[{"colspan": 2, "secondary_y": True}, None],
           [{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "scatter"}]],
    vertical_spacing=0.15
)

# Main Budget Allocation Plot (row=1, col=1) using final budget per episode
fig.add_trace(
    go.Scatter(x=final_budget['episode'], y=final_budget['budget_law_enforcement'], name="Law Enforcement Budget", line=dict(color='rgba(65, 105, 225, .9)')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=final_budget['episode'], y=final_budget['budget_shelter_services'], name="Shelter Services Budget", line=dict(color='rgba(220, 20, 60, .9)')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=final_budget['episode'], y=final_budget['homelessness_rate'], name="Homelessness Rate", line=dict(color="seagreen")),
    row=1, col=1, secondary_y=True
)

# Agent Budget Preferences (stacked area for inferred preferences)
agents = ['law_enforcement', 'shelter_services', 'city_government', 'residents']
colors = ['#2596be', '#f0b150']

for i, agent in enumerate(agents):
    # Filter preferences for the agent
    agent_pref = agent_preferences[agent_preferences['agent'] == agent]
    
    if agent_pref.empty:
        print(f"Warning: No preference data for agent '{agent}'. Check agent names and data.")
        continue
    
    # Calculate pseudo-budget values based on preference score
    law_enforcement_pref = agent_pref['preference_score']
    shelter_services_pref = 1 - law_enforcement_pref
    
    # Map to row and column for subplots
    row = 2 if i < 2 else 3
    col = (i % 2) + 1
    
    # Add stacked area for enforcement and shelter preferences
    fig.add_trace(
        go.Scatter(
            x=agent_pref['episode'],
            y=law_enforcement_pref,
            mode='lines',
            name=f"{agent} - Enforcement Preference",
            line=dict(width=0),
            stackgroup=agent,
            fillcolor='rgba(65, 105, 225, .9)',
            opacity=1,
            showlegend=False
        ),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=agent_pref['episode'],
            y=shelter_services_pref,
            mode='lines',
            name=f"{agent} - Shelter Preference",
            line=dict(width=0),
            stackgroup=agent,
            fillcolor='rgba(220, 20, 60, .6)',
            opacity=1,
            showlegend=False
        ),
        row=row, col=col
    )

# Update layout for visual appeal and font
fig.update_layout(
    title="Balancing the Budget: How AI Agents Tackled Homelessness",
    height=1000,
    width=1200,
    font=dict(family="Times New Roman", size=14),  # Larger font for titles and labels
    showlegend=True,
    legend_title="Metrics",
    legend=dict(font=dict(size=12))  # Slightly smaller legend font
)

# Update axes with larger font sizes
fig.update_xaxes(title_text="Episode", title_font_size=14, tickfont_size=12)
fig.update_yaxes(title_text="Budget Allocation", title_font_size=14, tickfont_size=12, row=1, col=1)
fig.update_yaxes(title_text="Homelessness Rate", title_font_size=14, tickfont_size=12, row=1, col=1, secondary_y=True)

for i in range(2, 4):  # Rows 2 and 3 for agents
    for j in range(1, 3):  # Columns 1 and 2
        fig.update_xaxes(title_text="Episode", title_font_size=14, tickfont_size=12, row=i, col=j)
        fig.update_yaxes(title_text="Preference Proportion", title_font_size=14, tickfont_size=12, row=i, col=j)

# Save as HTML
fig.write_html("vis/budget_allocation.html")