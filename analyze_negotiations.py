import json
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Analyze negotiation logs
def analyze_logs(log_file="negotiation_logs.json"):
    priorities = {agent: {} for agent in ["law_enforcement", "shelter_services", "city_government", "residents"]}
    with open(log_file, "r") as f:
        for line in f:
            log = json.loads(line)
            agent, argument = log["agent"], log["argument"]
            doc = nlp(argument)
            keywords = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
            for keyword in keywords:
                priorities[agent][keyword] = priorities[agent].get(keyword, 0) + 1

    # Calculate top priorities
    insights = {}
    for agent, counts in priorities.items():
        if counts:
            total = sum(counts.values())
            top_keywords = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
            insights[agent] = {k: (v / total * 100) for k, v in top_keywords}
    return insights

# Print insights
if __name__ == "__main__":
    insights = analyze_logs()
    for agent, priorities in insights.items():
        print(f"{agent.capitalize()} Priorities: {dict(priorities)}")