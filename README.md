# Singapore Environmental Intelligence Pipeline

An automated ETL pipeline that ingests, processes, and analyses Singapore environmental data to generate actionable intelligence recommendations.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_DEPLOYED_URL)

---

## Deployment

**Live app:** [YOUR_DEPLOYED_URL](YOUR_DEPLOYED_URL)

### Run locally

```bash
# 1. Clone the repo and create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
echo OPENAI_API_KEY=sk-... > .env

# 4. Run the pipeline to generate initial data
python main.py

# 5. Launch the dashboard
streamlit run app.py
```

The app auto-refreshes cached data every **5 minutes**. You can also trigger a manual refresh at any time using the **Refresh Data** button in the sidebar.
