from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

class DemandRequest(BaseModel):
    housing_type: str
    region: str
    future_years: int
    final_market_share: float

PAST_FILE = "GOGN_VERKX.xlsx"
FUTURE_FILE = "Framtidarspa.xlsx"

def load_excel(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [col.strip().lower() for col in df.columns]
    return df

def filter_data(df, region, demand_column):
    df = df[df['landshluti'].str.strip() == region.strip()].copy()
    df.columns = [col.strip().lower() for col in df.columns]
    demand_column = demand_column.lower()
    if demand_column not in df.columns:
        raise KeyError(f"Dálkur '{demand_column}' fannst ekki í gögnum.")
    df['ar'] = pd.to_numeric(df['ar'], errors='coerce')
    df = df.dropna(subset=['ar', demand_column])
    df = df.sort_values('ar')
    return df[['ar', demand_column]]

def linear_forecast(df, demand_column, future_years):
    X = df[['ar']].values
    y = df[demand_column].values
    model = LinearRegression().fit(X, y)
    last_year = int(df['ar'].max())
    future_years_range = np.array(range(last_year + 1, last_year + 1 + future_years))
    predictions = model.predict(future_years_range.reshape(-1, 1))
    return future_years_range, predictions

def monte_carlo_simulation(values, market_shares, simulations=10000, volatility=0.1):
    mean_val = np.mean(values)
    scale = abs(mean_val * volatility)
    results = []
    for _ in range(simulations):
        noise = np.random.normal(0, scale, len(values))
        simulated = (values + noise) * market_shares
        results.append(simulated)
    return np.array(results)

@app.post("/calculate_demand")
def calculate_demand(input: DemandRequest):
    try:
        initial_share = input.final_market_share * np.random.uniform(0.05, 0.1)
        market_shares = np.linspace(initial_share, input.final_market_share, input.future_years)

        sheet_name = f"{input.housing_type} eftir landshlutum"
        use_forecast = input.housing_type.lower() in ["íbúðir", "leikskólar"]

        past_df = load_excel(PAST_FILE, sheet_name)
        demand_column = 'fjoldi eininga'
        past_data = filter_data(past_df, input.region, demand_column)

        if past_data.empty:
            return {"error": "Engin fortíðargögn fundust fyrir valinn landshluta."}

        result = {}

        if use_forecast:
            future_df = load_excel(FUTURE_FILE, sheet_name)
            if 'sviðsmynd' in future_df.columns:
                future_df = future_df[future_df['sviðsmynd'].str.lower() == 'miðspá']
            future_df['ar'] = pd.to_numeric(future_df['ar'], errors='coerce')
            future_data = filter_data(future_df, input.region, 'fjoldi eininga')

            if future_data.empty:
                return {"error": "Engin framtíðarspágögn fundust."}

            future_vals = future_data['fjoldi eininga'].values[:input.future_years]
            linear_years, linear_pred = linear_forecast(past_data, demand_column, input.future_years)
            linear_pred = linear_pred[:len(future_vals)]

            avg_vals = (linear_pred + future_vals) / 2

            result["future_years"] = linear_years.tolist()
            result["linear_prediction"] = (linear_pred * market_shares).tolist()
            result["future_spa"] = (future_vals * market_shares).tolist()
            result["average_spa"] = (avg_vals * market_shares).tolist()

        else:
            future_years, pred = linear_forecast(past_data, demand_column, input.future_years)
            result["future_years"] = future_years.tolist()
            result["linear_prediction"] = (pred * market_shares).tolist()

        return result

    except Exception as e:
        return {"error": str(e)}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now (can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
