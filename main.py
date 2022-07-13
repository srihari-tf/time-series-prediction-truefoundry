
import logging
from fastapi.responses import HTMLResponse
from servicefoundry.service import fastapi
import joblib
import pandas as pd
from typing import List
import mlfoundry

logger = logging.getLogger(__name__)
app = fastapi.app()

client = mlfoundry.get_client(api_key='ZTg5NzdlMDQtMDU2Mi00NGFkLTg5N2MtNjlkYmIxNDNmMGQ3OjE3OWU3Ng==')
run = client.get_run('srihari/time-series-pred-skforecaster/clever-duck')
local_path = run.download_artifact('forecaster.pkl')
forecaster_loaded = joblib.load(local_path)

@app.post("/predict")
def predict(lag_series: List[dict]):
  df = pd.DataFrame(lag_series)
  df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
  df = df.set_index('date')
  df = df.asfreq('MS')

  result = forecaster_loaded.predict(last_window = df.y, steps=30)
  return dict(zip(result.index.format(), result))


@app.get("/", response_class=HTMLResponse)
def root():
    html_content = "<html><body>Open <a href='/docs'>Docs</a></body></html>"
    return HTMLResponse(content=html_content, status_code=200)

