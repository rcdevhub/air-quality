# Air Quality in London
<i>Work in Progress</i>

This repo contains a study of Nitrogen Oxide (NO2) emissions in London.

Emissions data was downloaded from the  <a href="https://www.londonair.org.uk/Londonair/API/" target="_blank">Environmental Research Group API</a>, which monitors polluton levels at over 200 sites around London. Five sites with sufficient historical data were randomly selected for the study.

Traffic is the main source of pollution in London and traffic counts are available <a href="https://roadtraffic.dft.gov.uk/downloads" target="_blank">here</a>. However, data is not at the required level of granularity (hourly) to be used in this study.

Weather can also affect pollution levels, especially wind and rain, which remove it from the air. Historical UK weather data is available from the <a href="https://roadtraffic.dft.gov.uk/downloads" target="_blank">CEDA</a> archive. One weather station (Heathrow) was chosen to represent London.

Emission levels over time are shown below, along with the main weather patterns. Unfortunately, some emissions data is missing for unknown reasons.

![Emissions over time](/plots/data_raw.png)

Selected sites are shown below, with Islington as the target site.

![Site locations](/plots/site-locations.png)
