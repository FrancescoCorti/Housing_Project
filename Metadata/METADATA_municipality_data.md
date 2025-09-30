# municipality_data - METADATA
**Data Source**: [IstatData](https://aster.istat.it/#/it/atlante_comuni/categories/CARTA_IDENTITA_COM/IT1,DF_CARTA_IDENTITA,1.0).

## Istat
The *National Institute of Statistics* (Istat, Istituto Nazionale di Statistica) is an Italian national agency responsible for providing statistics to citizens and policy makers.

## *municipality_data*
municipality_data is a dataset of Italian municipalities, containing annual population and surface area figures as of January 1st for each year from 2001 to 2024, with additional census data from 1991. The dataset is the result of pivoting and translating the source data, so that each record corresponds to a municipality-year observation.

## Columns description

-*mun_name*: name of the municipality;

-*year*: year of the observations;

-*population*: population of the municipality in the corrisponding year;

-*surface*: surface of the municipality in the corresponding year.

**REMINDINGS**
1. If a municipality ceased to exist due to an administrative merger, its records are reported as missing starting from the year following the change. The absorbing municipality continues with updated data.
