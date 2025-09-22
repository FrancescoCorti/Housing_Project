**Data Source**: [Agenzia Entrate - OMI](https://www.agenziaentrate.gov.it/portale/it/web/guest/schede/fabbricatiterreni/omi/forniture-dati-omi)\
*User manual and metadata are published in ["Manuale dell'Osservatorio del Mercato Immobiliare"](https://www.agenziaentrate.gov.it/portale/documents/20143/265514/Manuale_OMI_2025_03_20.pdf/6ae6f18b-1632-a588-7990-bc94969f354a?t=1742814198800).*

# Housing-Project

## Agenzia delle Entrate - OMI
*Agenzia delle Entrate* is the Italian governmental agency responsible for managing taxation. *Osservatorio del Mercato Immobiliare* (Observatory for Real Estate Market, OMI) was created in 1999 as a branch of *Agenzia delle Entrate*. Its main goal is to provide transparent public information about the real estate market and support the Agency’s estimation activities.\
This project is based on a dataset created by the author, which merges OMI's semi-annual estimates of the real estate market.

**OMI's estimations**\
OMI’s estimate is not to be considered a precise estimate of the market value of a property, but an estimate interval that most probably bounds the real market value. Each estimate is not referred to a precise property, but to a group of them that share the following features:

-*municipality (comune)*: the minimal territorial administration unit in Italy; 

-*sector (fascia)*: territorial area within the municipality, based on an urban location. A municipality can be divided into the following sectors: *central (centrale)*, *semi-central (semicentrale)*, *peripheral (periferica)*, *suburban (suburbana)*, and *extra urban (extraurbana)*;

-*zone (zona)*: a subdivision of a sector, defined by real estate market homogeneity (micro level, based on socio-economic/valuation factors);

-*intended use (destinazione d’uso)*: aggregation of properties based on their intended use (residential, commercial, etc.);

-*building type*: type of building, as a specification of *intended use*;

-*condition*: general maintenance state of buildings.

Zones are defined by considering various socio-economic factors, such as the attractiveness of the municipality, public transportation, parking availability, prevalent building types, and homogeneity of building structures, among others. Then, depending on the real estate market volume in a specific area, properties are sampled:

-in areas with enough market dynamics, properties are sampled from different sources, such as sale deeds, lease contracts, real estate agencies, online real estate listings, building companies, and so on;

-in areas with limited market dynamics, properties are sampled from multiple similar zones, and their prices are aligned on spatial/temporal price trends based on territorial clusters.

To determine buying/leasing intervals, Student statistical function for probabilities distributions is applied. This result in a 95% confidence interval for the estimated average properties market values.  

(from now on, a group of properties that share the same features (the minimal unit of analysis as described above) is referred to as a *listing*)

**REMINDINGS**
1. Data are collected in areas with sufficient real estate market movements. In 2024, OMI analysed the residential market in 1,500 Italian municipalities, where two-thirds of the national stock is located and where 72% of buying/selling movements took place.
2. The estimation is the result of sampling procedures, which may yeld possible sampling biases. For this reason OMI constantly reviews its samples.
3. For a complete description of the data and of procedures used to collect data and to define zones, always refer to ["Manuale dell'Osservatorio del Mercato Immobiliare"](https://www.agenziaentrate.gov.it/portale/documents/20143/265514/Manuale_OMI_2025_03_20.pdf/6ae6f18b-1632-a588-7990-bc94969f354a?t=1742814198800).


## Description of variables in the source datasets (*translation (original name)*)

-*area (Area_territoriale)*: macro areas for the listing (*NW (north-west)*, *NE (north-east)*, *C (center)*, *I (islands)*, *S (south)*);

-*region (Regione)*: Region of the listing;

-*prov (Prov)*: Province of the listing;

-*mun_istat (Comune_ISTAT)*: ISTAT (National Institute of Statistic) code of the municipality;

-*mun_nat (Comune_CAT)*: national code of the municipality;

-*mun_cad (Comune_amm)*: cadastral code of the municipality;

-*mun_name (Comune_descrizione)*: name of the municipality;

-*sector (Fascia)*: subdivision of the municipality (*B (central)*, *C (semi-central)*, *D (peripheral)*, *E (suburban)*, * R (extra urban)*);

-*zone (Zona)*: subdivision of sectors on OMI's socio-economic paramenters (for instance, a semi-central area can be devided in C1, C2, etc.);

-*type (Descr_Tipologia)*: building type of the listing;

-*condition (Stato)*: general maintenance state of the listing (*normal*, *excellent*, *poor*);

-*buy_min (Compr_min)*: estimated minimum buying price;

-*buy_max (Compr_max)*: estimated maximum buying price;

-*lease_min (Loc_min)*: estimated minimum renting price;

-*lease_max (Loc_max)*: estimated maximum renting price.

*(three variables from the original datasets were removed because they were either empty or had only a single value)*
