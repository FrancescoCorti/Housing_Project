## tourist_data - METADATA
**Data Source**: [IstatData](https://esploradati.istat.it/databrowser/#/it/dw/categories/IT1,Z0700SER,1.0/SER_TOURISM/SER_TOURISM_RELATED_FILES)\
*Original metadata published on [Istat website](https://www.istat.it/classificazione/classificazione-dei-comuni-in-base-alla-densita-turistica/)*

### Istat
The *National Institute of Statistics* (Istat, Istituto Nazionale di Statistica) is an Italian national agency responsible for providing statistics to citizens and policy makers.

### *tourist_data*
*tourist_data* is the result of cleaning and translating the source data. This was collected in 2020 to inform policy decisions that support local tourism following the COVID-19 pandemic. 
The purpose of this research was to propose a touristic classification of municipalities based on their geographical position and on anthropic and cultural factors such as:

-proximity to a lake (a municipality directly on the coast of a lake or with at least 50% of its area located at a maximum of 3km from a lake);

-proximity to the sea (a municipality directly on the coast of the sea or with at least 50% of its area located at a maximum of 10km from the sea);

-proximity to the mountains (a municipality 600m MASL);

-cultural relevance (municipality certified as culturally relevant);

-thermal (municipality defined as thermal);

-big city (municipality with a population of more than 250,000 in 2019);

-without specific touristic vocation (municipality with a touristic flow but not belonging to any of the previous categories);

-not touristic (municipality without touristic flow).

Also, the influence of tourism in municipalities was measured by accounting for quantitative information on tourism. In particular, composite indices were structured from various 
standardised source data. The result is three variables that measure (on a quintile scale) the presence of touristic infrastructures, the presence of tourist flows, and the impact of 
tourism on the local economy and occupation. In the source data, a fourth variable was created to provide a summary score for each municipality. 

For statistical modelling purposes, categorical variables related to the geographical and cultural characteristics of the locations were encoded as dummy variables. Quantitative measures 
of tourism were encoded on a scale from 1 to 5. 

**REMINDINGS**
1. For a complete description of the data and of procedures used to collect data and to define zones, always refer to the source [metadata](https://www.istat.it/classificazione/classificazione-dei-comuni-in-base-alla-densita-turistica/).

### Columns description

-*region*: region of the municipality;

-*prov_code*: provincial code of the location;

-*mun_name*: name of the municipality;

-*cat_code*: code to identify the main touristic vocation of the municipality (for correspondences, see source metadata);

-*cultural*: dummy variable to identify whether a municipality has a cultural vocation (1) or not (0);

-*lake*: dummy variable to identify whether a municipality is a lake touristic location (1) or not (0);

-*mountain*: dummy variable to identify whether a municipality is a mountain touristic location (1) or not (0);

-*sea*: dummy variable to identify whether a municipality is a sea touristic location (1) or not (0);

-*thermal*: dummy variable to identify whether a municipality is a thermal touristic location (1) or not (0);

-*cat_list*: list of touristic vocations of a municipality (cultural, lake, mountain, sea, thermal);

-*infrastructures*: presence of touristic infrastructures (1-5; if the municipality is classified as not touristic, the value is 0)

-*flows*: presence of touristic flows (see above for encoding);

-*economy*: impact of tourism on local economies (see above for encoding);

-*summary*: summary score of *infrastructures*, *flows*, and *economy*.


