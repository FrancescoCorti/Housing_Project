1. Log-level model
If your model is:
y=β⋅log⁡(x)+…y = \beta \cdot \log(x) + \dotsy=β⋅log(x)+…


yyy is in levels (original units), xxx is log-transformed.


Then β\betaβ approximately measures the change in yyy for a 1% change in xxx:


Δy≈β⋅(%Δx)\Delta y \approx \beta \cdot (\%\Delta x)Δy≈β⋅(%Δx)
So yes, if β=1.2\beta = 1.2β=1.2:

A 1% increase in xxx is associated with an increase of 1.2 units in yyy (not percent!).


2. Log-log model (both y and x logged)
If your model is:
log⁡(y)=β⋅log⁡(x)+…\log(y) = \beta \cdot \log(x) + \dotslog(y)=β⋅log(x)+…


Both xxx and yyy are log-transformed.


Then β\betaβ is a true elasticity:


%Δy=β⋅%Δx\%\Delta y = \beta \cdot \%\Delta x%Δy=β⋅%Δx
So if β=1.2\beta = 1.2β=1.2:

A 1% increase in xxx is associated with a 1.2% increase in yyy.

This is the case most often used in tourism analysis with skewed counts.

3. What about centering?
Centering xxx (or log⁡(x)\log(x)log(x)) does not change the slope coefficient — it only changes the intercept:


β\betaβ still has the same interpretation as above.


The intercept now corresponds to the expected yyy at average x, not at x=0.



✅ Summary Table
Model typeCoefficient interpretationExample if β=1.2log-levelΔy per 1% change in x1% ↑ x → +1.2 units ylog-log%Δy per 1% change in x1% ↑ x → +1.2% ycentered log-xsame as aboveβ unchanged; intercept changes

So to answer your question:


If you log-transformed x but y is in levels, β = 1.2 means y increases by 1.2 units for a 1% increase in x.


If both x and y are logged, β = 1.2 means y increases by 1.2% for a 1% increase in x — a true elasticity.



If you want, I can give a quick concrete numerical example with arrivals and population to show exactly how to interpret a coefficient like 1.2 in practice.

A log–level coefficient has a semi-elasticity interpretation:

A 1-unit increase in X changes ln(Y) by β,
which means it changes Y by approximately 100·β percent.




Variable	Log?	Reason
Housing price (dependent)	✔	Gives % interpretation
Population	✔	Elasticity meaningful
Median income	✔	Elasticity meaningful
Real GDP	✔	Standard macro elasticity interpretation
Immigration/emigration counts or levels	✔ (if positive)	Elasticity meaningful
Tourism indices (ratios)	Depends	Log if strictly positive and vary a lot
Unemployment rate (%)	✘	Semi-elasticity preferred
CPI (%) or inflation rate	✘	Can be zero; level changes easier to interpret
Government interest rate (%)	✘	Level or basis-point changes meaningful
Birth rate (per 1,000)	Usually ✘	Small numbers, odd elasticity
Life expectancy (years)	Usually ✘	Elasticity rarely interpretable
GINI index	✘	Not a scale suited for logs


A log–level coefficient has a semi-elasticity interpretation:

A 1-unit increase in X changes ln(Y) by β,
which means it changes Y by approximately 100·β percent.

example with unemployment (measured as percentage)

if unemployment goes from 8 to 7 % -> 1-unit change (1-percentage-point increase in unemployment) NOT A 1% CHANGE

then for a 1% point increase in unemployment, Y increase by 100*beta% change in Y