# Scripts-for-gas-adsorption-isotherms-model-fitting-IAST-prediction-and-multivariate-analysis
#!/usr/bin/env python3
------------------------------------------------------------------------------------------------------------
1. Fit adsorption-isotherm data (Langmuir & Sips) at three temperatures.
------------------------------------------------------------------------------------------------------------
• Needs NumPy ≥1.26, SciPy ≥1.12, Matplotlib ≥3.8.
• Data files:  data.dat  data2.dat  data3.dat
   Each with two columns:  P(bar)   V(mmol g⁻¹)

Run:
fit_isotherms_new.py

------------------------------------------------------------------------
2. IAST selectivity CO2/N2 vs pressure
------------------------------------------------------------------------
• Needs:  numpy  pyiast  (pip install pyiast)
• Fill in your own isotherm parameters and feed composition.

Run:
  pyIAST.py
  
------------------------------------------------------------------------------------------------------------
3. Multivariate analysis for CO2 adsorption — embedded dataset (no file I/O)
------------------------------------------------------------------------------------------------------------
- Arial font + LaTeX-style mathtext labels (subscripts/superscripts)
- Correlation heatmap (red–blue)
- PCA (X only)
- PLS2 (X -> multiple Y) with CV, VIP, standardized coefficients, Pred vs True

Run:
  python co2_multivariate_analysis.py
Outputs are written to ./outputs
