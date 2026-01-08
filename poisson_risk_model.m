% PROJECT: Regional Risk Analysis - Count Data Modeling
% MODEL:   Poisson Regression with Log Link Function
% OBJECTIVE: Analyze incident rates and forecast risk probabilities

clear; clc;

% --- 1. Data Ingestion ---
filename = 'data/fmd_incidents.csv';
if ~exist(filename, 'file')
    error('Data file not found. Ensure "data/fmd_incidents.csv" exists.'); 
end
T = readtable(filename);

% Variable Mapping (Robust selection)
% Assumes columns: 'FMD1998', 'Cattle', 'EasternTurkey'
F = T.FMD1998;
C = T.Cattle;
E = T.EasternTurkey; % 0 = West, 1 = East

% Filter NaNs
ok = ~isnan(F) & ~isnan(C) & ~isnan(E);
T = T(ok, :);

% --- 2. Descriptive Analytics ---
fprintf('----------------------------------------\n');
fprintf('DESCRIPTIVE STATISTICS\n');
fprintf('----------------------------------------\n');
grp_stats = grpstats(T, 'EasternTurkey', {'mean', 'std', 'max'}, 'DataVars', 'FMD1998');
disp(grp_stats);

% --- 3. GLM Estimation ---
% Model: log(Incidents) = b0 + b1*log(Cattle) + b2*Region
fprintf('\nFitting Poisson GLM...\n');
mdl = fitglm(T, 'FMD1998 ~ log(Cattle) + EasternTurkey', ...
             'Distribution', 'poisson', 'Link', 'log');
disp(mdl);

% Extract Coefficients
b = mdl.Coefficients.Estimate;
elasticity_cattle = b(2);
irr_region = exp(b(3));

fprintf('Key Insights:\n');
fprintf('> Cattle Elasticity: %.4f (1%% increase in cattle -> %.4f%% increase in incidents)\n', ...
        elasticity_cattle, elasticity_cattle);
fprintf('> Region IRR:        %.4f (Eastern region has %.1f%% higher risk, ceteris paribus)\n', ...
        irr_region, (irr_region-1)*100);

% --- 4. Predictive Modeling ---
% Forecast expected incidents for representative provinces
scenarios = table([170080; 157850], [1; 0], ...
    'VariableNames', {'Cattle', 'EasternTurkey'});

[lambda_pred, ~] = predict(mdl, scenarios);

fprintf('\n----------------------------------------\n');
fprintf('RISK PROBABILITY FORECAST\n');
fprintf('----------------------------------------\n');
% Calculate probability of <= 2 incidents (Safety Threshold)
p_safe = poisscdf(2, lambda_pred);

fprintf('Scenario 1 (East, High Density): Lambda=%.2f | P(<=2 Incidents)=%.2f%%\n', ...
        lambda_pred(1), p_safe(1)*100);
fprintf('Scenario 2 (West, Avg Density):  Lambda=%.2f | P(<=2 Incidents)=%.2f%%\n', ...
        lambda_pred(2), p_safe(2)*100);
