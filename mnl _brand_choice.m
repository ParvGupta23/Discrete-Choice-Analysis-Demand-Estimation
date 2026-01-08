% PROJECT: Discrete Choice Analysis - Brand Preference Modeling
% MODEL:   Multinomial Logit (MNL) with Alternative-Specific Regressors
% METHOD:  Maximum Likelihood Estimation (MLE) via Quasi-Newton Optimization

clear; clc;

% --- 1. Data Ingestion ---
% Load consumer choice data (ID, Choice, Price, Features)
% Structure: Stacked format (3 rows per individual for 3 alternatives)
filename = 'data/cola_brand_choice.csv'; 
if ~exist(filename, 'file')
    error('Data file not found. Ensure "data/cola_brand_choice.csv" exists.'); 
end

T = readtable(filename);
T = sortrows(T, 'id'); % Ensure data is grouped by individual ID

J = 3; % Number of alternatives (Pepsi, 7-Up, Coke)
n = height(T)/J;

% Reshape data for vectorization: n individuals x J alternatives
price  = reshape(T.price, J, n).';  
choice = reshape(T.choice, J, n).'; 

fprintf('Data Loaded: %d individuals, %d alternatives.\n', n, J);

% --- 2. Model Estimation (MLE) ---
% Define Objective Function: Negative Log-Likelihood
nll = @(b) negll_mnl(b, price, choice);

% Optimization Options (Quasi-Newton / BFGS)
b0 = 0; % Initial guess
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off');

fprintf('Running Maximum Likelihood Estimation...\n');
[b_hat, fval, ~,~,~, H] = fminunc(nll, b0, options);

% Calculate Inference Statistics
se = sqrt(diag(inv(H))); % Standard Errors from Hessian
tstat = b_hat ./ se;     % t-statistics

% Display Estimation Results
fprintf('\n----------------------------------------\n');
fprintf('ESTIMATION RESULTS\n');
fprintf('----------------------------------------\n');
fprintf('Coefficient (Price):  %8.4f\n', b_hat);
fprintf('Standard Error:       %8.4f\n', se);
fprintf('t-Statistic:          %8.4f\n', tstat);
fprintf('Log-Likelihood:       %8.4f\n', -fval);

% --- 3. Elasticity & Scenario Analysis ---
% Scenario: Calculate Market Shares at specific price points
p_scenario = [1.00; 1.25; 1.10]; % Pepsi, 7-Up, Coke
V_scenario = b_hat * p_scenario;
P_scenario = exp(V_scenario) ./ sum(exp(V_scenario));

fprintf('\n----------------------------------------\n');
fprintf('SCENARIO FORECAST (Market Shares)\n');
fprintf('----------------------------------------\n');
fprintf('Pepsi ($1.00): %8.2f%%\n', P_scenario(1)*100);
fprintf('7-Up  ($1.25): %8.2f%%\n', P_scenario(2)*100);
fprintf('Coke  ($1.10): %8.2f%%\n', P_scenario(3)*100);

% --- 4. Marginal Effects Analysis ---
% Calculate the impact of a $0.10 price increase on Pepsi
delta_price = 0.10;
dP_own   = b_hat * P_scenario(1) * (1 - P_scenario(1)) * delta_price;
dP_cross_2 = -b_hat * P_scenario(1) * P_scenario(2) * delta_price;
dP_cross_3 = -b_hat * P_scenario(1) * P_scenario(3) * delta_price;

fprintf('\n----------------------------------------\n');
fprintf('MARGINAL EFFECTS (+$0.10 Pepsi Price)\n');
fprintf('----------------------------------------\n');
fprintf('Change in Pepsi Share: %8.4f\n', dP_own);
fprintf('Change in 7-Up Share:  +%8.4f\n', dP_cross_2);
fprintf('Change in Coke Share:  +%8.4f\n', dP_cross_3);

% --- Helper Functions ---
function f = negll_mnl(b, price, choice)
    % Compute Systematic Utility (V)
    V = b .* price; 
    
    % Log-Sum-Exp Trick for Numerical Stability
    m = max(V, [], 2);
    log_sum = m + log(sum(exp(V - m), 2));
    
    % Negative Log Likelihood
    f = -(sum(choice(:) .* V(:)) - sum(log_sum));
end
