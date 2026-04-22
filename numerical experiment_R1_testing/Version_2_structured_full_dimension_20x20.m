function Version_2_structured_full_dimension_20x20()
%==========================================================================
% 20x20 scalable Version 2
%
% Main changes relative to the small-grid public version:
%   (i) no full path enumeration;
%  (ii) monotone shortest-path oracle solved by dynamic programming;
% (iii) corridor is constructed explicitly (not brute-force over all squares);
%  (iv) Stage I is closer to paper Section 7.3:
%       centered OLS --> pseudo-costs --> learned low-dimensional basis.
%
% IMPORTANT:
%   This is intentionally a scalable approximation to the paper pipeline.
%   It is closer to Algorithm 4 than your current public Version 2, but it
%   still does NOT implement exact Algorithm 2 / exact cumulative SDS.
%==========================================================================

clc; close all;

% ----------------------------- Configuration -----------------------------
cfg.seed         = 202603131;
cfg.g            = 20;      % 20x20 node grid
cfg.p            = 50;       % context dimension
cfg.dstar_target = 8;       % target intrinsic decision-relevant dimension

cfg.trainSizes   = [80 160 320 640 960];
cfg.nTest        = 1000;
cfg.nTrial       = 5;

% Full-dimensional box prior
cfg.lowBase      = 10;
cfg.radCorr      = 1;       % corridor edges in [9,11]
cfg.highBase     = 100;
cfg.radOut       = 1;       % outside edges in [99,101]

% Contextual signal and noise
cfg.signalAmp    = 1.00;    % safe for normalized gadget basis
cfg.noiseCorr    = 0.05;
cfg.noiseOut     = 0.02;

% Stage I (contextual representation learning)
cfg.rankTol      = 5e-2;
cfg.maxRank      = 12;
cfg.ridge        = 1e-6;    % tiny OLS ridge regularization

% Stage II (SPO+)
cfg.numEpochs    = 12;
cfg.lrFull       = 0.010;
cfg.lrRed        = 0.030;
cfg.batchSize    = 64;
cfg.gradClip     = 5.0;

rng(cfg.seed, 'twister');

% ----------------------------- Build problem -----------------------------
edge = build_grid_edge_maps(cfg.g);
d = edge.d;
L = 2 * (cfg.g - 1);

[gadgetInfo, corridorEdges, Ustar] = build_diagonal_corridor(edge, cfg.dstar_target); %#ok<NASGU>
rstar = size(Ustar, 2);

lbC = (cfg.highBase - cfg.radOut) * ones(d,1);
ubC = (cfg.highBase + cfg.radOut) * ones(d,1);
lbC(corridorEdges) = cfg.lowBase - cfg.radCorr;
ubC(corridorEdges) = cfg.lowBase + cfg.radCorr;
cBase = 0.5 * (lbC + ubC);
affdimC = sum(ubC > lbC + 1e-12);

fprintf('=== Version 2, scalable 20x20 experiment ===\n');
fprintf('Grid size g = %d, #edges d = %d, path length L = %d\n', cfg.g, d, L);
fprintf('affdim(C) = %d (should equal d = %d)\n', affdimC, d);
fprintf('Constructed true decision-relevant dimension d_* = %d\n', rstar);
fprintf('Corridor edges = %d / %d\n', numel(corridorEdges), d);

verify_domination_sufficient_condition(cfg);

nN = numel(cfg.trainSizes);
riskFull = zeros(cfg.nTrial, nN);
riskRed  = zeros(cfg.nTrial, nN);
dimW     = zeros(cfg.nTrial, nN);

% -------------------------------- Trials ---------------------------------
for ii = 1:nN
    nTrain = cfg.trainSizes(ii);
    fprintf('\n--- nTrain = %d ---\n', nTrain);

    for tr = 1:cfg.nTrial
        rng(cfg.seed + 100*ii + tr, 'twister');

        Atrue = randn(rstar, cfg.p);

        [Xtr, Ctr] = sample_corridor_costs( ...
            nTrain, cfg.p, cBase, Ustar, Atrue, cfg.signalAmp, ...
            cfg.noiseCorr, cfg.noiseOut, lbC, ubC, corridorEdges);

        [Xte, Cte] = sample_corridor_costs( ...
            cfg.nTest, cfg.p, cBase, Ustar, Atrue, cfg.signalAmp, ...
            cfg.noiseCorr, cfg.noiseOut, lbC, ubC, corridorEdges);

        % ---------------- Stage I: contextual representation learning ----
        % Closer to paper Section 7.3:
        %   1) fit centered OLS for E[c | x] - cBase
        %   2) evaluate on fresh discovery contexts
        %   3) learn low-dimensional basis from pseudo-costs
        nReg = max(cfg.p + 5, floor(0.5 * nTrain));
        nDisc = max(cfg.p + 5, nTrain - nReg);

        Xreg = Xtr(1:nReg, :);
        Creg = Ctr(1:nReg, :);

        Ulearn = stage1_contextual_basis_ols( ...
            Xreg, Creg, nDisc, cfg.p, cBase, cfg.rankTol, cfg.maxRank, cfg.ridge);

        % ---------------- Stage II: full vs reduced SPO+ -----------------
        Bfull = train_spoplus_full_dp( ...
            Xtr, Ctr, edge, cBase, cfg.numEpochs, cfg.lrFull, ...
            cfg.batchSize, cfg.gradClip, lbC, ubC);

        Greduced = train_spoplus_reduced_dp( ...
            Xtr, Ctr, edge, cBase, Ulearn, cfg.numEpochs, cfg.lrRed, ...
            cfg.batchSize, cfg.gradClip, lbC, ubC);

        % ---------------- Evaluation -------------------------------------
        riskFull(tr,ii) = mean_spo_risk_full(Bfull, Xte, Cte, edge, cBase, lbC, ubC);
        riskRed(tr,ii)  = mean_spo_risk_reduced(Greduced, Ulearn, Xte, Cte, edge, cBase, lbC, ubC);
        dimW(tr,ii)     = size(Ulearn, 2);

        fprintf('trial %2d/%2d | dim(W)=%.0f | risk_full=%.4g | risk_red=%.4g\n', ...
            tr, cfg.nTrial, dimW(tr,ii), riskFull(tr,ii), riskRed(tr,ii));
    end
end

% ----------------------------- Summaries ---------------------------------
xAxis = cfg.trainSizes;
[mF, ciF] = mean_ci90(log10(riskFull + 1e-12));
[mR, ciR] = mean_ci90(log10(riskRed  + 1e-12));
mD = mean(dimW, 1, 'omitnan');

paramFull = d * (cfg.p + 1);
paramRed  = mean(dimW(:,end), 'all') * (cfg.p + 1);

fprintf('\n=== Compression summary ===\n');
fprintf('Full linear model params      = d*(p+1) = %d\n', paramFull);
fprintf('Mean final learned dim(W)     = %.2f\n', mean(dimW(:,end), 'all'));
fprintf('Approx reduced model params   = %.2f\n', paramRed);

% -------------------------------- Plots ----------------------------------
figRisk = figure('Name', 'Version 2 (20x20): Test SPO risk');
hold on; grid on; box on;
errorbar(xAxis, mF, ciF, 'LineWidth', 1.2);
errorbar(xAxis, mR, ciR, 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('log10(Test SPO risk)');
legend({'Full SPO+', 'Reduced SPO+ after learned subspace'}, 'Location', 'best');
title(sprintf('Version 2, g=%d, d=%d, true d_*=%d, affdim(C)=d', cfg.g, d, rstar));

figDim = figure('Name', 'Version 2 (20x20): learned dimension');
hold on; grid on; box on;
plot([0, xAxis], [0, mD], '-o', 'LineWidth', 1.6, 'MarkerSize', 5);
yline(rstar, '--', 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('dim(W) (mean over trials)');
legend({'Mean learned dim(W)', 'True d_*'}, 'Location', 'best');
title('Stage I contextual representation learning');

resultsDir = prepare_results_dir();
save_figure(figRisk, fullfile(resultsDir, 'full_dim_20x20_spo_risk.png'));
save_figure(figDim,  fullfile(resultsDir, 'full_dim_20x20_dimW.png'));
save(fullfile(resultsDir, 'full_dim_20x20_summary.mat'), ...
    'cfg', 'riskFull', 'riskRed', 'dimW', 'mF', 'ciF', 'mR', 'ciR', 'mD', ...
    'rstar', 'corridorEdges', 'paramFull', 'paramRed', 'affdimC', 'cBase');

fprintf('Saved results to %s\n', resultsDir);
end

%==========================================================================
% Stage I: contextual representation learning
%==========================================================================

function U = stage1_contextual_basis_ols(Xreg, Creg, nDisc, p, cBase, tolRel, maxRank, ridge)
% Paper-inspired scalable Stage I:
%   centered OLS for mu(x)-cBase, then pseudo-costs on fresh contexts,
%   then low-rank basis extraction by SVD.

d = size(Creg, 2);
if size(Xreg,1) < p
    U = zeros(d, 0);
    return;
end

% centered regression target
Yreg = Creg - cBase.';

% solve Ahat in (c - cBase) ≈ Ahat * x
XtX = Xreg.' * Xreg + ridge * eye(p);
Ahat = (Yreg.' * Xreg) / XtX;      % d x p

% fresh discovery contexts, as in the pseudo-cost step
Xdisc = randn(nDisc, p);
YhatDisc = Xdisc * Ahat.';         % nDisc x d

U = learn_basis_from_centered_samples(YhatDisc, tolRel, maxRank);
end

%==========================================================================
% Data generation
%==========================================================================

function [X, C] = sample_corridor_costs(n, p, cBase, Ustar, Atrue, signalAmp, noiseCorr, noiseOut, lbC, ubC, corridorEdges)
% Full-dimensional box prior:
%   every edge varies in an interval => affdim(C)=d
% Decision-relevant contextual signal:
%   only injected along the true low-dimensional corridor-switch basis Ustar

d = numel(cBase);
X = randn(n, p);

Lat = tanh((Atrue * X.') / sqrt(p));    % r x n
Signal = signalAmp * (Ustar * Lat);     % d x n

Noise = noiseOut * (2*rand(d, n) - 1);
Noise(corridorEdges, :) = noiseCorr * (2*rand(numel(corridorEdges), n) - 1);

Cmat = repmat(cBase, 1, n) + Signal + Noise;
Cmat = min(max(Cmat, lbC), ubC);

C = Cmat.';
end

%==========================================================================
% Stage II: basis learning helper
%==========================================================================

function U = learn_basis_from_centered_samples(M, tolRel, maxRank)
if isempty(M)
    U = zeros(size(M,2), 0);
    return;
end

[Utmp, S, ~] = svd(M.', 'econ');
s = diag(S);

if isempty(s) || s(1) < 1e-12
    U = zeros(size(M,2), 0);
    return;
end

r = sum((s >= tolRel * s(1)) & (s > 1e-10));
r = min(r, maxRank);

if r == 0
    U = zeros(size(M,2), 0);
else
    U = Utmp(:, 1:r);
end
end

%==========================================================================
% SPO+ training
%==========================================================================

function B = train_spoplus_full_dp(X, C, edge, cRef, numEpochs, lr0, batchSize, gradClip, lbPred, ubPred)
[n, p] = size(X);
d = edge.d;
B = zeros(d, p+1);

for ep = 1:numEpochs
    eta = lr0 / sqrt(ep);
    perm = randperm(n);

    for startIdx = 1:batchSize:n
        ids = perm(startIdx:min(startIdx + batchSize - 1, n));
        Grad = zeros(d, p+1);

        for kk = 1:numel(ids)
            i = ids(kk);
            phi = [X(i,:)'; 1];
            c = C(i,:)';

            chat = cRef + B * phi;
            chat = min(max(chat, lbPred), ubPred);

            subg = spoplus_subgrad_dp(chat, c, edge);
            Grad = Grad + full(subg) * phi.';
        end

        Grad = Grad / numel(ids);
        gnorm = norm(Grad(:), 2);
        if gnorm > gradClip
            Grad = Grad * (gradClip / gnorm);
        end

        B = B - eta * Grad;
    end
end
end

function G = train_spoplus_reduced_dp(X, C, edge, cRef, U, numEpochs, lr0, batchSize, gradClip, lbPred, ubPred)
[n, p] = size(X);
r = size(U, 2);

if r == 0
    G = zeros(0, p+1);
    return;
end

G = zeros(r, p+1);

for ep = 1:numEpochs
    eta = lr0 / sqrt(ep);
    perm = randperm(n);

    for startIdx = 1:batchSize:n
        ids = perm(startIdx:min(startIdx + batchSize - 1, n));
        Grad = zeros(r, p+1);

        for kk = 1:numel(ids)
            i = ids(kk);
            phi = [X(i,:)'; 1];
            c = C(i,:)';

            chat = cRef + U * (G * phi);
            chat = min(max(chat, lbPred), ubPred);

            subg = spoplus_subgrad_dp(chat, c, edge);
            Grad = Grad + (U.' * full(subg)) * phi.';
        end

        Grad = Grad / numel(ids);
        gnorm = norm(Grad(:), 2);
        if gnorm > gradClip
            Grad = Grad * (gradClip / gnorm);
        end

        G = G - eta * Grad;
    end
end
end

%==========================================================================
% Evaluation
%==========================================================================

function risk = mean_spo_risk_full(B, X, C, edge, cRef, lbPred, ubPred)
n = size(X,1);
tot = 0;

for i = 1:n
    phi = [X(i,:)'; 1];
    c = C(i,:)';

    chat = cRef + B * phi;
    chat = min(max(chat, lbPred), ubPred);

    tot = tot + spo_loss_dp(chat, c, edge);
end

risk = tot / n;
end

function risk = mean_spo_risk_reduced(G, U, X, C, edge, cRef, lbPred, ubPred)
n = size(X,1);
tot = 0;

for i = 1:n
    phi = [X(i,:)'; 1];
    c = C(i,:)';

    if isempty(U)
        chat = cRef;
    else
        chat = cRef + U * (G * phi);
    end
    chat = min(max(chat, lbPred), ubPred);

    tot = tot + spo_loss_dp(chat, c, edge);
end

risk = tot / n;
end

%==========================================================================
% SPO / SPO+ primitives
%==========================================================================

function loss = spo_loss_dp(chat, ctrue, edge)
[~, wHat] = oracle_monotone_path_dp(chat, edge);
[~, wOpt] = oracle_monotone_path_dp(ctrue, edge);
loss = ctrue.' * (wHat - wOpt);
end

function subg = spoplus_subgrad_dp(chat, ctrue, edge)
[~, w0] = oracle_monotone_path_dp(ctrue, edge);
[~, w1] = oracle_monotone_path_dp(2*chat - ctrue, edge);
subg = 2 * (w0 - w1);
end

%==========================================================================
% Monotone shortest-path oracle by DP
%==========================================================================

function [bestCost, w] = oracle_monotone_path_dp(c, edge)
g = edge.g;
h = edge.h;
v = edge.v;

D = inf(g, g);
parent = zeros(g, g, 'uint8');  % 1 = from left, 2 = from up
D(1,1) = 0;

for i = 1:g
    for j = 1:g
        cur = D(i,j);
        if isinf(cur)
            continue;
        end

        if j < g
            cand = cur + c(h(i,j));
            if cand < D(i, j+1)
                D(i, j+1) = cand;
                parent(i, j+1) = 1;
            end
        end

        if i < g
            cand = cur + c(v(i,j));
            if cand < D(i+1, j)
                D(i+1, j) = cand;
                parent(i+1, j) = 2;
            end
        end
    end
end

bestCost = D(g,g);
w = sparse(edge.d, 1);

i = g;
j = g;
while (i > 1) || (j > 1)
    if parent(i,j) == 1
        e = h(i, j-1);
        w(e) = 1;
        j = j - 1;
    elseif parent(i,j) == 2
        e = v(i-1, j);
        w(e) = 1;
        i = i - 1;
    else
        error('oracle_monotone_path_dp: failed to reconstruct path.');
    end
end
end

%==========================================================================
% Corridor construction
%==========================================================================

function [info, corridorEdges, Ustar] = build_diagonal_corridor(edge, m)
% Construct a narrow corridor with exactly m disjoint local switch gadgets.
% All path differences among corridor paths live in span{q_1,...,q_m}.

g = edge.g;
maxPossible = floor((g - 1) / 2);
if m > maxPossible
    error('Requested m=%d gadgets, but at most %d fit on a %dx%d grid.', m, maxPossible, g, g);
end

mask = false(edge.d, 1);
Q = zeros(edge.d, m);
squareTL = zeros(m, 2);

curR = 1;
curC = 1;

for k = 1:m
    i = 2*k - 1;
    j = 2*k - 1;
    squareTL(k,:) = [i, j];

    % fixed connector from current node to gadget start
    conn = connector_edges(curR, curC, i, j, edge.h, edge.v);
    mask(conn) = true;

    % square boundary: both RD and DR local alternatives
    sqEdges = [
        edge.h(i,   j);
        edge.h(i+1, j);
        edge.v(i,   j);
        edge.v(i, j+1)
    ];
    mask(sqEdges) = true;

    % local decision-relevant path-difference direction
    q = zeros(edge.d, 1);
    q(edge.h(i,   j)) =  1;
    q(edge.v(i, j+1)) =  1;
    q(edge.v(i,   j)) = -1;
    q(edge.h(i+1, j)) = -1;
    Q(:,k) = q / norm(q);

    curR = i + 1;
    curC = j + 1;
end

% final connector from last gadget to sink
conn = connector_edges(curR, curC, g, g, edge.h, edge.v);
mask(conn) = true;

corridorEdges = find(mask);
Ustar = Q;

info.squareTL = squareTL;
info.numGadgets = m;
end

function E = connector_edges(r0, c0, r1, c1, h, v)
% Deterministic monotone connector: R, D, R, D, ...
if r1 < r0 || c1 < c0
    error('connector_edges: end point must dominate start point.');
end

E = zeros(0,1);
r = r0;
c = c0;

while (r < r1) || (c < c1)
    if c < c1
        E(end+1,1) = h(r,c); %#ok<AGROW>
        c = c + 1;
    end
    if r < r1
        E(end+1,1) = v(r,c); %#ok<AGROW>
        r = r + 1;
    end
end
end

function verify_domination_sufficient_condition(cfg)
L = 2 * (cfg.g - 1);

% worst corridor path cost
maxCorr = L * (cfg.lowBase + cfg.radCorr);

% any non-corridor path must use at least one outside edge
minOneOutside = (cfg.highBase - cfg.radOut) + (L - 1) * (cfg.lowBase - cfg.radCorr);

fprintf('Domination sufficient condition:\n');
fprintf('  max corridor-path cost <= %.2f\n', maxCorr);
fprintf('  min path cost with one outside edge >= %.2f\n', minOneOutside);

if minOneOutside <= maxCorr
    warning('Outside-edge penalty is not large enough. Increase highBase.');
else
    fprintf('  OK: any path using an outside edge is always worse than a pure corridor path.\n');
end
end

%==========================================================================
% Geometry: edge maps
%==========================================================================

function edge = build_grid_edge_maps(g)
h = zeros(g, g-1);
idx = 1;
for i = 1:g
    for j = 1:(g-1)
        h(i,j) = idx;
        idx = idx + 1;
    end
end

v = zeros(g-1, g);
for i = 1:(g-1)
    for j = 1:g
        v(i,j) = idx;
        idx = idx + 1;
    end
end

edge.h = h;
edge.v = v;
edge.g = g;
edge.d = idx - 1;
end

%==========================================================================
% Helpers
%==========================================================================

function [m, ci] = mean_ci90(M)
z = 1.645;
m = mean(M, 1, 'omitnan');
nEff = sum(~isnan(M), 1);
sd = std(M, 0, 1, 'omitnan');
se = sd ./ max(sqrt(nEff), 1);
ci = z * se;
end

function resultsDir = prepare_results_dir()
repoDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(repoDir, 'results');
if exist(resultsDir, 'dir') ~= 7
    mkdir(resultsDir);
end
end

function save_figure(figHandle, filePath)
if exist('exportgraphics', 'file') == 2
    exportgraphics(figHandle, filePath, 'Resolution', 300);
else
    saveas(figHandle, filePath);
end
end