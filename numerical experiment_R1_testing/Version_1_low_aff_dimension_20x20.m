function Version_1_low_aff_dimension_20x20()
%==========================================================================
% 20x20 scalable Version 1
%
% Main changes relative to the small-grid public version:
%   (i) no enumeration of all monotone paths;
%  (ii) shortest-path oracle is solved by dynamic programming on the DAG;
% (iii) Stage I uses a scalable low-rank basis learner from observed costs.
%
% Interpretation:
%   affdim(C) = d_star, because c = c0 + U_star * s and all variation lives
%   in the learned low-dimensional subspace.
%==========================================================================

clc; close all;

% ----------------------------- Configuration -----------------------------
cfg.seed       = 20260313;
cfg.g          = 20;     % 20x20 node grid
cfg.p          = 50;      % context dimension
cfg.dstar_true = 8;      % true low affine dimension

cfg.trainSizes = [40 80 160 320 640];
cfg.nTest      = 1000;
cfg.nTrial     = 5;

cfg.c0         = 10.0;   % baseline edge cost
cfg.signalAmp  = 1.20;   % latent signal amplitude
cfg.latentNoise = 0.05;  % noise in latent coordinates only

cfg.numEpochs  = 12;
cfg.lrFull     = 0.020;
cfg.lrRed      = 0.040;
cfg.batchSize  = 64;
cfg.gradClip   = 5.0;

cfg.rankTol    = 1e-2;   % relative singular-value threshold
cfg.maxRank    = 12;
cfg.predClip   = 2.0;    % clip predictions to [c0-predClip, c0+predClip]

rng(cfg.seed, 'twister');

% ----------------------------- Build problem -----------------------------
edge = build_grid_edge_maps(cfg.g);
d = edge.d;
L = 2 * (cfg.g - 1);

fprintf('=== Version 1, scalable 20x20 experiment ===\n');
fprintf('Grid size g = %d, #edges d = %d, path length L = %d\n', cfg.g, d, L);

% True low-dimensional basis: disjoint local 2x2 switch gadgets
[Ustar, gadgetInfo] = build_diagonal_switch_basis(edge, cfg.dstar_true); %#ok<NASGU>
rstar = size(Ustar, 2);

fprintf('True affine dimension d_* = %d\n', rstar);

cRef = cfg.c0 * ones(d,1);
lbPred = (cfg.c0 - cfg.predClip) * ones(d,1);
ubPred = (cfg.c0 + cfg.predClip) * ones(d,1);

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

        % fresh contextual map each trial
        Atrue = randn(rstar, cfg.p);

        [Xtr, Ctr] = sample_low_affine_costs( ...
            nTrain, cfg.p, cRef, Ustar, Atrue, cfg.signalAmp, cfg.latentNoise);
        [Xte, Cte] = sample_low_affine_costs( ...
            cfg.nTest, cfg.p, cRef, Ustar, Atrue, cfg.signalAmp, cfg.latentNoise);

        % ---------------- Stage I: scalable representation learning -------
        centeredTrain = Ctr - cRef.';
        Ulearn = learn_basis_from_centered_samples(centeredTrain, cfg.rankTol, cfg.maxRank);

        % ---------------- Stage II: SPO+ training -------------------------
        Bfull = train_spoplus_full_dp( ...
            Xtr, Ctr, edge, cRef, cfg.numEpochs, cfg.lrFull, ...
            cfg.batchSize, cfg.gradClip, lbPred, ubPred);

        Greduced = train_spoplus_reduced_dp( ...
            Xtr, Ctr, edge, cRef, Ulearn, cfg.numEpochs, cfg.lrRed, ...
            cfg.batchSize, cfg.gradClip, lbPred, ubPred);

        % ---------------- Evaluation -------------------------------------
        riskFull(tr,ii) = mean_spo_risk_full(Bfull, Xte, Cte, edge, cRef, lbPred, ubPred);
        riskRed(tr,ii)  = mean_spo_risk_reduced(Greduced, Ulearn, Xte, Cte, edge, cRef, lbPred, ubPred);
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
figRisk = figure('Name', 'Version 1 (20x20): Test SPO risk');
hold on; grid on; box on;
errorbar(xAxis, mF, ciF, 'LineWidth', 1.2);
errorbar(xAxis, mR, ciR, 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('log10(Test SPO risk)');
legend({'Full SPO+', 'Reduced SPO+ after learned subspace'}, 'Location', 'best');
title(sprintf('Version 1, g=%d, d=%d, true d_*=%d', cfg.g, d, rstar));

figDim = figure('Name', 'Version 1 (20x20): learned dimension');
hold on; grid on; box on;
plot([0, xAxis], [0, mD], '-o', 'LineWidth', 1.6, 'MarkerSize', 5);
yline(rstar, '--', 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('dim(W) (mean over trials)');
legend({'Mean learned dim(W)', 'True d_*'}, 'Location', 'best');
title('Stage I representation learning');

resultsDir = prepare_results_dir();
save_figure(figRisk, fullfile(resultsDir, 'low_affdim_20x20_spo_risk.png'));
save_figure(figDim,  fullfile(resultsDir, 'low_affdim_20x20_dimW.png'));
save(fullfile(resultsDir, 'low_affdim_20x20_summary.mat'), ...
    'cfg', 'riskFull', 'riskRed', 'dimW', 'mF', 'ciF', 'mR', 'ciR', 'mD', ...
    'rstar', 'paramFull', 'paramRed');

fprintf('Saved results to %s\n', resultsDir);
end

%==========================================================================
% Data generation
%==========================================================================

function [X, C] = sample_low_affine_costs(n, p, cRef, Ustar, Atrue, signalAmp, latentNoise)
% c = cRef + signalAmp * Ustar * s(x), with all variation restricted to span(Ustar)

d = numel(cRef);
r = size(Ustar, 2);

X = randn(n, p);
G = tanh((Atrue * X') / sqrt(p));       % r x n
G = G + latentNoise * randn(r, n);      % latent-only noise
Cmat = repmat(cRef, 1, n) + signalAmp * (Ustar * G);  % d x n

C = Cmat.'; % n x d
if size(C,2) ~= d
    error('sample_low_affine_costs: dimension mismatch.');
end
end

%==========================================================================
% Stage I: basis learning from centered samples
%==========================================================================

function U = learn_basis_from_centered_samples(M, tolRel, maxRank)
% M is n x d, already centered relative to cRef.
% We learn a low-rank basis via SVD.

[dummyN, d] = size(M); %#ok<NASGU>
if isempty(M)
    U = zeros(d, 0);
    return;
end

[Utmp, S, ~] = svd(M.', 'econ');   % d x n
s = diag(S);

if isempty(s) || s(1) < 1e-12
    U = zeros(d, 0);
    return;
end

r = sum((s >= tolRel * s(1)) & (s > 1e-10));
r = min(r, maxRank);

if r == 0
    U = zeros(d, 0);
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
% Geometry: edge maps and true low-dimensional basis
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

function [Ustar, info] = build_diagonal_switch_basis(edge, m)
% Build m disjoint 2x2 switch gadgets along the diagonal.
% Each gadget contributes one local path-difference direction.

g = edge.g;
maxPossible = floor((g - 1) / 2);
if m > maxPossible
    error('Requested m=%d gadgets, but at most %d fit on a %dx%d grid.', m, maxPossible, g, g);
end

Q = zeros(edge.d, m);
squareTL = zeros(m, 2);

for k = 1:m
    i = 2*k - 1;
    j = 2*k - 1;

    squareTL(k,:) = [i, j];

    q = zeros(edge.d, 1);
    q(edge.h(i,   j)) =  1;
    q(edge.v(i, j+1)) =  1;
    q(edge.v(i,   j)) = -1;
    q(edge.h(i+1, j)) = -1;

    Q(:,k) = q / norm(q);
end

Ustar = Q;
info.squareTL = squareTL;
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