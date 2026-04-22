function Version_3p2c_structured_full_dimension_20x20_break_implicit_compression()
%==========================================================================
% 20x20 Version 3.2c: break implicit compression for full SPO,
% and make SLO harder in a decision-visible way.
%
% Motivation:
%   In Version 3.2 / 3.2b, the centered full SPO+ model often gets
%   implicitly compressed into W* by the gradient dynamics: starting from
%   zero, its SPO+ subgradients stay inside W* for the planted corridor
%   family. Also, the SLO nuisance in 3.2b lived strictly in W*^perp, which
%   made it prediction-hard but almost decision-invisible.
%
% Main changes here:
%   (1) Reduce the corridor/outside margin while preserving domination.
%       This makes path decisions more sensitive.
%   (2) Initialize the full centered-affine SPO+ model with a small ambient
%       random matrix, so it no longer starts exactly inside the clean W*
%       dynamics. Reduced SPO+ still starts from zero.
%   (3) Replace the pure W*^perp nuisance with a mixed nuisance mean map:
%         - a large predictive component in W*^perp,
%         - plus a controlled leakage term inside W*.
%       This keeps the true planted W* = span(Ustar), but makes finite-
%       sample SLO decisions more sensitive to whether the learner chases
%       the ambient nuisance.
%
% Stage I remains:
%   centered OLS -> pseudo-costs -> cumulative warm-start learner.
%
% Stage II compares the same families as in 3.2b:
%   SPO+:  full / reduced learned-W / reduced oracle-W*
%   SLO:   full / reduced learned-W / reduced oracle-W*
%==========================================================================

clc; close all;

% ----------------------------- Configuration -----------------------------
cfg.seed         = 202603131;
cfg.g            = 20;
cfg.p            = 120;
cfg.dstar_target = 8;

cfg.trainSizes   = [160 200 240];
cfg.nTest        = 1000;
cfg.nTrial       = 5;

% Smaller scale, smaller domination margin, but still pure corridor optima
cfg.lowBase      = 1.0;
cfg.radCorr      = 0.10;    % corridor edges in [0.9, 1.1]
cfg.highBase     = 10.0;
cfg.radOut       = 0.10;    % outside edges in [9.9, 10.1]

% Well-specified relevant mean component inside W*
cfg.signalAmp        = 1.0;
cfg.relLatentBound   = 0.12;

% Zero-mean bounded noise
cfg.noiseCorr        = 0.012;
cfg.noiseOut         = 0.008;

% Stage I (regression + pseudo-cost discovery)
cfg.ridge            = 1e-6;
cfg.fiTol            = 1e-9;
cfg.indepTol         = 1e-8;
cfg.regFrac          = 0.50;
cfg.maxPointwiseIters = cfg.dstar_target + 2;

% Stage II (SPO+)
cfg.numEpochsStage2  = 18;
cfg.lrStage2         = 0.010;
cfg.batchSize        = 64;
cfg.gradClip         = 5.0;
cfg.fullSpoInitStd   = 0.90;   % random ambient init to break implicit compression

% SLO ridge
cfg.sloRidge         = 1e-8;

% Mixed nuisance for harder SLO
cfg.nuisRank         = 80;
cfg.nuisPerpEdgeTarget = 0.025;   % target max per-edge nuisance from W*^perp part before global scaling
cfg.nuisLeakLatentBound = 0.05;   % bounded leakage coordinates inside W*
cfg.nuisLeakAmp      = 1.0;

rng(cfg.seed, 'twister');

% ----------------------------- Build problem -----------------------------
edge = build_grid_edge_maps(cfg.g);
d = edge.d;
L = 2 * (cfg.g - 1);

[gadgetInfo, corridorEdges, Ustar] = build_diagonal_corridor(edge, cfg.dstar_target);
rstar = size(Ustar, 2);

lbC = (cfg.highBase - cfg.radOut) * ones(d,1);
ubC = (cfg.highBase + cfg.radOut) * ones(d,1);
lbC(corridorEdges) = cfg.lowBase - cfg.radCorr;
ubC(corridorEdges) = cfg.lowBase + cfg.radCorr;
cBase = 0.5 * (lbC + ubC);
affdimC = sum(ubC > lbC + 1e-12);

qNuis = min(cfg.nuisRank, d - rstar);
Vnuis = build_nuisance_basis(d, qNuis, Ustar);

fprintf('=== Version 3.2c: break implicit compression / harder mixed-nuisance test ===\n');
fprintf('Grid size g = %d, #edges d = %d, path length L = %d\n', cfg.g, d, L);
fprintf('affdim(C) = %d (should equal d = %d)\n', affdimC, d);
fprintf('True planted W* dimension d_* = %d\n', rstar);
fprintf('Corridor edges = %d / %d\n', numel(corridorEdges), d);
verify_domination_sufficient_condition(cfg);

nN = numel(cfg.trainSizes);
riskFull         = zeros(cfg.nTrial, nN);
riskLearn        = zeros(cfg.nTrial, nN);
riskOracle       = zeros(cfg.nTrial, nN);

riskFullSLO      = zeros(cfg.nTrial, nN);
riskLearnSLO     = zeros(cfg.nTrial, nN);
riskOracleSLO    = zeros(cfg.nTrial, nN);

dimW       = zeros(cfg.nTrial, nN);
capTrue    = zeros(cfg.nTrial, nN);
capHat     = zeros(cfg.nTrial, nN);
maxAngDeg  = zeros(cfg.nTrial, nN);
numHard    = zeros(cfg.nTrial, nN);

for ii = 1:nN
    nTrain = cfg.trainSizes(ii);
    fprintf('\n--- nTrain = %d ---\n', nTrain);

    for tr = 1:cfg.nTrial
        rng(cfg.seed + 100*ii + tr, 'twister');

        Arel  = make_bounded_linear_map(rstar, cfg.p, cfg.relLatentBound);
        Aleak = make_bounded_linear_map(rstar, cfg.p, cfg.nuisLeakLatentBound);

        [Mmean, meanInfo] = build_mixed_mean_map(cBase, lbC, ubC, Ustar, Vnuis, Arel, Aleak, corridorEdges, cfg);

        [Xtr, Ctr] = sample_corridor_costs_from_mean_map( ...
            nTrain, cfg.p, cBase, Mmean, cfg.noiseCorr, cfg.noiseOut, lbC, ubC, corridorEdges);
        [Xte, Cte] = sample_corridor_costs_from_mean_map( ...
            cfg.nTest, cfg.p, cBase, Mmean, cfg.noiseCorr, cfg.noiseOut, lbC, ubC, corridorEdges);

        % ---------------- Stage I ----------------------------------------
        nReg = max(cfg.p + 5, floor(cfg.regFrac * nTrain));
        nDisc = max(cfg.p + 5, nTrain - nReg);

        Xreg = Xtr(1:nReg, :);
        Creg = Ctr(1:nReg, :);

        [Ulearn, stage1] = stage1_contextual_basis_alg24_like( ...
            Xreg, Creg, nDisc, cBase, lbC, ubC, edge, gadgetInfo, cfg);
        stats = compare_subspaces(Ulearn, Ustar);

        % ---------------- Stage II: SPO+ ---------------------------------
        Bfull = train_spoplus_full_centered_affine_dp_random_init( ...
            Xtr, Ctr, edge, cBase, cfg.numEpochsStage2, cfg.lrStage2, ...
            cfg.batchSize, cfg.gradClip, cfg.fullSpoInitStd);

        Glearn = train_spoplus_reduced_centered_affine_dp( ...
            Xtr, Ctr, edge, cBase, Ulearn, cfg.numEpochsStage2, ...
            cfg.lrStage2, cfg.batchSize, cfg.gradClip);

        Goracle = train_spoplus_reduced_centered_affine_dp( ...
            Xtr, Ctr, edge, cBase, Ustar, cfg.numEpochsStage2, ...
            cfg.lrStage2, cfg.batchSize, cfg.gradClip);

        % ---------------- Stage II: SLO ----------------------------------
        BfullSLO = fit_slo_full_centered_affine_ridge(Xtr, Ctr, cBase, cfg.sloRidge);
        GlearnSLO = fit_slo_reduced_centered_affine_ridge(Xtr, Ctr, cBase, Ulearn, cfg.sloRidge);
        GoracleSLO = fit_slo_reduced_centered_affine_ridge(Xtr, Ctr, cBase, Ustar, cfg.sloRidge);

        % ---------------- Evaluation -------------------------------------
        riskFull(tr,ii)      = mean_spo_risk_full_centered_affine(Bfull, Xte, Cte, edge, cBase);
        riskLearn(tr,ii)     = mean_spo_risk_reduced_centered_affine(Glearn, Ulearn, Xte, Cte, edge, cBase);
        riskOracle(tr,ii)    = mean_spo_risk_reduced_centered_affine(Goracle, Ustar, Xte, Cte, edge, cBase);

        riskFullSLO(tr,ii)   = mean_spo_risk_full_centered_affine(BfullSLO, Xte, Cte, edge, cBase);
        riskLearnSLO(tr,ii)  = mean_spo_risk_reduced_centered_affine(GlearnSLO, Ulearn, Xte, Cte, edge, cBase);
        riskOracleSLO(tr,ii) = mean_spo_risk_reduced_centered_affine(GoracleSLO, Ustar, Xte, Cte, edge, cBase);

        dimW(tr,ii)      = size(Ulearn, 2);
        capTrue(tr,ii)   = stats.captureTrueInHat;
        capHat(tr,ii)    = stats.captureHatInTrue;
        maxAngDeg(tr,ii) = stats.maxAngleDeg;
        numHard(tr,ii)   = stage1.numHard;

        fprintf(['trial %2d/%2d | dim(W)=%2d | hard=%2d | captureTrue=%.3f ' ...
                 '| maxAngle=%.2f deg | meanScale=%.3f | risk SPO full/learn/oracle = %.4g / %.4g / %.4g ' ...
                 '| risk SLO full/learn/oracle = %.4g / %.4g / %.4g\n'], ...
            tr, cfg.nTrial, dimW(tr,ii), numHard(tr,ii), capTrue(tr,ii), maxAngDeg(tr,ii), meanInfo.scale, ...
            riskFull(tr,ii), riskLearn(tr,ii), riskOracle(tr,ii), ...
            riskFullSLO(tr,ii), riskLearnSLO(tr,ii), riskOracleSLO(tr,ii));
    end
end

% ----------------------------- Summaries ---------------------------------
xAxis = cfg.trainSizes;
[mF,  ciF]  = mean_ci90(log10(riskFull      + 1e-12));
[mL,  ciL]  = mean_ci90(log10(riskLearn     + 1e-12));
[mO,  ciO]  = mean_ci90(log10(riskOracle    + 1e-12));
[mFS, ciFS] = mean_ci90(log10(riskFullSLO   + 1e-12));
[mLS, ciLS] = mean_ci90(log10(riskLearnSLO  + 1e-12));
[mOS, ciOS] = mean_ci90(log10(riskOracleSLO + 1e-12));
[mD,  ciD]  = mean_ci90(dimW);
[mC,  ciC]  = mean_ci90(capTrue);

fig1 = figure('Name', 'Version 3.2c: SPO risk');
hold on; grid on; box on;
errorbar(xAxis, mF, ciF, 'LineWidth', 1.2);
errorbar(xAxis, mL, ciL, 'LineWidth', 1.2);
errorbar(xAxis, mO, ciO, 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('log10(Test SPO risk)');
legend({'Full SPO+ (random ambient init)', 'Reduced SPO+ after learned W', 'Reduced SPO+ with oracle W*'}, 'Location', 'best');
title(sprintf('Version 3.2c SPO, g=%d, d=%d, d_*=%d', cfg.g, d, rstar));

fig2 = figure('Name', 'Version 3.2c: SLO risk');
hold on; grid on; box on;
errorbar(xAxis, mFS, ciFS, 'LineWidth', 1.2);
errorbar(xAxis, mLS, ciLS, 'LineWidth', 1.2);
errorbar(xAxis, mOS, ciOS, 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('log10(Test SPO risk)');
legend({'Full SLO', 'Reduced SLO after learned W', 'Reduced SLO with oracle W*'}, 'Location', 'best');
title(sprintf('Version 3.2c SLO, mixed nuisance, g=%d, d=%d, d_*=%d', cfg.g, d, rstar));

fig3 = figure('Name', 'Version 3.2c: Stage I alignment');
hold on; grid on; box on;
errorbar(xAxis, mD, ciD, 'LineWidth', 1.2);
yline(rstar, '--', 'LineWidth', 1.2);
errorbar(xAxis, mC, ciC, 'LineWidth', 1.2);
xlabel('# labeled training samples');
legend({'Mean learned dim(W)', 'True d_*', 'captureTrueInHat'}, 'Location', 'best');
title('Version 3.2c Stage I diagnostics');

resultsDir = prepare_results_dir_version3p2c();
save_figure(fig1, fullfile(resultsDir, 'version3p2c_spo_risk.png'));
save_figure(fig2, fullfile(resultsDir, 'version3p2c_slo_risk.png'));
save_figure(fig3, fullfile(resultsDir, 'version3p2c_stage1.png'));
save(fullfile(resultsDir, 'version3p2c_summary.mat'), ...
    'cfg', 'riskFull', 'riskLearn', 'riskOracle', ...
    'riskFullSLO', 'riskLearnSLO', 'riskOracleSLO', ...
    'dimW', 'capTrue', 'capHat', 'maxAngDeg', 'numHard', ...
    'mF', 'ciF', 'mL', 'ciL', 'mO', 'ciO', 'mFS', 'ciFS', 'mLS', 'ciLS', 'mOS', 'ciOS', ...
    'mD', 'ciD', 'mC', 'ciC', 'Ustar', 'corridorEdges', 'gadgetInfo', 'cBase');

fprintf('Saved results to %s\n', resultsDir);
end

%==========================================================================
% Stage I: same as before
%==========================================================================

function [U, info] = stage1_contextual_basis_alg24_like(Xreg, Creg, nDisc, cBase, lbC, ubC, edge, gadgetInfo, cfg)
[nReg, p] = size(Xreg);
d = size(Creg, 2);

if nReg < p
    U = zeros(d, 0);
    info.D = zeros(d,0);
    info.numHard = 0;
    info.hardSet = zeros(0,1);
    info.dimAfterPseudo = zeros(nDisc,1);
    return;
end

Yreg = Creg - cBase.';
XtX = Xreg.' * Xreg + cfg.ridge * eye(p);
Ahat = (Yreg.' * Xreg) / XtX;   % d x p

Xdisc = 2*rand(nDisc, p) - 1;
Chat = repmat(cBase.', nDisc, 1) + Xdisc * Ahat.';
Chat = min(Chat, repmat(ubC.', nDisc, 1));
Chat = max(Chat, repmat(lbC.', nDisc, 1));

D = zeros(d, 0);
T = zeros(0, 1);
dimAfterPseudo = zeros(nDisc, 1);

for j = 1:nDisc
    cAnchor = Chat(j, :).';
    Dold = D;
    [D, ~] = pointwise_corridor_cutting_plane(cAnchor, D, edge, gadgetInfo, lbC, ubC, cfg);

    if size(D,2) > size(Dold,2)
        T(end+1, 1) = j; %#ok<AGROW>
    end

    dimAfterPseudo(j) = size(D, 2);
end

U = orth(D);
info.D = D;
info.numHard = numel(T);
info.hardSet = T;
info.dimAfterPseudo = dimAfterPseudo;
info.Ahat = Ahat;
end

function [D, trace] = pointwise_corridor_cutting_plane(cAnchor, Dinit, edge, gadgetInfo, lbC, ubC, cfg)
D = Dinit;
trace.addedIdx = zeros(0,1);
trace.alpha = zeros(0,1);
trace.mmin = zeros(0,1);

for it = 1:cfg.maxPointwiseIters
    candidates = enumerate_corridor_witness_directions(cAnchor, edge, gadgetInfo);

    violated = false(numel(candidates), 1);
    alphaVals = inf(numel(candidates), 1);
    mVals = inf(numel(candidates), 1);
    qVals = cell(numel(candidates), 1);

    for kk = 1:numel(candidates)
        q = candidates(kk).q;
        qVals{kk} = q;

        [mVal, ~] = min_linear_over_box_fiber(q, D, cAnchor, lbC, ubC);
        cinVal = q.' * cAnchor;

        mVals(kk) = mVal;
        if mVal < -cfg.fiTol
            violated(kk) = true;
            denom = cinVal - mVal;
            if denom <= 1e-14
                alphaVals(kk) = 0;
            else
                alphaVals(kk) = cinVal / denom;
            end
        end
    end

    if ~any(violated)
        break;
    end

    ids = find(violated);
    [~, loc] = min(alphaVals(ids));
    pick = ids(loc);

    [D, wasAdded] = append_direction_if_new(D, qVals{pick}, cfg.indepTol);
    trace.addedIdx(end+1,1) = candidates(pick).gadgetIndex; %#ok<AGROW>
    trace.alpha(end+1,1)    = alphaVals(pick); %#ok<AGROW>
    trace.mmin(end+1,1)     = mVals(pick); %#ok<AGROW>

    if ~wasAdded
        break;
    end
end
end

function candidates = enumerate_corridor_witness_directions(cAnchor, edge, gadgetInfo)
[~, w] = oracle_monotone_path_dp(cAnchor, edge);
m = gadgetInfo.numGadgets;
candidates = repmat(struct('gadgetIndex', 0, 'q', zeros(edge.d,1)), m, 1);

for k = 1:m
    i = gadgetInfo.squareTL(k, 1);
    j = gadgetInfo.squareTL(k, 2);

    qbase = zeros(edge.d, 1);
    qbase(edge.h(i,   j)) =  1;
    qbase(edge.v(i, j+1)) =  1;
    qbase(edge.v(i,   j)) = -1;
    qbase(edge.h(i+1, j)) = -1;

    usesRD = (full(w(edge.h(i,   j))) > 0.5) && (full(w(edge.v(i, j+1))) > 0.5);
    usesDR = (full(w(edge.v(i,   j))) > 0.5) && (full(w(edge.h(i+1, j))) > 0.5);

    if usesDR
        q = qbase;
    elseif usesRD
        q = -qbase;
    else
        error('enumerate_corridor_witness_directions: invalid gadget traversal at anchor path.');
    end

    candidates(k).gadgetIndex = k;
    candidates(k).q = q;
end
end

function [mVal, cOut] = min_linear_over_box_fiber(q, D, cAnchor, lbC, ubC)
if exist('linprog', 'file') == 2
    if isempty(D)
        Aeq = [];
        beq = [];
    else
        Aeq = D.';
        beq = Aeq * cAnchor;
    end

    f = q;
    try
        if exist('optimoptions', 'file') == 2
            opts = optimoptions('linprog', 'Display', 'none');
        else
            opts = optimset('Display', 'off'); %#ok<OPTIMSET>
        end
        [cOut, fval, exitflag] = linprog(f, [], [], Aeq, beq, lbC, ubC, opts); %#ok<ASGLU>
        if exitflag > 0 && ~isempty(cOut)
            mVal = fval;
            return;
        end
    catch
    end
end
[mVal, cOut] = min_linear_over_box_fiber_closed_form(q, D, cAnchor, lbC, ubC);
end

function [mVal, cOut] = min_linear_over_box_fiber_closed_form(q, D, cAnchor, lbC, ubC)
cOut = cAnchor;
if isempty(D)
    coeff = zeros(0,1);
else
    coeff = (D.' * D) \ (D.' * q);
end
res = q - D * coeff;
if norm(res) <= 1e-10 * max(1, norm(q))
    mVal = q.' * cAnchor;
    return;
end
pos = q >  1e-12;
neg = q < -1e-12;
cOut(pos) = lbC(pos);
cOut(neg) = ubC(neg);
mVal = q.' * cOut;
end

function [Dnew, wasAdded] = append_direction_if_new(D, q, tol)
q = q / max(norm(q), 1e-12);
if isempty(D)
    Dnew = q;
    wasAdded = true;
    return;
end
coeff = (D.' * D) \ (D.' * q);
res = q - D * coeff;
if norm(res) <= tol * max(1, norm(q))
    Dnew = D;
    wasAdded = false;
else
    Dnew = [D, res / norm(res)]; %#ok<AGROW>
    wasAdded = true;
end
end

function S = compare_subspaces(Uhat, Utrue)
Uhat = orth(Uhat);
Utrue = orth(Utrue);
S.dimHat = size(Uhat, 2);
S.dimTrue = size(Utrue, 2);
if isempty(Uhat) || isempty(Utrue)
    S.captureTrueInHat = 0;
    S.captureHatInTrue = 0;
    S.maxAngleDeg = 90;
    S.minAngleDeg = 90;
    return;
end
sv = svd(Uhat.' * Utrue);
sv = max(min(sv, 1), -1);
ang = acosd(sv);
S.captureTrueInHat = norm(Uhat.' * Utrue, 'fro')^2 / size(Utrue, 2);
S.captureHatInTrue = norm(Utrue.' * Uhat, 'fro')^2 / size(Uhat, 2);
S.minAngleDeg = min(ang);
S.maxAngleDeg = max(ang);
if size(Uhat,2) ~= size(Utrue,2)
    S.maxAngleDeg = max(S.maxAngleDeg, 90);
end
end

%==========================================================================
% Mean map construction
%==========================================================================

function A = make_bounded_linear_map(r, p, latentBound)
A = randn(r, p);
for i = 1:r
    s = sum(abs(A(i,:)));
    if s > 0
        A(i,:) = (latentBound / s) * A(i,:);
    end
end
end

function V = build_nuisance_basis(d, q, Ustar)
Z = randn(d, q);
if ~isempty(Ustar)
    Z = Z - Ustar * (Ustar.' * Z);
end
V = orth(Z);
if size(V,2) > q
    V = V(:,1:q);
end
end

function [Mmean, info] = build_mixed_mean_map(cBase, lbC, ubC, Ustar, Vnuis, Arel, Aleak, corridorEdges, cfg)
% E[c|x] = cBase + Mrel x + Mperp x + Mleak x
%   Mrel  in W*
%   Mperp in W*^perp (large predictive nuisance)
%   Mleak in W* (small leakage so nuisance matters to decisions under finite samples)

d = numel(cBase);
q = size(Vnuis, 2);

if q > 0
    rowL1 = sum(abs(Vnuis), 2);
    perpLatentBound = cfg.nuisPerpEdgeTarget / max(max(rowL1), 1e-12);
    Aperp = make_bounded_linear_map(q, cfg.p, perpLatentBound);
else
    Aperp = zeros(0, cfg.p);
end

Mrel  = cfg.signalAmp * (Ustar * Arel);
Mperp = Vnuis * Aperp;
Mleak = cfg.nuisLeakAmp * (Ustar * Aleak);
Mraw = Mrel + Mperp + Mleak;

noiseBound = cfg.noiseOut * ones(d,1);
noiseBound(corridorEdges) = cfg.noiseCorr;
headroom = min(ubC - cBase, cBase - lbC) - noiseBound - 1e-3;
rowL1raw = sum(abs(Mraw), 2);
ratio = headroom ./ max(rowL1raw, 1e-12);
scale = min([1; ratio]);
scale = max(min(scale, 1), 1e-6);

Mmean = scale * Mraw;
info.scale = scale;
info.relFro = norm(scale*Mrel, 'fro');
info.perpFro = norm(scale*Mperp, 'fro');
info.leakFro = norm(scale*Mleak, 'fro');
end

function [X, C] = sample_corridor_costs_from_mean_map(n, p, cBase, Mmean, noiseCorr, noiseOut, lbC, ubC, corridorEdges)
d = numel(cBase);
X = 2*rand(n, p) - 1;
Signal = Mmean * X.';

Noise = noiseOut * (2*rand(d, n) - 1);
Noise(corridorEdges, :) = noiseCorr * (2*rand(numel(corridorEdges), n) - 1);

Cmat = repmat(cBase, 1, n) + Signal + Noise;
lbRep = repmat(lbC, 1, n);
ubRep = repmat(ubC, 1, n);
if any(Cmat(:) < lbRep(:) - 1e-10) || any(Cmat(:) > ubRep(:) + 1e-10)
    error('sample_corridor_costs_from_mean_map: generated costs left the box prior C.');
end
C = Cmat.';
end

%==========================================================================
% SPO+ / SLO training
%==========================================================================

function B = train_spoplus_full_centered_affine_dp_random_init(X, C, edge, cRef, numEpochs, lr0, batchSize, gradClip, initStd)
[n, p] = size(X);
d = edge.d;
B = (initStd / sqrt(p + 1)) * randn(d, p+1);

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

function G = train_spoplus_reduced_centered_affine_dp(X, C, edge, cRef, U, numEpochs, lr0, batchSize, gradClip)
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

function B = fit_slo_full_centered_affine_ridge(X, C, cRef, ridge)
[n, p] = size(X);
Phi = [X, ones(n,1)];
K = Phi.' * Phi + ridge * eye(p+1);
Y = C - cRef.';
W = K \ (Phi.' * Y);
B = W.';
end

function G = fit_slo_reduced_centered_affine_ridge(X, C, cRef, U, ridge)
[n, p] = size(X);
r = size(U, 2);
if r == 0
    G = zeros(0, p+1);
    return;
end
Phi = [X, ones(n,1)];
K = Phi.' * Phi + ridge * eye(p+1);
Y = C - cRef.';
S = Y * U;
W = K \ (Phi.' * S);
G = W.';
end

%==========================================================================
% Risk / SPO+ primitives
%==========================================================================

function risk = mean_spo_risk_full_centered_affine(B, X, C, edge, cRef)
n = size(X,1);
tot = 0;
for i = 1:n
    phi = [X(i,:)'; 1];
    c = C(i,:)';
    chat = cRef + B * phi;
    tot = tot + spo_loss_dp(chat, c, edge);
end
risk = tot / n;
end

function risk = mean_spo_risk_reduced_centered_affine(G, U, X, C, edge, cRef)
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
    tot = tot + spo_loss_dp(chat, c, edge);
end
risk = tot / n;
end

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
% Monotone shortest path oracle
%==========================================================================

function [bestCost, w] = oracle_monotone_path_dp(c, edge)
g = edge.g;
h = edge.h;
v = edge.v;
D = inf(g, g);
parent = zeros(g, g, 'uint8');
D(1,1) = 0;
for i = 1:g
    for j = 1:g
        cur = D(i,j);
        if isinf(cur), continue; end
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
i = g; j = g;
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
% Corridor construction and geometry
%==========================================================================

function [info, corridorEdges, Ustar] = build_diagonal_corridor(edge, m)
g = edge.g;
maxPossible = floor((g - 1) / 2);
if m > maxPossible
    error('Requested m=%d gadgets, but at most %d fit on a %dx%d grid.', m, maxPossible, g, g);
end
mask = false(edge.d, 1);
Q = zeros(edge.d, m);
squareTL = zeros(m, 2);
curR = 1; curC = 1;
for k = 1:m
    i = 2*k - 1;
    j = 2*k - 1;
    squareTL(k,:) = [i, j];
    conn = connector_edges(curR, curC, i, j, edge.h, edge.v);
    mask(conn) = true;
    sqEdges = [edge.h(i, j); edge.h(i+1, j); edge.v(i, j); edge.v(i, j+1)];
    mask(sqEdges) = true;
    q = zeros(edge.d, 1);
    q(edge.h(i,   j)) =  1;
    q(edge.v(i, j+1)) =  1;
    q(edge.v(i,   j)) = -1;
    q(edge.h(i+1, j)) = -1;
    Q(:,k) = q / norm(q);
    curR = i + 1;
    curC = j + 1;
end
conn = connector_edges(curR, curC, g, g, edge.h, edge.v);
mask(conn) = true;
corridorEdges = find(mask);
Ustar = Q;
info.squareTL = squareTL;
info.numGadgets = m;
end

function E = connector_edges(r0, c0, r1, c1, h, v)
if r1 < r0 || c1 < c0
    error('connector_edges: end point must dominate start point.');
end
E = zeros(0,1);
r = r0; c = c0;
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
maxCorr = L * (cfg.lowBase + cfg.radCorr);
minOneOutside = (cfg.highBase - cfg.radOut) + (L - 1) * (cfg.lowBase - cfg.radCorr);
margin = minOneOutside - maxCorr;
fprintf('Domination sufficient condition:\n');
fprintf('  max corridor-path cost <= %.4f\n', maxCorr);
fprintf('  min path cost with one outside edge >= %.4f\n', minOneOutside);
fprintf('  domination margin = %.4f\n', margin);
if minOneOutside <= maxCorr
    warning('Outside-edge penalty is not large enough. Increase highBase or reduce radCorr.');
else
    fprintf('  OK: any path using an outside edge is always worse than a pure corridor path.\n');
end
end

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

function [m, ci] = mean_ci90(M)
z = 1.645;
m = mean(M, 1, 'omitnan');
nEff = sum(~isnan(M), 1);
sd = std(M, 0, 1, 'omitnan');
se = sd ./ max(sqrt(nEff), 1);
ci = z * se;
end

function resultsDir = prepare_results_dir_version3p2c()
repoDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(repoDir, 'results_version3p2c');
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
