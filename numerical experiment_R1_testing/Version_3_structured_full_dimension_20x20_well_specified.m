function Version_3_structured_full_dimension_20x20_well_specified()
%==========================================================================
% 20x20 Version 3 (well-specified linear contextual model)
%
% Goal:
%   Keep the Version-2 corridor geometry, but make Stage I closer to the
%   paper's Algorithm 4 + Algorithm 2 pipeline while making the underlying
%   contextual model WELL SPECIFIED for linear predictors.
%
% Key changes relative to Version_2:
%   (1) The data-generating conditional mean is EXACTLY linear:
%           E[c | x] = cBase + Ustar * Atrue * x.
%       No tanh nonlinearity is used.
%   (2) Contexts are sampled from a bounded box x in [-1,1]^p and Atrue is
%       row-normalized so that the signal remains strictly inside the box
%       prior C. Thus C-valuedness does not rely on clipping.
%   (3) Stage I uses:
%           centered OLS -> pseudo-costs -> cumulative warm-start learner.
%       The pointwise learner is specialized to THIS corridor family and
%       checks the planted corridor-local decision-relevant moves.
%
% IMPORTANT:
%   The Stage-I candidate set here is NOT the full set of adjacent vertices
%   of the complete monotone-path polytope. It is the structured candidate
%   family of corridor-local witness directions {q_1,...,q_m}, which equals
%   the true W* for this planted Version-2 instance.
%==========================================================================

clc; close all;

% ----------------------------- Configuration
% -----------------------------,
cfg.seed         = 202603131;
cfg.g            = 20;      % 20x20 node grid
cfg.p            = 50;      % context dimension
cfg.dstar_target = 8;       % true decision-relevant dimension / # gadgets

cfg.trainSizes   = [80 160 320 640 960];
cfg.nTest        = 1000;
cfg.nTrial       = 5;

% Full-dimensional box prior
cfg.lowBase      = 10;
cfg.radCorr      = 1;       % corridor edges in [9,11]
cfg.highBase     = 100;
cfg.radOut       = 1;       % outside edges in [99,101]

% Well-specified contextual linear signal
cfg.signalAmp    = 1.00;
cfg.latentBound  = 1.50;    % deterministic bound on each latent coordinate
cfg.noiseCorr    = 0.05;    % zero-mean bounded noise on corridor edges
cfg.noiseOut     = 0.02;    % zero-mean bounded noise on outside edges

% Stage I (regression + pseudo-cost discovery)
cfg.ridge        = 1e-6;
cfg.fiTol        = 1e-9;
cfg.indepTol     = 1e-8;
cfg.regFrac      = 0.50;
cfg.maxPointwiseIters = cfg.dstar_target + 2;

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

[gadgetInfo, corridorEdges, Ustar] = build_diagonal_corridor(edge, cfg.dstar_target);
rstar = size(Ustar, 2);

lbC = (cfg.highBase - cfg.radOut) * ones(d,1);
ubC = (cfg.highBase + cfg.radOut) * ones(d,1);
lbC(corridorEdges) = cfg.lowBase - cfg.radCorr;
ubC(corridorEdges) = cfg.lowBase + cfg.radCorr;
cBase = 0.5 * (lbC + ubC);
affdimC = sum(ubC > lbC + 1e-12);

fprintf('=== Version 3 (well-specified), 20x20 structured full-dimensional experiment ===\n');
fprintf('Grid size g = %d, #edges d = %d, path length L = %d\n', cfg.g, d, L);
fprintf('affdim(C) = %d (should equal d = %d)\n', affdimC, d);
fprintf('True decision-relevant dimension d_* = %d\n', rstar);
fprintf('Corridor edges = %d / %d\n', numel(corridorEdges), d);
verify_domination_sufficient_condition(cfg);

nN = numel(cfg.trainSizes);
riskFull   = zeros(cfg.nTrial, nN);
riskLearn  = zeros(cfg.nTrial, nN);
riskOracle = zeros(cfg.nTrial, nN);
dimW       = zeros(cfg.nTrial, nN);
capTrue    = zeros(cfg.nTrial, nN);
capHat     = zeros(cfg.nTrial, nN);
maxAngDeg  = zeros(cfg.nTrial, nN);
numHard    = zeros(cfg.nTrial, nN);

% -------------------------------- Trials ---------------------------------
for ii = 1:nN
    nTrain = cfg.trainSizes(ii);
    fprintf('\n--- nTrain = %d ---\n', nTrain);

    for tr = 1:cfg.nTrial
        rng(cfg.seed + 100*ii + tr, 'twister');

        Atrue = make_bounded_linear_map(rstar, cfg.p, cfg.latentBound);

        [Xtr, Ctr] = sample_corridor_costs_linear( ...
            nTrain, cfg.p, cBase, Ustar, Atrue, cfg.signalAmp, ...
            cfg.noiseCorr, cfg.noiseOut, lbC, ubC, corridorEdges);

        [Xte, Cte] = sample_corridor_costs_linear( ...
            cfg.nTest, cfg.p, cBase, Ustar, Atrue, cfg.signalAmp, ...
            cfg.noiseCorr, cfg.noiseOut, lbC, ubC, corridorEdges);

        % ---------------- Stage I: Algorithm-4-like ----------------------
        nReg = max(cfg.p + 5, floor(cfg.regFrac * nTrain));
        nDisc = max(cfg.p + 5, nTrain - nReg);

        Xreg = Xtr(1:nReg, :);
        Creg = Ctr(1:nReg, :);

        [Ulearn, stage1] = stage1_contextual_basis_alg24_like( ...
            Xreg, Creg, nDisc, cBase, lbC, ubC, edge, gadgetInfo, cfg);

        stats = compare_subspaces(Ulearn, Ustar);

        % ---------------- Stage II: full / learned / oracle --------------
        Bfull = train_spoplus_full_dp( ...
            Xtr, Ctr, edge, cBase, cfg.numEpochs, cfg.lrFull, ...
            cfg.batchSize, cfg.gradClip, lbC, ubC);

        Glearn = train_spoplus_reduced_dp( ...
            Xtr, Ctr, edge, cBase, Ulearn, cfg.numEpochs, cfg.lrRed, ...
            cfg.batchSize, cfg.gradClip, lbC, ubC);

        Goracle = train_spoplus_reduced_dp( ...
            Xtr, Ctr, edge, cBase, Ustar, cfg.numEpochs, cfg.lrRed, ...
            cfg.batchSize, cfg.gradClip, lbC, ubC);

        % ---------------- Evaluation -------------------------------------
        riskFull(tr,ii)   = mean_spo_risk_full(Bfull, Xte, Cte, edge, cBase, lbC, ubC);
        riskLearn(tr,ii)  = mean_spo_risk_reduced(Glearn, Ulearn, Xte, Cte, edge, cBase, lbC, ubC);
        riskOracle(tr,ii) = mean_spo_risk_reduced(Goracle, Ustar, Xte, Cte, edge, cBase, lbC, ubC);

        dimW(tr,ii)      = size(Ulearn, 2);
        capTrue(tr,ii)   = stats.captureTrueInHat;
        capHat(tr,ii)    = stats.captureHatInTrue;
        maxAngDeg(tr,ii) = stats.maxAngleDeg;
        numHard(tr,ii)   = stage1.numHard;

        fprintf(['trial %2d/%2d | dim(W)=%2d | hard=%2d | captureTrue=%.3f ' ...
                 '| maxAngle=%.2f deg | risk_full=%.4g | risk_learn=%.4g | risk_oracle=%.4g\n'], ...
            tr, cfg.nTrial, dimW(tr,ii), numHard(tr,ii), capTrue(tr,ii), ...
            maxAngDeg(tr,ii), riskFull(tr,ii), riskLearn(tr,ii), riskOracle(tr,ii));
    end
end

% ----------------------------- Summaries ---------------------------------
xAxis = cfg.trainSizes;
[mF, ciF] = mean_ci90(log10(riskFull   + 1e-12));
[mL, ciL] = mean_ci90(log10(riskLearn  + 1e-12));
[mO, ciO] = mean_ci90(log10(riskOracle + 1e-12));
[mD, ciD] = mean_ci90(dimW);
[mC, ciC] = mean_ci90(capTrue);
[mA, ciA] = mean_ci90(maxAngDeg);
[mH, ciH] = mean_ci90(numHard);

paramFull   = d * (cfg.p + 1);
paramOracle = rstar * (cfg.p + 1);
paramLearn  = mean(dimW(:,end), 'all') * (cfg.p + 1);

fprintf('\n=== Version 3 (well-specified) compression summary ===\n');
fprintf('Full linear model params            = d*(p+1) = %d\n', paramFull);
fprintf('Oracle reduced model params         = d_*(p+1) = %d\n', paramOracle);
fprintf('Mean final learned dim(W)           = %.2f\n', mean(dimW(:,end), 'all'));
fprintf('Approx learned reduced params       = %.2f\n', paramLearn);
fprintf('Mean final captureTrueInHat         = %.4f\n', mean(capTrue(:,end), 'all'));
fprintf('Mean final max principal angle (deg)= %.4f\n', mean(maxAngDeg(:,end), 'all'));

% -------------------------------- Plots ----------------------------------
figRisk = figure('Name', 'Version 3 (well-specified): Test SPO risk');
hold on; grid on; box on;
errorbar(xAxis, mF, ciF, 'LineWidth', 1.2);
errorbar(xAxis, mL, ciL, 'LineWidth', 1.2);
errorbar(xAxis, mO, ciO, 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('log10(Test SPO risk)');
legend({'Full SPO+', 'Reduced SPO+ after learned W', 'Reduced SPO+ with oracle W*'}, ...
    'Location', 'best');
title(sprintf('Version 3 well-specified, g=%d, d=%d, true d_*=%d, affdim(C)=d', cfg.g, d, rstar));

figDim = figure('Name', 'Version 3 (well-specified): learned dimension');
hold on; grid on; box on;
errorbar(xAxis, mD, ciD, 'LineWidth', 1.2);
yline(rstar, '--', 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('dim(W)');
legend({'Mean learned dim(W)', 'True d_*'}, 'Location', 'best');
title('Stage I: learned representation dimension');

figAlign = figure('Name', 'Version 3 (well-specified): alignment with true W*');
hold on; grid on; box on;
errorbar(xAxis, mC, ciC, 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('captureTrueInHat');
ylim([0, 1.05]);
title('Stage I: alignment of learned W with true W*');

resultsDir = prepare_results_dir_version3_well_specified();
save_figure(figRisk,  fullfile(resultsDir, 'version3_well_specified_spo_risk.png'));
save_figure(figDim,   fullfile(resultsDir, 'version3_well_specified_dimW.png'));
save_figure(figAlign, fullfile(resultsDir, 'version3_well_specified_alignment.png'));
save(fullfile(resultsDir, 'version3_well_specified_summary.mat'), ...
    'cfg', 'riskFull', 'riskLearn', 'riskOracle', ...
    'dimW', 'capTrue', 'capHat', 'maxAngDeg', 'numHard', ...
    'mF', 'ciF', 'mL', 'ciL', 'mO', 'ciO', ...
    'mD', 'ciD', 'mC', 'ciC', 'mA', 'ciA', 'mH', 'ciH', ...
    'rstar', 'corridorEdges', 'paramFull', 'paramOracle', 'paramLearn', ...
    'affdimC', 'cBase', 'Ustar', 'gadgetInfo');

fprintf('Saved results to %s\n', resultsDir);
end

%==========================================================================
% Stage I: Algorithm-4-like learner from contextual samples
%==========================================================================

function [U, info] = stage1_contextual_basis_alg24_like(Xreg, Creg, nDisc, cBase, lbC, ubC, edge, gadgetInfo, cfg)
% Stage I closer to paper Algorithm 4:
%   - fit centered OLS for mu(x) - cBase,
%   - form pseudo-costs on fresh discovery contexts,
%   - run a cumulative warm-start pointwise routine over pseudo-costs.

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

% Centered OLS: c - cBase ≈ Ahat * x
Yreg = Creg - cBase.';
XtX = Xreg.' * Xreg + cfg.ridge * eye(p);
Ahat = (Yreg.' * Xreg) / XtX;   % d x p

% Fresh discovery contexts and C-valued pseudo-costs
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

%==========================================================================
% Pointwise cutting-plane-like routine on one pseudo-cost
%==========================================================================

function [D, trace] = pointwise_corridor_cutting_plane(cAnchor, Dinit, edge, gadgetInfo, lbC, ubC, cfg)
% Warm-started pointwise routine specialized to the structured corridor
% family. At each iteration we:
%   - keep the current pseudo-cost cAnchor as the anchor c_in,
%   - compute x*(c_in),
%   - enumerate the PLANTED corridor-local candidate directions q_k,
%   - solve FI(q; fiber) for each q = x' - x*(c_in),
%   - add the facet-hit candidate if some minimum is negative.
%
% IMPORTANT:
%   This is NOT a complete enumeration of all adjacent vertices of the full
%   monotone-path polytope. It is a structured candidate family tailored to
%   the planted Version-2 corridor instance, where these directions span W*.

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
% Enumerate the planted decision-relevant corridor-local flip directions.
% These are the q_k directions spanning W* in this structured instance.

[~, w] = oracle_monotone_path_dp(cAnchor, edge);
m = gadgetInfo.numGadgets;
candidates = repmat(struct('gadgetIndex', 0, 'q', zeros(edge.d,1)), m, 1);

for k = 1:m
    i = gadgetInfo.squareTL(k, 1);
    j = gadgetInfo.squareTL(k, 2);

    qbase = zeros(edge.d, 1);
    qbase(edge.h(i,   j)) =  1;   % RD edge 1
    qbase(edge.v(i, j+1)) =  1;   % RD edge 2
    qbase(edge.v(i,   j)) = -1;   % DR edge 1
    qbase(edge.h(i+1, j)) = -1;   % DR edge 2

    usesRD = (full(w(edge.h(i,   j))) > 0.5) && (full(w(edge.v(i, j+1))) > 0.5);
    usesDR = (full(w(edge.v(i,   j))) > 0.5) && (full(w(edge.h(i+1, j))) > 0.5);

    if usesDR
        q = qbase;    % q = x_RD - x_DR
    elseif usesRD
        q = -qbase;   % q = x_DR - x_RD
    else
        error('enumerate_corridor_witness_directions: anchor path is not using a valid gadget traversal.');
    end

    candidates(k).gadgetIndex = k;
    candidates(k).q = q;
end
end

%==========================================================================
% Face-intersection subproblem on the box fiber
%==========================================================================

function [mVal, cOut] = min_linear_over_box_fiber(q, D, cAnchor, lbC, ubC)
% Solve min q' c' s.t. lbC <= c' <= ubC and D' c' = D' cAnchor.
% Use linprog when available; otherwise use a closed-form fallback that is
% valid for the current disjoint-gadget construction.

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
    catch %#ok<CTCH>
        % Fall back below.
    end
end

[mVal, cOut] = min_linear_over_box_fiber_closed_form(q, D, cAnchor, lbC, ubC);
end

function [mVal, cOut] = min_linear_over_box_fiber_closed_form(q, D, cAnchor, lbC, ubC)
% Closed-form fallback for the Version-3 structured corridor family.
% Because all discovered directions are gadget-local and mutually disjoint,
% every candidate q is either already in span(D) or orthogonal to span(D).

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

%==========================================================================
% Subspace comparison
%==========================================================================

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
% Data generation: WELL SPECIFIED LINEAR MODEL
%==========================================================================

function A = make_bounded_linear_map(r, p, latentBound)
% Construct a full-row-rank linear map whose rows have l1 norm latentBound.
% For x in [-1,1]^p, each latent coordinate then satisfies |(A x)_k| <= latentBound.

A = randn(r, p);
for k = 1:r
    s = sum(abs(A(k,:)));
    if s < 1e-12
        A(k,1) = 1;
        s = 1;
    end
    A(k,:) = (latentBound / s) * A(k,:);
end
end

function [X, C] = sample_corridor_costs_linear(n, p, cBase, Ustar, Atrue, signalAmp, noiseCorr, noiseOut, lbC, ubC, corridorEdges)
% WELL SPECIFIED contextual model:
%   X ~ Uniform([-1,1]^p),
%   c = cBase + signalAmp * Ustar * Atrue * X + noise,
%   E[noise | X] = 0,
% so that E[c | X] = cBase + signalAmp * Ustar * Atrue * X is exactly linear.
%
% Because X is bounded and the rows of Atrue are l1-normalized, all generated
% costs lie strictly inside the box C by construction; clipping is not used.

 d = numel(cBase);
 X = 2*rand(n, p) - 1;              % bounded contexts in [-1,1]^p
 Lat = Atrue * X.';                 % EXACTLY linear in X, bounded coordinatewise
 Signal = signalAmp * (Ustar * Lat);

 Noise = noiseOut * (2*rand(d, n) - 1);
 Noise(corridorEdges, :) = noiseCorr * (2*rand(numel(corridorEdges), n) - 1);

 Cmat = repmat(cBase, 1, n) + Signal + Noise;

 if any(Cmat(:) < min(lbC) - 1e-10) || any(Cmat(:) > max(ubC) + 1e-10)
     error('sample_corridor_costs_linear: generated costs left the box prior C. Decrease latentBound or signalAmp.');
 end

 % coordinate-wise safety check
 lbRep = repmat(lbC, 1, n);
 ubRep = repmat(ubC, 1, n);
 if any(Cmat(:) < lbRep(:) - 1e-10) || any(Cmat(:) > ubRep(:) + 1e-10)
     error('sample_corridor_costs_linear: generated costs left the box prior C.');
 end

 C = Cmat.';
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

function resultsDir = prepare_results_dir_version3_well_specified()
repoDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(repoDir, 'results_version3_well_specified');
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
