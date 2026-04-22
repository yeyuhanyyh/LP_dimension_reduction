
function Version_4_general_random_ball_20x20_contextual()
%==========================================================================
% 20x20 Version 4 (general random-cost instance with ball prior)
%
% Goal:
%   Move away from the planted / corridor-structured Version-3 example and
%   test the pipeline in a less structured, more "real-world" setting.
%
% Main changes relative to Version 3.2:
%   (1) Each edge gets a random INTEGER base cost c0 with large spread.
%   (2) The uncertainty set C is a small Euclidean ball centered at c0.
%   (3) There is NO planted corridor, NO oracle subspace, and NO precomputed d_*.
%   (4) Stage I learns a decision-relevant span directly from observed cost
%       samples using a generic pointwise-sufficiency cutting-plane routine.
%   (5) Stage II compares ambient SPO+ baselines against a reduced SPO+ model
%       restricted to the learned span(W).
%
% Interpretation:
%   The "effective" decision-relevant dimension is induced implicitly by the
%   interaction among the random base costs c0, the ball prior C, and the
%   monotone shortest-path polytope. This makes the instance substantially
%   less structured than Version 3.
%==========================================================================

clc; close all;

% ----------------------------- Configuration -----------------------------
cfg.seed       = 20260314;
cfg.g          = 20;      % 20x20 node grid
cfg.p          = 40;      % context dimension

cfg.trainSizes = [80 160 320];
cfg.nTest      = 1200;
cfg.nTrial     = 3;

% Random integer base costs c0
cfg.baseCostLow  = 20;
cfg.baseCostHigh = 120;

% Ball prior C = {c : ||c - c0||_2 <= radC}
% We pick the radius from local 2x2 flip margins to make C small but nontrivial.
cfg.ballQuantile  = 0.45;
cfg.ballRadiusMin = 50.0;
cfg.ballRadiusMax = 250.0;

% Contextual generation inside the ball (exactly linear conditional mean)
cfg.signalFrac = 0.75;    % fraction of radius used by signal bound
cfg.noiseFrac  = 0.20;    % fraction of radius used by bounded zero-mean noise

% Stage I: generic pointwise-sufficiency learner on observed cost samples
cfg.stage1Frac       = 0.50;
cfg.maxStage1Samples = 60;
cfg.fiTol            = 1e-8;
cfg.indepTol         = 1e-8;
cfg.tightTol         = 1e-9;
cfg.nVerifyStage1    = 40;   % holdout costs used to estimate pointwise failure rate

% Stage II: SPO+
cfg.numEpochsCentered  = 12;  % centered + projected full baseline
cfg.lrFullCentered     = 0.010;
cfg.numEpochsAmbient   = 20;  % ambient affine full baseline
cfg.lrFullAmbient      = 0.003;
cfg.numEpochsRed       = 12;  % reduced + projected
cfg.lrRed              = 0.030;
cfg.numEpochsRedNoProj = 12;  % reduced, no projection
cfg.lrRedNoProj        = 0.030;
cfg.batchSize          = 64;
cfg.gradClip           = 5.0;
cfg.l2reg              = 1e-5;

rng(cfg.seed, 'twister');

% ----------------------------- Build graph / LP --------------------------
edge = build_grid_edge_maps(cfg.g);
flow = build_grid_flow_model(edge);

d = edge.d;
L = 2 * (cfg.g - 1);
numNodes = cfg.g^2;

if rank(flow.Aeq) ~= size(flow.Aeq,1)
    error('Flow equality matrix does not have full row rank.');
end

% ----------------------------- Random base costs -------------------------
c0 = randi([cfg.baseCostLow, cfg.baseCostHigh], d, 1);

localMargins = local_square_flip_margins(c0, edge);
%radC = 0.5 * quantile(localMargins, cfg.ballQuantile);
radC = 0.9 * quantile(localMargins, 0.60);
radC = min(max(radC, cfg.ballRadiusMin), cfg.ballRadiusMax);

cfg.signalBound = cfg.signalFrac * radC;
cfg.noiseRadius = cfg.noiseFrac * radC;

if cfg.signalBound + cfg.noiseRadius >= radC
    error('Need signalBound + noiseRadius < radC.');
end

fprintf('=== Version 4 (general random ball prior), 20x20 full-dimensional experiment ===\n');
fprintf('Grid size g = %d, #edges d = %d, #nodes = %d, path length L = %d\n', ...
    cfg.g, d, numNodes, L);
fprintf('Base costs c0: random integers in [%d, %d]\n', cfg.baseCostLow, cfg.baseCostHigh);
fprintf('Ball radius radC = %.4f (from local flip-margin quantile %.2f)\n', ...
    radC, cfg.ballQuantile);
fprintf('Signal bound inside ball = %.4f | noise radius = %.4f\n', ...
    cfg.signalBound, cfg.noiseRadius);
fprintf('No planted corridor, no oracle U*, no precomputed d_*.\n');

% ----------------------------- Storage ----------------------------------
nN = numel(cfg.trainSizes);

riskFullCentered = zeros(cfg.nTrial, nN);
riskFullAmbient  = zeros(cfg.nTrial, nN);
riskLearn        = zeros(cfg.nTrial, nN);
riskLearnNoProj  = zeros(cfg.nTrial, nN);

dimW       = zeros(cfg.nTrial, nN);
numHard    = zeros(cfg.nTrial, nN);
failStage1 = zeros(cfg.nTrial, nN);

% -------------------------------- Trials --------------------------------
for ii = 1:nN
    nTrain = cfg.trainSizes(ii);
    fprintf('\n--- nTrain = %d ---\n', nTrain);

    for tr = 1:cfg.nTrial
        rng(cfg.seed + 100*ii + tr, 'twister');

        Atrue = make_dense_bounded_linear_map_ball(d, cfg.p, cfg.signalBound);

        [Xtr, Ctr] = sample_general_ball_costs_linear( ...
            nTrain, cfg.p, c0, Atrue, cfg.noiseRadius, radC);

        [Xte, Cte] = sample_general_ball_costs_linear( ...
            cfg.nTest, cfg.p, c0, Atrue, cfg.noiseRadius, radC);

        % ---------------- Stage I: learn span(W) from observed cost samples -----
        nStage1 = min([nTrain, cfg.maxStage1Samples, max(25, floor(cfg.stage1Frac * nTrain))]);
        permStage1 = randperm(nTrain);
        idsStage1 = permStage1(1:nStage1);

        [Ulearn, stage1] = stage1_learn_basis_from_cost_samples( ...
            Ctr(idsStage1, :), edge, flow, c0, radC, cfg);

        % ---------------- Stage II: full / reduced SPO+ -------------------------
        BfullCentered = train_spoplus_full_centered_projected_dp( ...
            Xtr, Ctr, edge, c0, radC, cfg.numEpochsCentered, cfg.lrFullCentered, ...
            cfg.batchSize, cfg.gradClip, cfg.l2reg);

        BfullAmbient = train_spoplus_full_affine_dp( ...
            Xtr, Ctr, edge, cfg.numEpochsAmbient, cfg.lrFullAmbient, ...
            cfg.batchSize, cfg.gradClip, cfg.l2reg);

        Glearn = train_spoplus_reduced_centered_projected_dp( ...
            Xtr, Ctr, edge, c0, Ulearn, radC, cfg.numEpochsRed, cfg.lrRed, ...
            cfg.batchSize, cfg.gradClip, cfg.l2reg);

        GlearnNoProj = train_spoplus_reduced_noproj_dp( ...
            Xtr, Ctr, edge, c0, Ulearn, cfg.numEpochsRedNoProj, cfg.lrRedNoProj, ...
            cfg.batchSize, cfg.gradClip, cfg.l2reg);

        % ---------------- Evaluation --------------------------------------------
        riskFullCentered(tr,ii) = mean_spo_risk_full_centered_projected( ...
            BfullCentered, Xte, Cte, edge, c0, radC);

        riskFullAmbient(tr,ii) = mean_spo_risk_full_affine( ...
            BfullAmbient, Xte, Cte, edge);

        riskLearn(tr,ii) = mean_spo_risk_reduced_centered_projected( ...
            Glearn, Ulearn, Xte, Cte, edge, c0, radC);

        riskLearnNoProj(tr,ii) = mean_spo_risk_reduced_noproj( ...
            GlearnNoProj, Ulearn, Xte, Cte, edge, c0);

        dimW(tr,ii)    = size(Ulearn, 2);
        numHard(tr,ii) = stage1.numHard;

        nVerify = min(cfg.nVerifyStage1, size(Cte,1));
        failStage1(tr,ii) = estimate_pointwise_failure_rate( ...
            Ulearn, Cte(1:nVerify, :), edge, flow, c0, radC, cfg);

        fprintf(['trial %2d/%2d | dim(W)=%2d | hard=%2d | stage1_fail=%.3f ' ...
                 '| risk_fullCentered=%.4g | risk_fullAmbient=%.4g ' ...
                 '| risk_learn=%.4g | risk_learnNoProj=%.4g\n'], ...
            tr, cfg.nTrial, dimW(tr,ii), numHard(tr,ii), failStage1(tr,ii), ...
            riskFullCentered(tr,ii), riskFullAmbient(tr,ii), ...
            riskLearn(tr,ii), riskLearnNoProj(tr,ii));
    end
end

% ----------------------------- Summaries ---------------------------------
xAxis = cfg.trainSizes;
[mFC, ciFC] = mean_ci90(log10(riskFullCentered + 1e-12));
[mFA, ciFA] = mean_ci90(log10(riskFullAmbient  + 1e-12));
[mL,  ciL]  = mean_ci90(log10(riskLearn        + 1e-12));
[mLN, ciLN] = mean_ci90(log10(riskLearnNoProj  + 1e-12));

[mD, ciD]   = mean_ci90(dimW);
[mH, ciH]   = mean_ci90(numHard);
[mF1, ciF1] = mean_ci90(failStage1);

paramFull  = d * (cfg.p + 1);
paramLearn = mean(dimW(:,end), 'all') * (cfg.p + 1);

fprintf('\n=== Version 4 (general random ball prior) compression summary ===\n');
fprintf('Full ambient model params            = d*(p+1) = %d\n', paramFull);
fprintf('Mean final learned dim(W)            = %.2f\n', mean(dimW(:,end), 'all'));
fprintf('Approx learned reduced params        = %.2f\n', paramLearn);
fprintf('Mean final Stage-I holdout fail rate = %.4f\n', mean(failStage1(:,end), 'all'));
fprintf('Mean final #hard Stage-I costs       = %.2f\n', mean(numHard(:,end), 'all'));

% -------------------------------- Plots ----------------------------------
figRisk = figure('Name', 'Version 4: Test SPO risk');
hold on; grid on; box on;
errorbar(xAxis, mFC, ciFC, 'LineWidth', 1.2);
errorbar(xAxis, mFA, ciFA, 'LineWidth', 1.2);
errorbar(xAxis, mL,  ciL,  'LineWidth', 1.2);
errorbar(xAxis, mLN, ciLN, 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('log10(Test SPO risk)');
legend({'Full SPO+ (centered + ball-projected)', ...
        'Full SPO+ (ambient affine, no projection)', ...
        'Reduced SPO+ after learned W (centered + ball-projected)', ...
        'Reduced SPO+ after learned W (centered, no projection)'}, ...
    'Location', 'best');
title(sprintf('Version 4 random-ball prior, g=%d, d=%d, p=%d', cfg.g, d, cfg.p));

figDim = figure('Name', 'Version 4: learned dimension');
hold on; grid on; box on;
errorbar(xAxis, mD, ciD, 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('dim(W)');
title('Stage I: learned representation dimension');

figStage1 = figure('Name', 'Version 4: Stage I pointwise failure estimate');
hold on; grid on; box on;
errorbar(xAxis, mF1, ciF1, 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('estimated pointwise failure rate');
ylim([0, 1.05]);
title('Stage I: holdout pointwise-sufficiency failure estimate');

resultsDir = prepare_results_dir_version4_general_ball();
save_figure(figRisk,   fullfile(resultsDir, 'version4_general_ball_spo_risk.png'));
save_figure(figDim,    fullfile(resultsDir, 'version4_general_ball_dimW.png'));
save_figure(figStage1, fullfile(resultsDir, 'version4_general_ball_stage1_fail.png'));
save(fullfile(resultsDir, 'version4_general_ball_summary.mat'), ...
    'cfg', 'c0', 'radC', 'localMargins', ...
    'riskFullCentered', 'riskFullAmbient', 'riskLearn', 'riskLearnNoProj', ...
    'dimW', 'numHard', 'failStage1', ...
    'mFC', 'ciFC', 'mFA', 'ciFA', 'mL', 'ciL', 'mLN', 'ciLN', ...
    'mD', 'ciD', 'mH', 'ciH', 'mF1', 'ciF1', ...
    'paramFull', 'paramLearn');

fprintf('Saved results to %s\n', resultsDir);
end

%==========================================================================
% Stage I: generic representation learner from observed cost samples
%==========================================================================

function [U, info] = stage1_learn_basis_from_cost_samples(Csamples, edge, flow, c0, radC, cfg)
nS = size(Csamples, 1);
Q = zeros(edge.d, 0);
hardSet = zeros(0,1);
dimAfter = zeros(nS, 1);

for i = 1:nS
    Qold = Q;
    [Q, ~] = pointwise_cutting_plane_ball_dp(Csamples(i,:).', Q, edge, flow, c0, radC, cfg);

    if size(Q,2) > size(Qold,2)
        hardSet(end+1,1) = i; %#ok<AGROW>
    end
    dimAfter(i) = size(Q,2);
end

U = Q;
info.numHard = numel(hardSet);
info.hardSet = hardSet;
info.dimAfterSample = dimAfter;
end

%==========================================================================
% Generic pointwise-sufficiency cutting-plane routine for a Euclidean ball
%==========================================================================

function [Q, trace] = pointwise_cutting_plane_ball_dp(cAnchor, Qinit, edge, flow, c0, radC, cfg)
Q = Qinit;
trace.added = zeros(0,1);
trace.mmin = zeros(0,1);

for it = 1:edge.d
    stats = containment_stats_ball_dp(cAnchor, Q, edge, flow, c0, radC, cfg);

    if stats.mmin >= -cfg.fiTol
        return;
    end

    cinVals = cAnchor.' * stats.Delta;
    coutVals = stats.cout.'  * stats.Delta;

    violated = find(coutVals < -cfg.fiTol);
    if isempty(violated)
        violated = stats.loc;
    end

    cinVals = max(cinVals, 0);
    denom = cinVals(violated) - coutVals(violated);

    alpha = inf(1, numel(violated));
    good = denom > 1e-12;
    alpha(good) = cinVals(violated(good)) ./ denom(good);

    [~, ord] = sort(alpha, 'ascend');

    wasAdded = false;
    for kk = 1:numel(ord)
        qCand = stats.Delta(:, violated(ord(kk)));
        [Qtry, added] = append_direction_if_new(Q, qCand, cfg.indepTol);
        if added
            Q = Qtry;
            trace.added(end+1,1) = violated(ord(kk)); %#ok<AGROW>
            trace.mmin(end+1,1) = stats.mmin; %#ok<AGROW>
            wasAdded = true;
            break;
        end
    end

    if ~wasAdded
        % Numerical stall: in theory the facet-hit direction should be new.
        return;
    end
end
end

function stats = containment_stats_ball_dp(cAnchor, Q, edge, flow, c0, radC, cfg)
[~, w] = oracle_monotone_path_dp(cAnchor, edge);
B = build_optimal_tree_basis(w, cAnchor, edge, flow, cfg.tightTol);

isBasic = false(1, edge.d);
isBasic(B) = true;
N = find(~isBasic);

Ab = flow.Aeq(:, B);
AN = flow.Aeq(:, N);
DeltaB = -Ab \ AN;

Delta = zeros(edge.d, numel(N));
Delta(B, :) = DeltaB;
for kk = 1:numel(N)
    Delta(N(kk), kk) = 1;
end

[cPerp, rho, projDelta, projNorm] = fiber_ball_geometry(cAnchor, Q, c0, radC, Delta);

vals = (cPerp.' * Delta) - rho * projNorm;
[mmin, loc] = min(vals);

if projNorm(loc) > 1e-12
    cout = cPerp - rho * projDelta(:,loc) / projNorm(loc);
else
    cout = cPerp;
end

stats.B = B;
stats.N = N;
stats.Delta = Delta;
stats.w = w;
stats.cPerp = cPerp;
stats.rho = rho;
stats.vals = vals;
stats.mmin = mmin;
stats.loc = loc;
stats.cout = cout;
end

function [cPerp, rho, projDelta, projNorm] = fiber_ball_geometry(cAnchor, Q, c0, radC, Delta)
if isempty(Q)
    cPerp = c0;
    rho = radC;
    projDelta = Delta;
else
    shift = Q.' * (cAnchor - c0);
    cPerp = c0 + Q * shift;
    rho2 = radC^2 - sum(shift.^2);
    rho = sqrt(max(rho2, 0));
    projDelta = Delta - Q * (Q.' * Delta);
end

projNorm = sqrt(sum(projDelta.^2, 1));
end

function [Qnew, wasAdded] = append_direction_if_new(Q, q, tol)
q = full(q(:));
if isempty(Q)
    nq = norm(q);
    if nq <= 1e-14
        Qnew = Q;
        wasAdded = false;
    else
        Qnew = q / nq;
        wasAdded = true;
    end
    return;
end

res = q - Q * (Q.' * q);
nr = norm(res);
if nr <= tol * max(1, norm(q))
    Qnew = Q;
    wasAdded = false;
else
    Qnew = [Q, res / nr]; %#ok<AGROW>
    wasAdded = true;
end
end

function rate = estimate_pointwise_failure_rate(Q, Cbatch, edge, flow, c0, radC, cfg)
n = size(Cbatch,1);
bad = 0;
for i = 1:n
    stats = containment_stats_ball_dp(Cbatch(i,:).', Q, edge, flow, c0, radC, cfg);
    bad = bad + double(stats.mmin < -cfg.fiTol);
end
rate = bad / n;
end

%==========================================================================
% Data generation: general random ball-prior contextual model
%==========================================================================

function A = make_dense_bounded_linear_map_ball(d, p, bound2)
% Create a dense map A such that ||A x||_2 <= bound2 for all x in [-1,1]^p.
% We enforce sum_j ||A(:,j)||_2 <= bound2.
A = randn(d, p);
colNorm = sqrt(sum(A.^2, 1));
colNorm(colNorm < 1e-12) = 1;
A = A ./ repmat(colNorm, d, 1);
A = A * (bound2 / p);
end

function [X, C] = sample_general_ball_costs_linear(n, p, c0, Atrue, noiseRadius, radC)
% Exact linear conditional mean model inside the Euclidean ball:
%   X ~ Uniform([-1,1]^p),
%   c = c0 + Atrue * X + noise,
% where ||Atrue*x||_2 <= signalBound for all x in [-1,1]^p and
% ||noise||_2 <= noiseRadius, so generated costs remain inside the ball.
d = numel(c0);
X = 2*rand(n, p) - 1;

Signal = Atrue * X.';                       % d x n
Noise  = sample_uniform_ball_noise(d, n, noiseRadius);

Cmat = repmat(c0, 1, n) + Signal + Noise;

distToCenter = sqrt(sum((Cmat - repmat(c0,1,n)).^2, 1));
if any(distToCenter > radC + 1e-10)
    error('sample_general_ball_costs_linear: generated cost left the ball prior C.');
end

C = Cmat.';
end

function E = sample_uniform_ball_noise(d, n, radius)
if radius <= 0
    E = zeros(d, n);
    return;
end

G = randn(d, n);
G = G ./ max(sqrt(sum(G.^2, 1)), 1e-12);
r = radius * (rand(1, n).^(1/d));
E = G .* repmat(r, d, 1);
end

%==========================================================================
% Stage II: SPO+ training
%==========================================================================

function B = train_spoplus_full_centered_projected_dp(X, C, edge, cRef, radPred, numEpochs, lr0, batchSize, gradClip, l2reg)
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
            chat = project_to_ball(chat, cRef, radPred);

            subg = spoplus_subgrad_dp(chat, c, edge);
            Grad = Grad + full(subg) * phi.';
        end

        Grad = Grad / numel(ids);
        if l2reg > 0
            Grad = Grad + l2reg * B;
        end

        gnorm = norm(Grad(:), 2);
        if gnorm > gradClip
            Grad = Grad * (gradClip / gnorm);
        end

        B = B - eta * Grad;
    end
end
end

function B = train_spoplus_full_affine_dp(X, C, edge, numEpochs, lr0, batchSize, gradClip, l2reg)
% Ambient affine full baseline with FREE intercept and NO projection.
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

            chat = B * phi;
            subg = spoplus_subgrad_dp(chat, c, edge);
            Grad = Grad + full(subg) * phi.';
        end

        Grad = Grad / numel(ids);
        if l2reg > 0
            Grad = Grad + l2reg * B;
        end

        gnorm = norm(Grad(:), 2);
        if gnorm > gradClip
            Grad = Grad * (gradClip / gnorm);
        end

        B = B - eta * Grad;
    end
end
end

function G = train_spoplus_reduced_centered_projected_dp(X, C, edge, cRef, U, radPred, numEpochs, lr0, batchSize, gradClip, l2reg)
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
            chat = project_to_ball(chat, cRef, radPred);

            subg = spoplus_subgrad_dp(chat, c, edge);
            Grad = Grad + (U.' * full(subg)) * phi.';
        end

        Grad = Grad / numel(ids);
        if l2reg > 0
            Grad = Grad + l2reg * G;
        end

        gnorm = norm(Grad(:), 2);
        if gnorm > gradClip
            Grad = Grad * (gradClip / gnorm);
        end

        G = G - eta * Grad;
    end
end
end

function G = train_spoplus_reduced_noproj_dp(X, C, edge, cRef, U, numEpochs, lr0, batchSize, gradClip, l2reg)
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
        if l2reg > 0
            Grad = Grad + l2reg * G;
        end

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

function risk = mean_spo_risk_full_centered_projected(B, X, C, edge, cRef, radPred)
n = size(X,1);
tot = 0;

for i = 1:n
    phi = [X(i,:)'; 1];
    c = C(i,:)';

    chat = cRef + B * phi;
    chat = project_to_ball(chat, cRef, radPred);

    tot = tot + spo_loss_dp(chat, c, edge);
end

risk = tot / n;
end

function risk = mean_spo_risk_full_affine(B, X, C, edge)
n = size(X,1);
tot = 0;

for i = 1:n
    phi = [X(i,:)'; 1];
    c = C(i,:)';
    chat = B * phi;
    tot = tot + spo_loss_dp(chat, c, edge);
end

risk = tot / n;
end

function risk = mean_spo_risk_reduced_centered_projected(G, U, X, C, edge, cRef, radPred)
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
    chat = project_to_ball(chat, cRef, radPred);

    tot = tot + spo_loss_dp(chat, c, edge);
end

risk = tot / n;
end

function risk = mean_spo_risk_reduced_noproj(G, U, X, C, edge, cRef)
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

function cProj = project_to_ball(c, cRef, radius)
u = c - cRef;
nu = norm(u, 2);
if nu <= radius || nu <= 1e-14
    cProj = c;
else
    cProj = cRef + (radius / nu) * u;
end
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
% Graph / flow model helpers
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

function flow = build_grid_flow_model(edge)
g = edge.g;
d = edge.d;

nodeId = reshape(1:(g*g), g, g);
numNodes = g * g;
source = nodeId(1,1);
sink   = nodeId(g,g);

tail = zeros(d,1);
head = zeros(d,1);

outEdges = cell(numNodes,1);

for i = 1:g
    for j = 1:(g-1)
        e = edge.h(i,j);
        u = nodeId(i,j);
        v = nodeId(i,j+1);
        tail(e) = u;
        head(e) = v;
        outEdges{u}(end+1) = e; %#ok<AGROW>
    end
end

for i = 1:(g-1)
    for j = 1:g
        e = edge.v(i,j);
        u = nodeId(i,j);
        v = nodeId(i+1,j);
        tail(e) = u;
        head(e) = v;
        outEdges{u}(end+1) = e; %#ok<AGROW>
    end
end

Afull = zeros(numNodes, d);
for e = 1:d
    Afull(tail(e), e) = 1;
    Afull(head(e), e) = -1;
end

keepRows = setdiff(1:numNodes, sink);
Aeq = Afull(keepRows, :);

bfull = zeros(numNodes, 1);
bfull(source) = 1;
bfull(sink) = -1;
beq = bfull(keepRows);

flow.nodeId   = nodeId;
flow.numNodes = numNodes;
flow.source   = source;
flow.sink     = sink;
flow.tail     = tail;
flow.head     = head;
flow.outEdges = outEdges;
flow.Aeq      = Aeq;
flow.beq      = beq;
end

function B = build_optimal_tree_basis(w, c, edge, flow, tightTol)
% Build an optimal spanning-tree basis for the shortest-path LP:
% one zero-reduced-cost outgoing arc per non-sink node, while forcing
% the currently selected path arcs to be in the basis.
numNodes = flow.numNodes;
pathOut = zeros(numNodes, 1);

cur = flow.source;
while cur ~= flow.sink
    outs = flow.outEdges{cur};
    sel = outs(full(w(outs)) > 0.5);
    if numel(sel) ~= 1
        error('build_optimal_tree_basis: failed to trace current optimal path.');
    end
    e = sel(1);
    pathOut(cur) = e;
    cur = flow.head(e);
end

distToSink = shortest_distance_to_sink_dp(c, edge, flow);

B = zeros(numNodes - 1, 1);
kk = 0;
for node = 1:numNodes
    if node == flow.sink
        continue;
    end

    if pathOut(node) ~= 0
        e = pathOut(node);
    else
        outs = flow.outEdges{node};
        rc = c(outs) + distToSink(flow.head(outs)) - distToSink(node);
        tight = outs(abs(rc) <= tightTol);
        if isempty(tight)
            [~, loc] = min(rc);
            e = outs(loc);
        else
            e = tight(1);
        end
    end

    kk = kk + 1;
    B(kk) = e;
end

B = unique(B, 'stable');
if numel(B) ~= numNodes - 1
    error('build_optimal_tree_basis: basis does not have the correct size.');
end
end

function distToSink = shortest_distance_to_sink_dp(c, edge, flow)
g = edge.g;
nodeId = flow.nodeId;
dist = inf(g, g);
dist(g,g) = 0;

for i = g:-1:1
    for j = g:-1:1
        if (i == g) && (j == g)
            continue;
        end

        best = inf;
        if j < g
            e = edge.h(i,j);
            best = min(best, c(e) + dist(i, j+1));
        end
        if i < g
            e = edge.v(i,j);
            best = min(best, c(e) + dist(i+1, j));
        end

        dist(i,j) = best;
    end
end

distToSink = zeros(flow.numNodes, 1);
for i = 1:g
    for j = 1:g
        distToSink(nodeId(i,j)) = dist(i,j);
    end
end
end

function margins = local_square_flip_margins(c0, edge)
g = edge.g;
margins = zeros((g-1)^2, 1);
kk = 0;

for i = 1:(g-1)
    for j = 1:(g-1)
        q = zeros(edge.d, 1);
        q(edge.h(i,   j)) =  1;   % R then D
        q(edge.v(i, j+1)) =  1;
        q(edge.v(i,   j)) = -1;   % D then R
        q(edge.h(i+1, j)) = -1;

        kk = kk + 1;
        margins(kk) = abs(c0.' * q);
    end
end
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

function resultsDir = prepare_results_dir_version4_general_ball()
repoDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(repoDir, 'results_version4_general_ball');
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
