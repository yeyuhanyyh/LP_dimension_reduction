function lp_random_standardform_alg12_demo()
%==========================================================================
% lp_random_standardform_alg12_demo.m
%
% Goal
%   A generic-looking standard-form LP benchmark that is much closer to the
%   paper's Algorithm 1 and 2 than the shortest-path special case.
%
% Polytope
%   X = {x >= 0 : A x = b},  A = [I_m, T],  T > 0.
%   This makes X bounded and (with probability one) nondegenerate.
%
% Prior set
%   C = { c0 + U z : ||z||_2 <= rho },
%   where U is chosen so that only a planted subset of adjacent pivots is
%   reachable. Consequently, d* is meaningful and known exactly.
%
% Stage I
%   Literal generic routine:
%     - solve the LP with linprog,
%     - recover a basis B from the positive support of the optimal BFS,
%     - enumerate all delta(B,j),
%     - solve all FI(delta(B,j); fiber) in closed form on the subspace-ball,
%     - apply facet-hit and warm-start cumulatively across samples.
%==========================================================================

clc; close all;

cfg.seed       = 20260602;
cfg.m          = 480;
cfg.nExtra     = 400;
cfg.dstar      = 15;
cfg.nTrain     = 45;
cfg.nTest      = 80;
cfg.nTrials    = 3;

cfg.fiTol      = 1e-9;
cfg.indepTol   = 1e-8;
cfg.basisTol   = 1e-7;
cfg.maxPointwiseIters = 64;

rng(cfg.seed, 'twister');
[prob, truth] = build_random_lp_instance(cfg);
opts = make_linprog_options();

fprintf('=== random standard-form LP Alg.1/2 demo ===\n');
fprintf('m = %d, d = %d, planted d* = %d, radius rho = %.4f\n', ...
    prob.m, prob.d, truth.dstar, truth.rho);

rankMat = zeros(cfg.nTrials, cfg.nTrain);
finalDim = zeros(cfg.nTrials,1);
numHard = zeros(cfg.nTrials,1);
fullTime = zeros(cfg.nTrials,1);
redTime = zeros(cfg.nTrials,1);
speedup = zeros(cfg.nTrials,1);
objGapMax = zeros(cfg.nTrials,1);
firstTrace = cell(cfg.nTrials,1);

for tr = 1:cfg.nTrials
    rng(cfg.seed + tr, 'twister');
    Ctrain = sample_subspace_ball_costs(cfg.nTrain, prob.c0, truth.Ucost, truth.rho);

    [Dlearn, rankAfterSample, hardSet, trace] = cumulative_learn_generic_alg2(...
        Ctrain, prob, truth, cfg, opts);

    Ulearn = orth(Dlearn);
    rankMat(tr,:) = rankAfterSample(:)';
    finalDim(tr) = size(Ulearn,2);
    numHard(tr) = numel(hardSet);
    firstTrace{tr} = trace;

    Ctest = sample_subspace_ball_costs(cfg.nTest, prob.c0, truth.Ucost, truth.rho);
    tFull = zeros(cfg.nTest,1);
    tRed = zeros(cfg.nTest,1);
    gaps = zeros(cfg.nTest,1);
    for i = 1:cfg.nTest
        c = Ctest(i,:)';
        tic;
        objFull = solve_full_standard_lp(c, prob, opts);
        tFull(i) = toc;

        tic;
        objRed = solve_reduced_standard_lp(c, truth.xBase, Ulearn, opts);
        tRed(i) = toc;

        gaps(i) = abs(objFull - objRed);
    end
    fullTime(tr) = mean(tFull);
    redTime(tr) = mean(tRed);
    speedup(tr) = fullTime(tr) / max(redTime(tr), 1e-12);
    objGapMax(tr) = max(gaps);

    fprintf(['trial %2d/%2d | final dim = %2d | hard = %2d | full = %.4fs ' ...
             '| red = %.4fs | speedup = %.2fx | max gap = %.2e\n'], ...
             tr, cfg.nTrials, finalDim(tr), numHard(tr), fullTime(tr), redTime(tr), speedup(tr), objGapMax(tr));
end

[meanDim, ciDim] = mean_ci90(rankMat);
[meanFull, ciFull] = mean_ci90(fullTime);
[meanRed, ciRed] = mean_ci90(redTime);
[meanSpeed, ciSpeed] = mean_ci90(speedup);

resultsDir = prepare_results_dir('random_lp_alg12_results');
fig1 = figure('Name','Stage I mean learned dimension');
hold on; grid on; box on;
errorbar(1:cfg.nTrain, meanDim, ciDim, 'LineWidth', 1.5);
yline(truth.dstar, '--', 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('t = dim(\hat W_i)');
title(sprintf('Random standard-form LP: Stage I mean learned dimension (%d trials)', cfg.nTrials));
legend({'Mean learned dim(\hat W_i)','True d^*'}, 'Location','southeast');
save_figure(fig1, fullfile(resultsDir,'stage1_mean_dimension_random_lp.png'));

fig2 = figure('Name','Runtime full vs reduced');
hold on; grid on; box on;
barData = [meanFull, meanRed];
bar(1:2, barData);
errorbar(1:2, barData, [ciFull, ciRed], '.k', 'LineWidth', 1.2);
set(gca, 'XTick', 1:2, 'XTickLabel', {'Full LP','Reduced LP'});
set(gca, 'YScale', 'log');
ylabel('Average solve time (seconds)');
title('Random standard-form LP: runtime comparison');
save_figure(fig2, fullfile(resultsDir,'runtime_full_vs_reduced_random_lp.png'));

T = table((1:cfg.nTrials)', finalDim, numHard, fullTime, redTime, speedup, objGapMax, ...
    'VariableNames', {'trial','final_dim','num_hard','avg_full_time_s','avg_reduced_time_s','speedup','max_abs_objective_gap'});
writetable(T, fullfile(resultsDir, 'summary_random_lp.csv'));
save(fullfile(resultsDir,'summary_random_lp.mat'), 'cfg', 'prob', 'truth', 'rankMat', 'finalDim', 'numHard', 'fullTime', 'redTime', 'speedup', 'objGapMax');

fprintf('Saved random LP results to %s\n', resultsDir);
end

%==========================================================================
% Problem generation
%==========================================================================
function [prob, truth] = build_random_lp_instance(cfg)
rng(cfg.seed, 'twister');
found = false;
tries = 0;
while ~found && tries < 200
    tries = tries + 1;
    m = cfg.m; nExtra = cfg.nExtra; d = m + nExtra;
    T = 0.3 + rand(m, nExtra);
    b = 0.8 + rand(m,1);
    A = [eye(m), T];
    xBase = [b; zeros(nExtra,1)];

    % B0 = first m columns. Nonbasic candidates are the last nExtra.
    Delta = zeros(d, nExtra);
    for j = 1:nExtra
        q = zeros(d,1);
        q(1:m) = -T(:,j);
        q(m+j) = 1;
        Delta(:,j) = q;
    end

    % Choose a planted set of adjacent pivots with independent delta directions.
    allIdx = randperm(nExtra);
    S = allIdx(1:cfg.dstar);
    if rank(Delta(:,S)) < cfg.dstar
        continue;
    end
    Ucost = orth(Delta(:,S));

    % Choose a center c0 deep in the cone, but with selected facets close.
    cB = 1 + rand(m,1);
    sSmall = 0.12;
    sLarge = 3.00;
    s = sLarge * ones(nExtra,1);
    s(S) = sSmall;
    cN = T' * cB + s;
    c0 = [cB; cN];

    % Radius choice: selected facets are reachable, unselected are not.
    projNorm = sqrt(sum((Ucost' * Delta).^2, 1))';
    selThresh = s(S) ./ max(projNorm(S), 1e-12);
    other = setdiff(1:nExtra, S);
    othThresh = s(other) ./ max(projNorm(other), 1e-12);
    rhoLo = 1.20 * max(selThresh);
    rhoHi = 0.80 * min(othThresh);
    if rhoLo >= rhoHi
        continue;
    end
    rho = 0.5 * (rhoLo + rhoHi);

    prob.m = m;
    prob.d = d;
    prob.A = A;
    prob.b = b;
    prob.c0 = c0;

    truth.S = S(:);
    truth.Ucost = Ucost;
    truth.Ustar = orth(Delta(:,S));
    truth.dstar = size(truth.Ustar,2);
    truth.rho = rho;
    truth.xBase = xBase;
    truth.DeltaSelected = Delta(:,S);
    found = true;
end
if ~found
    error('build_random_lp_instance: failed to construct a separated random LP instance.');
end
end

function C = sample_subspace_ball_costs(n, c0, U, rho)
r = size(U,2);
d = numel(c0);
C = zeros(n,d);
for i = 1:n
    z = randn(r,1);
    nz = norm(z);
    if nz < 1e-12
        z(1) = 1; nz = 1;
    end
    z = z / nz;
    rad = rho * rand()^(1/r);
    c = c0 + U * (rad * z);
    C(i,:) = c';
end
end

%==========================================================================
% Algorithm 2 cumulative learner
%==========================================================================
function [D, rankAfterSample, hardSet, firstTrace] = cumulative_learn_generic_alg2(Ctrain, prob, truth, cfg, opts)
D = zeros(prob.d, 0);
rankAfterSample = zeros(size(Ctrain,1),1);
hardSet = zeros(0,1);
firstTrace.rankAfterQuery = zeros(0,1);
firstTrace.addedNonbasic = zeros(0,1);
seenFirst = false;
for i = 1:size(Ctrain,1)
    cAnchor = Ctrain(i,:)';
    Dold = D;
    [D, trace] = pointwise_generic_alg1(cAnchor, D, prob, truth, cfg, opts);
    rankAfterSample(i) = size(D,2);
    if size(D,2) > size(Dold,2)
        hardSet(end+1,1) = i; %#ok<AGROW>
        if ~seenFirst
            firstTrace = trace;
            seenFirst = true;
        end
    end
end
end

%==========================================================================
% Algorithm 1 generic standard-form routine
%==========================================================================
function [D, trace] = pointwise_generic_alg1(cAnchor, Dinit, prob, truth, cfg, opts)
D = Dinit;
trace.rankAfterQuery = zeros(0,1);
trace.addedNonbasic = zeros(0,1);
for it = 1:cfg.maxPointwiseIters
    [xopt, B] = solve_standard_form_with_basis(cAnchor, prob, cfg, opts);
    N = setdiff((1:prob.d)', B);
    Delta = enumerate_all_delta_generic(prob, B, N);

    violated = false(numel(N),1);
    alphaVals = inf(numel(N),1);
    for k = 1:numel(N)
        q = Delta(:,k);
        [mVal, ~] = min_linear_over_subspace_ball_fiber(q, D, cAnchor, prob.c0, truth.Ucost, truth.rho);
        cinVal = q' * cAnchor;
        if mVal < -cfg.fiTol
            violated(k) = true;
            denom = cinVal - mVal;
            if denom <= 1e-14
                alphaVals(k) = 0;
            else
                alphaVals(k) = cinVal / denom;
            end
        end
    end

    if ~any(violated)
        break;
    end

    ids = find(violated);
    [~, loc] = min(alphaVals(ids));
    pick = ids(loc);
    [D, wasAdded] = append_direction_if_new(D, Delta(:,pick), cfg.indepTol);
    if wasAdded
        trace.rankAfterQuery(end+1,1) = size(D,2); %#ok<AGROW>
        trace.addedNonbasic(end+1,1) = N(pick); %#ok<AGROW>
    else
        break;
    end
end
end

function [xopt, B] = solve_standard_form_with_basis(c, prob, cfg, opts)
[x, ~, exitflag] = linprog(c(:), [], [], prob.A, prob.b, zeros(prob.d,1), [], opts);
if exitflag <= 0
    error('solve_standard_form_with_basis: linprog failed.');
end
xopt = x;
B = find(x > cfg.basisTol);
if numel(B) ~= prob.m
    % Fallback: take the largest m entries and verify invertibility.
    [~, ord] = sort(x, 'descend');
    B = sort(ord(1:prob.m));
end
if rank(prob.A(:,B)) < prob.m
    error('solve_standard_form_with_basis: failed to recover an invertible basis from the optimum.');
end
end

function Delta = enumerate_all_delta_generic(prob, B, N)
AB = prob.A(:,B);
AN = prob.A(:,N);
coeff = AB \ AN;
Delta = zeros(prob.d, numel(N));
for k = 1:numel(N)
    q = zeros(prob.d,1);
    q(B) = -coeff(:,k);
    q(N(k)) = 1;
    Delta(:,k) = q;
end
end

%==========================================================================
% FI over a subspace ball
%==========================================================================
function [mVal, cOut] = min_linear_over_subspace_ball_fiber(q, D, cAnchor, c0, U, rho)
% C = { c0 + U z : ||z||_2 <= rho }, U orthonormal.
% Fiber: D' c = D' cAnchor.
r = size(U,2);
uq = U' * q;
if isempty(D)
    zbar = zeros(r,1);
    Z = eye(r);
else
    M = D' * U;
    rhs = D' * (cAnchor - c0);
    zbar = pinv(M) * rhs;
    if norm(M*zbar - rhs) > 1e-7
        error('min_linear_over_subspace_ball_fiber: inconsistent fiber equations in z-space.');
    end
    Z = null(M);
    if isempty(Z)
        Z = zeros(r,0);
    end
end
rad2 = rho^2 - norm(zbar)^2;
if rad2 < -1e-10
    error('min_linear_over_subspace_ball_fiber: anchor fiber left the prior ball.');
end
rad = sqrt(max(rad2, 0));
if isempty(Z)
    zOut = zbar;
else
    proj = Z' * uq;
    np = norm(proj);
    if np <= 1e-12 || rad <= 1e-12
        zOut = zbar;
    else
        zOut = zbar - rad * Z * (proj / np);
    end
end
cOut = c0 + U * zOut;
mVal = q' * cOut;
end

%==========================================================================
% Full and reduced solves
%==========================================================================
function obj = solve_full_standard_lp(c, prob, opts)
[~, fval, exitflag] = linprog(c(:), [], [], prob.A, prob.b, zeros(prob.d,1), [], opts);
if exitflag <= 0, error('solve_full_standard_lp failed.'); end
obj = fval;
end

function obj = solve_reduced_standard_lp(c, xBase, U, opts)
r = size(U,2);
if r == 0
    obj = c' * xBase;
    return;
end
f = U' * c;
A = -U;
b = xBase;
[z, ~, exitflag] = linprog(f, A, b, [], [], -inf(r,1), inf(r,1), opts);
if exitflag <= 0, error('solve_reduced_standard_lp failed.'); end
x = xBase + U * z;
obj = c' * x;
end

%==========================================================================
% Utilities
%==========================================================================
function [m, ci] = mean_ci90(X)
if isvector(X), X = X(:); end
m = mean(X,1);
if size(X,1)==1, ci = zeros(size(m));
else, ci = 1.645 * std(X,0,1) / sqrt(size(X,1)); end
end

function opts = make_linprog_options()
if exist('optimoptions','file') == 2
    try
        opts = optimoptions('linprog', 'Display','none', 'Algorithm','dual-simplex');
    catch
        opts = optimoptions('linprog', 'Display','none');
    end
else
    opts = optimset('Display','off'); %#ok<OPTIMSET>
end
end

function save_figure(fig, filename)
set(fig, 'Color', 'w');
if exist('exportgraphics','file') == 2
    exportgraphics(fig, filename, 'Resolution', 200);
else
    saveas(fig, filename);
end
end

function resultsDir = prepare_results_dir(prefix)
stamp = datestr(now, 'yyyymmdd_HHMMSS');
resultsDir = fullfile(pwd, [prefix '_' stamp]);
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
end

function [Dnew, wasAdded] = append_direction_if_new(D, q, tol)
q = q / max(norm(q), 1e-12);
if isempty(D)
    Dnew = q; wasAdded = true; return;
end
coeff = (D' * D) \ (D' * q);
res = q - D * coeff;
if norm(res) <= tol * max(1, norm(q))
    Dnew = D; wasAdded = false;
else
    Dnew = [D, res / norm(res)]; %#ok<AGROW>
    wasAdded = true;
end
end
