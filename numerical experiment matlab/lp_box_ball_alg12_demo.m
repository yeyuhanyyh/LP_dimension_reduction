function lp_box_ball_alg12_demo()
%==========================================================================
% lp_box_ball_alg12_demo.m
%
% Goal
%   Exact, fully transparent Algorithm 1 / 2 style benchmark on the box LP
%       min c' x   s.t. 0 <= x <= 1.
%
% Why this case is useful
%   - The optimality cones are exactly sign patterns of c.
%   - The candidate directions delta(B,j) are exactly +/- e_i.
%   - With a Euclidean ball prior C = B(c0, rho), d* is EXACTLY the number
%     of coordinates i for which |c0_i| <= rho.
%
% This makes the intrinsic dimension d* analytically checkable.
%==========================================================================

clc; close all;

cfg.seed       = 20260603;
cfg.d          = 200;
cfg.dstar      = 10;
cfg.rho        = 0.40;
cfg.nTrain     = 50;
cfg.nTest      = 100;
cfg.nTrials    = 10;

cfg.fiTol      = 1e-9;
cfg.indepTol   = 1e-8;
cfg.maxPointwiseIters = 128;

rng(cfg.seed, 'twister');
[prob, truth] = build_box_ball_instance(cfg);
opts = make_linprog_options();

fprintf('=== box-ball Alg.1/2 demo ===\n');
fprintf('d = %d, exact d* = %d, rho = %.3f\n', prob.d, truth.dstar, prob.rho);

rankMat = zeros(cfg.nTrials, cfg.nTrain);
finalDim = zeros(cfg.nTrials,1);
numHard = zeros(cfg.nTrials,1);
fullTime = zeros(cfg.nTrials,1);
redTime = zeros(cfg.nTrials,1);
speedup = zeros(cfg.nTrials,1);
objGapMax = zeros(cfg.nTrials,1);

for tr = 1:cfg.nTrials
    rng(cfg.seed + tr, 'twister');
    Ctrain = sample_ball_costs(cfg.nTrain, prob.c0, prob.rho);
    [Dlearn, rankAfterSample, hardSet, ~] = cumulative_learn_box_alg2(Ctrain, prob, cfg);
    Ulearn = orth(Dlearn);
    rankMat(tr,:) = rankAfterSample(:)';
    finalDim(tr) = size(Ulearn,2);
    numHard(tr) = numel(hardSet);

    Ctest = sample_ball_costs(cfg.nTest, prob.c0, prob.rho);
    tFull = zeros(cfg.nTest,1);
    tRed = zeros(cfg.nTest,1);
    gaps = zeros(cfg.nTest,1);
    for i = 1:cfg.nTest
        c = Ctest(i,:)';
        tic; objFull = solve_box_full_linprog(c, prob.d, opts); tFull(i)=toc;
        tic; objRed = solve_box_reduced_linprog(c, truth.xBase, Ulearn, opts); tRed(i)=toc;
        gaps(i) = abs(objFull - objRed);
    end
    fullTime(tr)=mean(tFull); redTime(tr)=mean(tRed);
    speedup(tr)=fullTime(tr)/max(redTime(tr),1e-12);
    objGapMax(tr)=max(gaps);

    fprintf(['trial %2d/%2d | final dim = %2d | hard = %2d | full = %.4fs ' ...
             '| red = %.4fs | speedup = %.2fx | max gap = %.2e\n'], ...
             tr, cfg.nTrials, finalDim(tr), numHard(tr), fullTime(tr), redTime(tr), speedup(tr), objGapMax(tr));
end

[meanDim, ciDim] = mean_ci90(rankMat);
[meanFull, ciFull] = mean_ci90(fullTime);
[meanRed, ciRed] = mean_ci90(redTime);
resultsDir = prepare_results_dir('box_ball_alg12_results');

fig1 = figure('Name','Stage I mean learned dimension');
hold on; grid on; box on;
errorbar(1:cfg.nTrain, meanDim, ciDim, 'LineWidth', 1.5);
yline(truth.dstar, '--', 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('t = dim(\hat W_i)');
title(sprintf('Box LP with exact d*: Stage I mean learned dimension (%d trials)', cfg.nTrials));
legend({'Mean learned dim(\hat W_i)','True d^*'}, 'Location','southeast');
save_figure(fig1, fullfile(resultsDir,'stage1_mean_dimension_box_ball.png'));

fig2 = figure('Name','Runtime full vs reduced');
hold on; grid on; box on;
barData = [meanFull, meanRed];
bar(1:2, barData);
errorbar(1:2, barData, [ciFull, ciRed], '.k', 'LineWidth', 1.2);
set(gca, 'XTick', 1:2, 'XTickLabel', {'Full LP','Reduced LP'});
set(gca, 'YScale', 'log');
ylabel('Average solve time (seconds)');
title('Box LP: runtime comparison');
save_figure(fig2, fullfile(resultsDir,'runtime_full_vs_reduced_box_ball.png'));

T = table((1:cfg.nTrials)', finalDim, numHard, fullTime, redTime, speedup, objGapMax, ...
    'VariableNames', {'trial','final_dim','num_hard','avg_full_time_s','avg_reduced_time_s','speedup','max_abs_objective_gap'});
writetable(T, fullfile(resultsDir, 'summary_box_ball.csv'));
save(fullfile(resultsDir,'summary_box_ball.mat'), 'cfg', 'prob', 'truth', 'rankMat', 'finalDim', 'numHard', 'fullTime', 'redTime', 'speedup', 'objGapMax');

fprintf('Saved box-ball results to %s\n', resultsDir);
end

function [prob, truth] = build_box_ball_instance(cfg)
d = cfg.d; rho = cfg.rho;
S = randperm(d, cfg.dstar);
c0 = zeros(d,1);
for i = 1:d
    if ismember(i, S)
        c0(i) = 0.6 * rho * (2*rand - 1);   % sign can flip within the ball
    else
        if rand < 0.5
            c0(i) = 2.2 * rho + 0.3 * rand;
        else
            c0(i) = -2.2 * rho - 0.3 * rand;
        end
    end
end
xBase = double(c0 < 0);
prob.d = d;
prob.c0 = c0;
prob.rho = rho;
truth.S = sort(S(:));
truth.dstar = numel(S);
truth.Ustar = eye(d); truth.Ustar = truth.Ustar(:, truth.S);
truth.xBase = xBase;
end

function C = sample_ball_costs(n, c0, rho)
d = numel(c0);
C = zeros(n,d);
for i = 1:n
    z = randn(d,1); nz = norm(z);
    if nz < 1e-12, z(1)=1; nz=1; end
    z = z / nz;
    rad = rho * rand()^(1/d);
    C(i,:) = (c0 + rad * z)';
end
end

function [D, rankAfterSample, hardSet, firstTrace] = cumulative_learn_box_alg2(Ctrain, prob, cfg)
D = zeros(prob.d,0);
rankAfterSample = zeros(size(Ctrain,1),1);
hardSet = zeros(0,1);
firstTrace.rankAfterQuery = zeros(0,1);
seenFirst = false;
for i = 1:size(Ctrain,1)
    cAnchor = Ctrain(i,:)';
    Dold = D;
    [D, trace] = pointwise_box_alg1(cAnchor, D, prob, cfg);
    rankAfterSample(i) = size(D,2);
    if size(D,2) > size(Dold,2)
        hardSet(end+1,1) = i; %#ok<AGROW>
        if ~seenFirst
            firstTrace = trace; seenFirst = true;
        end
    end
end
end

function [D, trace] = pointwise_box_alg1(cAnchor, Dinit, prob, cfg)
D = Dinit;
trace.rankAfterQuery = zeros(0,1);
for it = 1:cfg.maxPointwiseIters
    [Q, signsIdx] = enumerate_box_deltas(cAnchor);
    violated = false(prob.d,1);
    alphaVals = inf(prob.d,1);
    for i = 1:prob.d
        q = Q(:,i);
        [mVal, ~] = min_linear_over_ball_fiber(q, D, cAnchor, prob.c0, prob.rho);
        cinVal = q' * cAnchor;
        if mVal < -cfg.fiTol
            violated(i) = true;
            denom = cinVal - mVal;
            if denom <= 1e-14
                alphaVals(i) = 0;
            else
                alphaVals(i) = cinVal / denom;
            end
        end
    end
    if ~any(violated), break; end
    ids = find(violated);
    [~, loc] = min(alphaVals(ids));
    pick = ids(loc);
    [D, wasAdded] = append_direction_if_new(D, Q(:,pick), cfg.indepTol);
    if wasAdded
        trace.rankAfterQuery(end+1,1) = size(D,2); %#ok<AGROW>
    else
        break;
    end
end
end

function [Q, signsIdx] = enumerate_box_deltas(cAnchor)
d = numel(cAnchor);
Q = zeros(d,d);
signsIdx = zeros(d,1);
for i = 1:d
    if cAnchor(i) >= 0
        Q(i,i) = 1;   % current x_i = 0, alternative x_i = 1
        signsIdx(i) = 1;
    else
        Q(i,i) = -1;  % current x_i = 1, alternative x_i = 0
        signsIdx(i) = -1;
    end
end
end

function [mVal, cOut] = min_linear_over_ball_fiber(q, D, cAnchor, c0, rho)
d = numel(c0);
if isempty(D)
    zbar = zeros(d,1);
    Z = eye(d);
else
    rhs = D' * (cAnchor - c0);
    zbar = pinv(D') * rhs;  % least-norm solution of D' z = rhs
    Z = null(D');
    if isempty(Z), Z = zeros(d,0); end
end
rad2 = rho^2 - norm(zbar)^2;
if rad2 < -1e-10
    error('min_linear_over_ball_fiber: anchor fiber left the prior ball.');
end
rad = sqrt(max(rad2,0));
if isempty(Z)
    zOut = zbar;
else
    proj = Z' * q;
    np = norm(proj);
    if np <= 1e-12 || rad <= 1e-12
        zOut = zbar;
    else
        zOut = zbar - rad * Z * (proj / np);
    end
end
cOut = c0 + zOut;
mVal = q' * cOut;
end

function obj = solve_box_full_linprog(c, d, opts)
A = [eye(d); -eye(d)];
b = [ones(d,1); zeros(d,1)];
[~, fval, exitflag] = linprog(c(:), A, b, [], [], [], [], opts);
if exitflag <= 0, error('solve_box_full_linprog failed.'); end
obj = fval;
end

function obj = solve_box_reduced_linprog(c, xBase, U, opts)
r = size(U,2);
if r == 0
    obj = c' * xBase;
    return;
end
f = U' * c;
A = [ U; -U ];
b = [ ones(numel(xBase),1) - xBase; xBase ];
[z, ~, exitflag] = linprog(f, A, b, [], [], -inf(r,1), inf(r,1), opts);
if exitflag <= 0, error('solve_box_reduced_linprog failed.'); end
x = xBase + U*z;
obj = c' * x;
end

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
