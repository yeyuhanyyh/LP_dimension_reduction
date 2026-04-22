function lp_shortest_path_all_delta_demo()
%==========================================================================
% lp_shortest_path_all_delta_demo.m
%
% Goal
%   Shortest-path benchmark whose Stage-I learner is much closer to the
%   paper's Algorithm 1 than the earlier corridor-local witness code.
%
% Main difference from the earlier demo
%   For each anchor cost c_in, we:
%     1) solve the shortest-path problem by DP,
%     2) build a canonical spanning-tree basis B that CONTAINS the anchor
%        path edges,
%     3) enumerate ALL delta(B,j) for j in the nonbasis N,
%     4) solve the FI subproblem for every delta(B,j),
%     5) apply the facet-hit rule exactly on this candidate family.
%
% Important caveat
%   The shortest-path flow polytope is degenerate, so this script is still
%   "closer to Algorithm 1" rather than a literal proof-faithful copy of
%   the paper's nondegenerate generic LP routine. What is exact here is the
%   enumeration of all delta(B,j) for ONE canonical optimal basis B.
%==========================================================================

clc; close all;

cfg.seed       = 20260601;
cfg.g          = 30;
cfg.dstar      = 12;
cfg.nTrain     = 40;
cfg.nTest      = 60;
cfg.nTrials    = 10;

cfg.lowBase    = 10;
cfg.radCorr    = 1;
cfg.highBase   = 100;
cfg.radOut     = 1;

cfg.latentBd       = 1.40;
cfg.signalAmp      = 1.00;
cfg.probNumActive  = [0.20, 0.60, 0.20];
cfg.noiseCorr      = 0.02;
cfg.noiseOut       = 0.00;

cfg.fiTol      = 1e-9;
cfg.indepTol   = 1e-8;
cfg.maxPointwiseIters = 128;

rng(cfg.seed, 'twister');

[edge, gadgetInfo, corridorEdges, Ustar, lbC, ubC, cBase, x0, Aeq, beq] = ...
    build_problem_instance(edge_cfg(cfg));
opts = make_linprog_options();
d = edge.d;
m = size(Aeq,1);

fprintf('=== shortest-path all-delta benchmark ===\n');
fprintf('g = %d, d = %d, m = %d, true d* = %d\n', cfg.g, d, m, cfg.dstar);

rankMat = zeros(cfg.nTrials, cfg.nTrain);
finalDim = zeros(cfg.nTrials,1);
numHard = zeros(cfg.nTrials,1);
firstTrace = cell(cfg.nTrials,1);
fullTime = zeros(cfg.nTrials,1);
redTime = zeros(cfg.nTrials,1);
speedup = zeros(cfg.nTrials,1);
objGapMax = zeros(cfg.nTrials,1);

for tr = 1:cfg.nTrials
    rng(cfg.seed + tr, 'twister');

    Ctrain = sample_corridor_costs_sparse(cfg.nTrain, cBase, Ustar, lbC, ubC, corridorEdges, cfg);
    [Dlearn, rankAfterSample, hardSet, trace] = cumulative_learn_sdd_all_delta(...
        Ctrain, Aeq, edge, lbC, ubC, cfg, opts);

    Ulearn = orthonormal_basis(Dlearn);
    rankMat(tr,:) = rankAfterSample(:)';
    finalDim(tr) = size(Ulearn,2);
    numHard(tr) = numel(hardSet);
    firstTrace{tr} = trace;

    Ctest = sample_corridor_costs_sparse(cfg.nTest, cBase, Ustar, lbC, ubC, corridorEdges, cfg);
    tFull = zeros(cfg.nTest,1);
    tRed = zeros(cfg.nTest,1);
    gaps = zeros(cfg.nTest,1);
    for i = 1:cfg.nTest
        c = Ctest(i,:)';
        tic;
        objFull = solve_full_lp_linprog(c, Aeq, beq, d, opts);
        tFull(i) = toc;

        tic;
        objRed = solve_reduced_lp_linprog(c, x0, Ulearn, opts);
        tRed(i) = toc;

        gaps(i) = abs(objFull - objRed);
    end
    fullTime(tr) = mean(tFull);
    redTime(tr) = mean(tRed);
    speedup(tr) = fullTime(tr) / max(redTime(tr),1e-12);
    objGapMax(tr) = max(gaps);

    fprintf(['trial %2d/%2d | final dim = %2d | hard = %2d | full = %.4fs ' ...
             '| red = %.4fs | speedup = %.2fx | max gap = %.2e\n'], ...
             tr, cfg.nTrials, finalDim(tr), numHard(tr), fullTime(tr), redTime(tr), speedup(tr), objGapMax(tr));
end

[meanDim, ciDim] = mean_ci90(rankMat);
[meanFull, ciFull] = mean_ci90(fullTime);
[meanRed, ciRed] = mean_ci90(redTime);
[meanSpeed, ciSpeed] = mean_ci90(speedup);

resultsDir = prepare_results_dir('sp_all_delta_results');

fig1 = figure('Name','Stage I mean learned dimension');
hold on; grid on; box on;
errorbar(1:cfg.nTrain, meanDim, ciDim, 'LineWidth', 1.5);
yline(cfg.dstar, '--', 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('t = dim(\hat W_i)');
title(sprintf('Shortest path Stage I with full delta(B,j) enumeration (mean over %d trials)', cfg.nTrials));
legend({'Mean learned dim(\hat W_i)','True d^*'}, 'Location','southeast');
save_figure(fig1, fullfile(resultsDir,'stage1_mean_dimension_all_delta.png'));

fig2 = figure('Name','First hard sample query progress');
hold on; grid on; box on;
tr0 = find(~cellfun(@(s) isempty(s.rankAfterQuery), firstTrace), 1, 'first');
if ~isempty(tr0)
    plot(1:numel(firstTrace{tr0}.rankAfterQuery), firstTrace{tr0}.rankAfterQuery, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
end
yline(cfg.dstar, '--', 'LineWidth', 1.2);
xlabel('Facet-hit query iteration');
ylabel('Discovered rank');
title('Within the first hard sample (all delta(B,j))');
save_figure(fig2, fullfile(resultsDir,'first_hard_query_progress_all_delta.png'));

fig3 = figure('Name','Runtime full vs reduced');
hold on; grid on; box on;
barData = [meanFull, meanRed];
bar(1:2, barData);
errorbar(1:2, barData, [ciFull, ciRed], '.k', 'LineWidth', 1.2);
set(gca, 'XTick', 1:2, 'XTickLabel', {'Full LP','Reduced LP'});
set(gca, 'YScale', 'log');
ylabel('Average solve time (seconds)');
title('Runtime comparison with learned explicit reduced LP');
save_figure(fig3, fullfile(resultsDir,'runtime_full_vs_reduced_all_delta.png'));

T = table((1:cfg.nTrials)', finalDim, numHard, fullTime, redTime, speedup, objGapMax, ...
    'VariableNames', {'trial','final_dim','num_hard','avg_full_time_s','avg_reduced_time_s','speedup','max_abs_objective_gap'});
writetable(T, fullfile(resultsDir, 'summary_all_delta.csv'));
save(fullfile(resultsDir,'summary_all_delta.mat'), 'cfg', 'rankMat', 'finalDim', 'numHard', 'fullTime', 'redTime', 'speedup', 'objGapMax');

fprintf('Saved all-delta shortest-path results to %s\n', resultsDir);
end

function s = edge_cfg(cfg)
s = cfg;
end

%==========================================================================
% Problem construction and graph helpers
%==========================================================================
function [edge, gadgetInfo, corridorEdges, Ustar, lbC, ubC, cBase, x0, Aeq, beq] = build_problem_instance(cfg)
edge = build_grid_edge_maps(cfg.g);
[gadgetInfo, corridorEdges, Ustar] = build_diagonal_corridor(edge, cfg.dstar);

lbC = (cfg.highBase - cfg.radOut) * ones(edge.d,1);
ubC = (cfg.highBase + cfg.radOut) * ones(edge.d,1);
lbC(corridorEdges) = cfg.lowBase - cfg.radCorr;
ubC(corridorEdges) = cfg.lowBase + cfg.radCorr;
cBase = 0.5 * (lbC + ubC);

verify_domination_sufficient_condition(cfg.g, cfg);

[~, x0] = oracle_monotone_path_dp(cBase, edge);
[AeqFull, beqFull] = build_flow_lp(edge);
Aeq = AeqFull(1:end-1,:);
beq = beqFull(1:end-1);
end

function [Aeq, beq] = build_flow_lp(edge)
g = edge.g;
d = edge.d;
Aeq = zeros(g*g, d);
beq = zeros(g*g,1);
node_id = @(ii,jj) (ii-1)*g + jj;
for e = 1:d
    t = edge.tail(e);
    h = edge.head(e);
    Aeq(t,e) = Aeq(t,e) + 1;
    Aeq(h,e) = Aeq(h,e) - 1;
end
beq(node_id(1,1)) = 1;
beq(node_id(g,g)) = -1;
end

function edge = build_grid_edge_maps(g)
h = zeros(g, g-1);
v = zeros(g-1, g);
tail = [];
head = [];
idx = 1;
node_id = @(ii,jj) (ii-1)*g + jj;
for i = 1:g
    for j = 1:(g-1)
        h(i,j) = idx;
        tail(idx,1) = node_id(i,j); %#ok<AGROW>
        head(idx,1) = node_id(i,j+1); %#ok<AGROW>
        idx = idx + 1;
    end
end
for i = 1:(g-1)
    for j = 1:g
        v(i,j) = idx;
        tail(idx,1) = node_id(i,j); %#ok<AGROW>
        head(idx,1) = node_id(i+1,j); %#ok<AGROW>
        idx = idx + 1;
    end
end
edge.g = g;
edge.h = h;
edge.v = v;
edge.d = idx-1;
edge.tail = tail;
edge.head = head;
end

function [info, corridorEdges, Ustar] = build_diagonal_corridor(edge, m)
g = edge.g;
if m > floor((g-1)/2)
    error('Too many gadgets for the chosen grid size.');
end
mask = false(edge.d,1);
Q = zeros(edge.d,m);
squareTL = zeros(m,2);
curR = 1; curC = 1;
for k = 1:m
    i = 2*k - 1; j = 2*k - 1;
    squareTL(k,:) = [i,j];
    conn = connector_edges(curR, curC, i, j, edge.h, edge.v);
    mask(conn) = true;
    sqEdges = [edge.h(i,j); edge.h(i+1,j); edge.v(i,j); edge.v(i,j+1)];
    mask(sqEdges) = true;
    q = zeros(edge.d,1);
    q(edge.h(i,j)) = 1;
    q(edge.v(i,j+1)) = 1;
    q(edge.v(i,j)) = -1;
    q(edge.h(i+1,j)) = -1;
    Q(:,k) = q / norm(q);
    curR = i+1; curC = j+1;
end
conn = connector_edges(curR, curC, g, g, edge.h, edge.v);
mask(conn) = true;
corridorEdges = find(mask);
Ustar = Q;
info.squareTL = squareTL;
info.numGadgets = m;
end

function E = connector_edges(r0,c0,r1,c1,h,v)
if r1 < r0 || c1 < c0, error('connector_edges: end point must dominate start point.'); end
E = zeros(0,1); r = r0; c = c0;
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

function verify_domination_sufficient_condition(g, cfg)
L = 2 * (g - 1);
maxCorr = L * (cfg.lowBase + cfg.radCorr);
minOneOutside = (cfg.highBase - cfg.radOut) + (L - 1) * (cfg.lowBase - cfg.radCorr);
if minOneOutside <= maxCorr
    warning('Outside-edge penalty may be too small for exact corridor dominance.');
end
end

%==========================================================================
% Sparse sampling
%==========================================================================
function C = sample_corridor_costs_sparse(n, cBase, Ustar, lbC, ubC, corridorEdges, cfg)
r = size(Ustar,2);
d = numel(cBase);
Cmat = repmat(cBase, 1, n);
for i = 1:n
    alpha = zeros(r,1);
    s = draw_num_active(cfg.probNumActive);
    if s > 0
        idx = randperm(r, s);
        alpha(idx) = cfg.latentBd * (2*rand(s,1) - 1);
    end
    signal = cfg.signalAmp * (Ustar * alpha);
    noise = cfg.noiseOut * (2*rand(d,1) - 1);
    noise(corridorEdges) = cfg.noiseCorr * (2*rand(numel(corridorEdges),1) - 1);
    c = cBase + signal + noise;
    c = min(c, ubC);
    c = max(c, lbC);
    Cmat(:,i) = c;
end
C = Cmat';
end

function s = draw_num_active(probVec)
u = rand; cut = cumsum(probVec(:));
if u <= cut(1)
    s = 0;
elseif u <= cut(2)
    s = 1;
else
    s = 2;
end
end

%==========================================================================
% Cumulative learner with full delta(B,j) enumeration
%==========================================================================
function [D, rankAfterSample, hardSet, firstTrace] = cumulative_learn_sdd_all_delta(Ctrain, Aeq, edge, lbC, ubC, cfg, opts)
D = zeros(edge.d, 0);
rankAfterSample = zeros(size(Ctrain,1),1);
hardSet = zeros(0,1);
firstTrace.rankAfterQuery = zeros(0,1);
firstTrace.addedNonbasic = zeros(0,1);
seenFirst = false;
for i = 1:size(Ctrain,1)
    cAnchor = Ctrain(i,:)';
    Dold = D;
    [D, trace] = pointwise_shortest_path_all_delta(cAnchor, D, Aeq, edge, lbC, ubC, cfg, opts);
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

function [D, trace] = pointwise_shortest_path_all_delta(cAnchor, Dinit, Aeq, edge, lbC, ubC, cfg, opts)
D = Dinit;
trace.rankAfterQuery = zeros(0,1);
trace.addedNonbasic = zeros(0,1);
trace.numCandidates = zeros(0,1);
for it = 1:cfg.maxPointwiseIters
    candidates = enumerate_all_delta_shortest_path(cAnchor, Aeq, edge);
    qVals = candidates.qVals;
    nonbasicIdx = candidates.nonbasicIdx;
    nCand = size(qVals,2);
    trace.numCandidates(end+1,1) = nCand; %#ok<AGROW>
    violated = false(nCand,1);
    alphaVals = inf(nCand,1);
    for kk = 1:nCand
        q = qVals(:,kk);
        [mVal, ~] = min_linear_over_box_fiber(q, D, cAnchor, lbC, ubC, opts);
        cinVal = q' * cAnchor;
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
    [D, wasAdded] = append_direction_if_new(D, qVals(:,pick), cfg.indepTol);
    if wasAdded
        trace.rankAfterQuery(end+1,1) = size(D,2); %#ok<AGROW>
        trace.addedNonbasic(end+1,1) = nonbasicIdx(pick); %#ok<AGROW>
    else
        break;
    end
end
end

function cand = enumerate_all_delta_shortest_path(cAnchor, Aeq, edge)
[~, w] = oracle_monotone_path_dp(cAnchor, edge);
B = canonical_tree_basis_from_path(w, edge);
allIdx = (1:edge.d)';
mask = true(edge.d,1); mask(B)=false; N = allIdx(mask);
AB = Aeq(:, B);
AN = Aeq(:, N);
coeff = AB \ AN;              % columns are A_B^{-1} A_j
qVals = zeros(edge.d, numel(N));
for k = 1:numel(N)
    q = zeros(edge.d,1);
    q(B) = -coeff(:,k);
    q(N(k)) = 1;
    qVals(:,k) = q;
end
cand.qVals = qVals;
cand.nonbasicIdx = N;
cand.basisIdx = B;
end

function B = canonical_tree_basis_from_path(w, edge)
% Build a canonical spanning tree basis containing all anchor-path edges.
% We solve a minimum-spanning-tree problem on the underlying undirected
% graph with weight 0 on path edges and 1 otherwise.
d = edge.d;
nNodes = edge.g * edge.g;
weights = ones(d,1);
weights(w > 0.5) = 0;
ord = [(1:d)', weights];
ord = sortrows(ord, [2 1]);
parent = 1:nNodes;
rankv = zeros(1,nNodes);
B = zeros(nNodes-1,1);
count = 0;
for t = 1:size(ord,1)
    e = ord(t,1);
    u = edge.tail(e); v = edge.head(e);
    ru = uf_find(parent, u);
    rv = uf_find(parent, v);
    if ru ~= rv
        [parent, rankv] = uf_union(parent, rankv, ru, rv);
        count = count + 1;
        B(count) = e;
        if count == nNodes-1
            break;
        end
    end
end
if count ~= nNodes - 1
    error('canonical_tree_basis_from_path: failed to build spanning tree.');
end
B = B(1:count);
end

function r = uf_find(parent, x)
r = x;
while parent(r) ~= r
    r = parent(r);
end
while parent(x) ~= x
    px = parent(x);
    parent(x) = r; %#ok<NASGU>
    x = px;
end
end

function [parent, rankv] = uf_union(parent, rankv, x, y)
if rankv(x) < rankv(y)
    parent(x) = y;
elseif rankv(x) > rankv(y)
    parent(y) = x;
else
    parent(y) = x;
    rankv(x) = rankv(x) + 1;
end
end

%==========================================================================
% FI over the box fiber
%==========================================================================
function [mVal, cOut] = min_linear_over_box_fiber(q, D, cAnchor, lbC, ubC, opts)
if isempty(D)
    Aeq = [];
    beq = [];
else
    Aeq = D.';
    beq = Aeq * cAnchor;
end
try
    [cOut, fval, exitflag] = linprog(q, [], [], Aeq, beq, lbC, ubC, opts); %#ok<ASGLU>
    if exitflag > 0 && ~isempty(cOut)
        mVal = fval;
        return;
    end
catch %#ok<CTCH>
end
% fallback
cOut = cAnchor;
if ~isempty(D)
    coeff = (D' * D) \ (D' * q);
    q = q - D * coeff;
end
pos = q > 1e-12; neg = q < -1e-12;
cOut(pos) = lbC(pos);
cOut(neg) = ubC(neg);
mVal = q' * cOut;
end

%==========================================================================
% LP solvers
%==========================================================================
function obj = solve_full_lp_linprog(c, Aeq, beq, d, opts)
[~, fval, exitflag] = linprog(c(:), [], [], Aeq, beq, zeros(d,1), [], opts);
if exitflag <= 0, error('solve_full_lp_linprog failed.'); end
obj = fval;
end

function obj = solve_reduced_lp_linprog(c, x0, U, opts)
r = size(U,2);
if r == 0
    obj = c' * x0;
    return;
end
f = U' * c;
A = -U;
b = x0;
[z, ~, exitflag] = linprog(f, A, b, [], [], -inf(r,1), inf(r,1), opts);
if exitflag <= 0, error('solve_reduced_lp_linprog failed.'); end
x = x0 + U * z;
obj = c' * x;
end

%==========================================================================
% Shortest-path oracle
%==========================================================================
function [bestCost, w] = oracle_monotone_path_dp(c, edge)
g = edge.g; h = edge.h; v = edge.v;
D = inf(g,g); parent = zeros(g,g,'uint8'); D(1,1)=0;
for i = 1:g
    for j = 1:g
        cur = D(i,j);
        if isinf(cur), continue; end
        if j < g
            cand = cur + c(h(i,j));
            if cand < D(i,j+1)
                D(i,j+1)=cand; parent(i,j+1)=1;
            end
        end
        if i < g
            cand = cur + c(v(i,j));
            if cand < D(i+1,j)
                D(i+1,j)=cand; parent(i+1,j)=2;
            end
        end
    end
end
bestCost = D(g,g);
w = zeros(edge.d,1);
i=g; j=g;
while (i>1)||(j>1)
    if parent(i,j)==1
        e = h(i,j-1); w(e)=1; j=j-1;
    elseif parent(i,j)==2
        e = v(i-1,j); w(e)=1; i=i-1;
    else
        error('oracle_monotone_path_dp reconstruction failed.');
    end
end
end

%==========================================================================
% Utilities
%==========================================================================
function U = orthonormal_basis(D)
if isempty(D), U = zeros(size(D,1),0); else, U = orth(D); end
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
if norm(res) <= tol * max(1,norm(q))
    Dnew = D; wasAdded = false;
else
    Dnew = [D, res / norm(res)]; %#ok<AGROW>
    wasAdded = true;
end
end
