function pca_unfriendly_iid_standardform_demo()
%==========================================================================
% PCA-UNFRIENDLY IID STANDARD-FORM LP DEMO
% -------------------------------------------------------------------------
% This fixes the earlier train/test-shift issue. Training and test costs are
% drawn i.i.d. from the SAME distribution.
%
% Key construction:
%   X = { x >= 0 : A x = b }, A=[I,T], T>=0 sparse.
%   There are d* selected nonbasic pivot directions delta_j.
%   For each selected j, the reduced cost c_{m+j} is NEGATIVE only with a
%   small probability pRare; otherwise it is positive. Hence, under finite
%   nTrain, the empirical training optima reveal only a small subset of the
%   relevant directions. PCA only sees these empirical optima and therefore
%   misses many rare directions. Our Algorithm-1/2 learner uses the known
%   prior C through FI and discovers all selected directions even when they
%   are rare in the sample.
%
% No distribution shift:
%   Ctrain and Ctest are both sampled by sample_iid_costs(...).
%
% Main metric:
%   Improvement capture ratio:
%       sum_i [c_i^T x0 - c_i^T x_method]_+ / sum_i [c_i^T x0 - c_i^T x_full]_+.
%   This is more stable than raw relative objective gap when rare events
%   make the full objective improvement intermittent.
%
% Requirements: MATLAB Optimization Toolbox (linprog).
%==========================================================================

clc; close all;

cfg.seed       = 20260420;
cfg.m          = 100;
cfg.nExtra     = 900;
cfg.dstar      = 60;
cfg.Tdensity   = 0.01;
cfg.nTrain     = 40;
cfg.nTest      = 300;
cfg.nTrials    = 10;
cfg.K          = cfg.dstar;

% Same train/test distribution parameters.
cfg.pRare      = 0.004;       % rare activation prob per selected direction
cfg.negLo      = -8.0;        % rare attractive selected reduced costs
cfg.negHi      = -4.0;
cfg.posLo      = 0.5;         % normal unattractive selected reduced costs
cfg.posHi      = 1.2;
cfg.unselLo    = 4.0;         % unselected directions never attractive
cfg.unselHi    = 6.0;

% Algorithm parameters.
cfg.maxAlg1Its = 80;
cfg.fiTol      = 1e-9;
cfg.indepTol   = 1e-9;

if exist('linprog','file') ~= 2
    error('This demo requires MATLAB linprog from Optimization Toolbox.');
end

rng(cfg.seed,'twister');

methodNames = {'Full','OracleUstar','Ours','PCA','ColRandPivot'};
nMethod = numel(methodNames);
rankMat = zeros(cfg.nTrials, cfg.nTrain);
learnDim = zeros(cfg.nTrials,1);
pcaRank = zeros(cfg.nTrials,1);
numObservedSelected = zeros(cfg.nTrials,1);

capture = zeros(nMethod, cfg.nTrials);
relGap = zeros(nMethod, cfg.nTrials);
solveTime = zeros(nMethod, cfg.nTrials, cfg.nTest);

for tr = 1:cfg.nTrials
    rng(cfg.seed + tr, 'twister');
    fam = generate_planted_standardform_family(cfg);
    Ctrain = sample_iid_costs(cfg.nTrain, fam, cfg);
    Ctest  = sample_iid_costs(cfg.nTest,  fam, cfg);

    % Our Algorithm 2: uses prior C via FI, not merely observed optimum variance.
    [Dlearn, info] = alg2_cumulative_planted(Ctrain, fam, cfg);
    Ulearn = Dlearn;  % keep sparse raw pivot directions; do not orthogonalize.
    rankMat(tr,:) = info.rankAfterSample(:)';
    learnDim(tr) = size(Ulearn,2);

    % PCA baseline from empirical training optima.
    XtrainOpt = zeros(cfg.nTrain, fam.d);
    observedSel = false(numel(fam.selected),1);
    for i = 1:cfg.nTrain
        [xOpt, ~] = solve_full_lp(Ctrain(i,:)', fam);
        XtrainOpt(i,:) = xOpt';
        activeExtra = find(xOpt(fam.m+fam.selected) > 1e-8);
        observedSel(activeExtra) = true;
    end
    numObservedSelected(tr) = sum(observedSel);
    [Upca, xpcaAnchor, pcaRank(tr)] = pca_basis_from_training_optima(XtrainOpt, cfg.K);

    % Random pivot baseline.
    randCols = randperm(fam.nExtra, cfg.K);
    Urand = fam.DeltaAll(:, randCols);

    baseVals = zeros(cfg.nTest,1);
    fullVals = zeros(cfg.nTest,1);
    vals = zeros(nMethod, cfg.nTest);

    for i = 1:cfg.nTest
        c = Ctest(i,:)';
        baseVals(i) = c' * fam.x0;

        tic; [~, vals(1,i)] = solve_full_lp(c, fam); solveTime(1,tr,i) = toc;
        fullVals(i) = vals(1,i);

        tic; [~, vals(2,i)] = solve_reduced_lp(c, fam.Ustar, fam.x0, fam); solveTime(2,tr,i) = toc;
        tic; [~, vals(3,i)] = solve_reduced_lp(c, Ulearn, fam.x0, fam); solveTime(3,tr,i) = toc;
        tic; [~, vals(4,i)] = solve_reduced_lp(c, Upca, xpcaAnchor, fam); solveTime(4,tr,i) = toc;
        tic; [~, vals(5,i)] = solve_reduced_lp(c, Urand, fam.x0, fam); solveTime(5,tr,i) = toc;
    end

    fullImprove = max(baseVals - fullVals, 0);
    denom = max(sum(fullImprove), 1e-10);
    for mtd = 1:nMethod
        methodImprove = max(baseVals - vals(mtd,:)', 0);
        capture(mtd,tr) = max(0, min(1, sum(methodImprove) / denom));
        relGap(mtd,tr) = mean(max(0, vals(mtd,:)' - fullVals) ./ max(1, abs(fullVals)));
    end

    fprintf('trial %2d/%2d | learned dim=%2d | PCA rank=%2d | observed selected=%2d/%2d | Ours cap=%.3f | PCA cap=%.3f\n', ...
        tr, cfg.nTrials, learnDim(tr), pcaRank(tr), numObservedSelected(tr), cfg.dstar, capture(3,tr), capture(4,tr));
end

% ------------------------------ Plots -----------------------------------
outDir = prepare_results_dir('pca_unfriendly_iid_results');
[mRank, ciRank] = mean_ci90(rankMat);
fig1 = figure('Color','w'); hold on; box on; grid on;
errorbar(1:cfg.nTrain, mRank, ciRank, 'LineWidth',1.5);
yline(cfg.dstar,'--k','LineWidth',1.2);
xlabel('Training sample index i');
ylabel('t_i = dim(\hat W_i)');
title('IID PCA-unfriendly family: our Stage I discovers prior-relevant directions');
legend({'mean \pm 90% CI','true d^*'}, 'Location','southeast');

meanCap = mean(capture,2);
ciCap = 1.645 * std(capture,0,2) / sqrt(cfg.nTrials);
fig2 = figure('Color','w'); hold on; box on; grid on;
bar(meanCap);
errorbar(1:nMethod, meanCap, ciCap, '.k', 'LineWidth',1.2);
set(gca,'XTick',1:nMethod,'XTickLabel',methodNames,'XTickLabelRotation',25);
ylabel('Improvement capture ratio');
title(sprintf('Same train/test distribution: rare relevant pivots (p=%.4f)', cfg.pRare));
ylim([0,1.05]);

meanGap = mean(relGap,2);
ciGap = 1.645 * std(relGap,0,2) / sqrt(cfg.nTrials);
fig3 = figure('Color','w'); hold on; box on; grid on;
bar(meanGap);
errorbar(1:nMethod, meanGap, ciGap, '.k', 'LineWidth',1.2);
set(gca,'XTick',1:nMethod,'XTickLabel',methodNames,'XTickLabelRotation',25);
ylabel('Mean relative objective gap');
title('Relative gap: PCA misses rare pivot directions unseen in training optima');

meanTime = squeeze(mean(mean(solveTime,3),2));
ciTime = squeeze(1.645 * std(mean(solveTime,3),0,2) / sqrt(cfg.nTrials));
fig4 = figure('Color','w'); hold on; box on; grid on;
bar(meanTime);
errorbar(1:nMethod, meanTime, ciTime, '.k', 'LineWidth',1.2);
set(gca,'YScale','log');
set(gca,'XTick',1:nMethod,'XTickLabel',methodNames,'XTickLabelRotation',25);
ylabel('Mean solve time per test LP (seconds)');
title('Runtime comparison');

fig5 = figure('Color','w'); hold on; box on; grid on;
bar([mean(learnDim), mean(pcaRank), mean(numObservedSelected)]);
set(gca,'XTick',1:3,'XTickLabel',{'learned dim','PCA rank','observed selected'});
ylabel('Dimension / count');
title('Why PCA fails: empirical optima reveal only rare subset');

save_figure(fig1, fullfile(outDir,'stage1_rank_curve.png'));
save_figure(fig2, fullfile(outDir,'improvement_capture_bars.png'));
save_figure(fig3, fullfile(outDir,'relative_gap_bars.png'));
save_figure(fig4, fullfile(outDir,'runtime_bars.png'));
save_figure(fig5, fullfile(outDir,'rank_vs_observed_selected.png'));
save(fullfile(outDir,'summary.mat'), 'cfg', 'methodNames', 'rankMat', 'learnDim', 'pcaRank', ...
    'numObservedSelected', 'capture', 'relGap', 'solveTime');

fprintf('\nSaved IID PCA-unfriendly results to %s\n', outDir);
end

%=========================================================================%
% Family generation and iid costs
%=========================================================================%
function fam = generate_planted_standardform_family(cfg)
m = cfg.m; nExtra = cfg.nExtra; d = m+nExtra;
T = sprand(m, nExtra, cfg.Tdensity);
T = abs(T);
for j = 1:nExtra
    if nnz(T(:,j)) < 3
        rows = randperm(m,3);
        T(rows,j) = 0.05 + rand(3,1);
    end
end
T = T + 1e-3*sprand(m,nExtra,cfg.Tdensity);
T = sparse(T);
A = [speye(m), T];
b = 2.0 + rand(m,1);
x0 = [b; zeros(nExtra,1)];
sel = sort(randperm(nExtra, cfg.dstar));
unsel = setdiff(1:nExtra, sel);
DeltaAll = sparse(d,nExtra);
for j = 1:nExtra
    col = sparse(d,1);
    col(1:m) = -T(:,j);
    col(m+j) = 1;
    DeltaAll(:,j) = col;
end
Ustar = DeltaAll(:,sel);

lbC = zeros(d,1); ubC = zeros(d,1);
lbC(1:m) = 0; ubC(1:m) = 0;
lbC(m+sel) = cfg.negLo; ubC(m+sel) = cfg.posHi;
lbC(m+unsel) = cfg.unselLo; ubC(m+unsel) = cfg.unselHi;

fam.A=A; fam.b=b; fam.T=T; fam.m=m; fam.nExtra=nExtra; fam.d=d; fam.x0=x0;
fam.selected=sel; fam.unselected=unsel; fam.DeltaAll=DeltaAll; fam.Ustar=Ustar;
fam.lbC=lbC; fam.ubC=ubC;
end

function C = sample_iid_costs(n, fam, cfg)
C = zeros(n, fam.d);
for i = 1:n
    c = zeros(fam.d,1);
    % selected pivots: rare attractive negative cost, otherwise positive
    rare = rand(numel(fam.selected),1) < cfg.pRare;
    selVals = cfg.posLo + (cfg.posHi-cfg.posLo)*rand(numel(fam.selected),1);
    selVals(rare) = cfg.negLo + (cfg.negHi-cfg.negLo)*rand(sum(rare),1);
    c(fam.m + fam.selected) = selVals;
    % unselected pivots always unattractive
    c(fam.m + fam.unselected) = cfg.unselLo + (cfg.unselHi-cfg.unselLo)*rand(numel(fam.unselected),1);
    C(i,:) = c';
end
end

%=========================================================================%
% Algorithm 2 / 1 with planted FI under the known prior C
%=========================================================================%
function [Dfinal, info] = alg2_cumulative_planted(Ctrain, fam, cfg)
D = sparse(fam.d,0);
rankAfterSample = zeros(size(Ctrain,1),1);
for i = 1:size(Ctrain,1)
    c = Ctrain(i,:)';
    D = alg1_pointwise_planted(c, D, fam, cfg);
    rankAfterSample(i) = size(D,2);
end
Dfinal = D;
info.rankAfterSample = rankAfterSample;
end

function D = alg1_pointwise_planted(cAnchor, Dinit, fam, cfg)
D = Dinit;
for it = 1:cfg.maxAlg1Its
    violated = false(fam.nExtra,1);
    alphaVals = inf(fam.nExtra,1);
    for j = 1:fam.nExtra
        q = fam.DeltaAll(:,j);
        [mVal, ~] = min_linear_over_planted_fiber(j, q, D, cAnchor, fam);
        cinVal = q' * cAnchor;
        if mVal < -cfg.fiTol
            violated(j) = true;
            denom = cinVal - mVal;
            if denom <= 1e-14
                alphaVals(j) = 0;
            else
                alphaVals(j) = cinVal / denom;
            end
        end
    end
    if ~any(violated), break; end
    ids = find(violated);
    [~,loc] = min(alphaVals(ids));
    pick = ids(loc);
    qPick = fam.DeltaAll(:,pick);
    [D, wasAdded] = append_direction_if_new_raw(D, qPick, cfg.indepTol);
    if ~wasAdded, break; end
end
end

function [mVal, cOut] = min_linear_over_planted_fiber(j, q, D, cAnchor, fam)
if is_in_span(D, q, 1e-9)
    mVal = q' * cAnchor;
    cOut = cAnchor;
else
    idx = fam.m + j;
    cOut = cAnchor;
    cOut(idx) = fam.lbC(idx);
    mVal = fam.lbC(idx);
end
end

function tf = is_in_span(D, q, tol)
if isempty(D), tf = false; return; end
coef = D \ q;
res = q - D*coef;
tf = norm(res) <= tol * max(1,norm(q));
end

function [Dnew, wasAdded] = append_direction_if_new_raw(D, q, tol)
q = q / max(norm(q),1e-12);
if isempty(D)
    Dnew = sparse(q); wasAdded = true; return;
end
coef = D \ q; res = q - D*coef;
if norm(res) <= tol * max(1,norm(q))
    Dnew = D; wasAdded = false;
else
    Dnew = [D, sparse(q)]; %#ok<AGROW>
    wasAdded = true;
end
end

%=========================================================================%
% Solvers and PCA
%=========================================================================%
function [x, val] = solve_full_lp(c, fam)
opts = make_linprog_options();
[x, val, exitflag] = linprog(c, [], [], fam.A, fam.b, zeros(fam.d,1), [], opts); %#ok<ASGLU>
if exitflag <= 0 || isempty(x), error('Full LP solve failed.'); end
end

function [x, val] = solve_reduced_lp(c, U, xAnchor, fam)
if isempty(U), x=xAnchor; val=c'*x; return; end
opts = make_linprog_options();
r = U' * c;
Aineq = -U; bineq = xAnchor;
AU = fam.A * U;
if norm(AU,'fro') <= 1e-8
    Aeq=[]; beq=[];
else
    Aeq=AU; beq=fam.b - fam.A*xAnchor;
end
lbz = -inf(size(U,2),1);
[z,~,exitflag] = linprog(r, Aineq, bineq, Aeq, beq, lbz, [], opts); %#ok<ASGLU>
if exitflag <= 0 || isempty(z), z=zeros(size(U,2),1); end
x = xAnchor + U*z;
val = c'*x;
end

function [U, xbar, effRank] = pca_basis_from_training_optima(Xtrain, K)
xbar = mean(Xtrain,1)';
Xc = Xtrain - mean(Xtrain,1);
[~,S,V] = svd(Xc, 'econ');
sv = diag(S);
if isempty(sv)
    effRank = 0;
else
    effRank = sum(sv > 1e-9 * max(1, sv(1)));
end
r = min([K, effRank, size(V,2)]);
if r==0, U=zeros(size(Xtrain,2),0); else, U=V(:,1:r); end
end

%=========================================================================%
% Helpers
%=========================================================================%
function [m, ci] = mean_ci90(X)
m = mean(X,1);
if size(X,1)==1, ci=zeros(size(m)); else, ci=1.645*std(X,0,1)/sqrt(size(X,1)); end
end

function opts = make_linprog_options()
if exist('optimoptions','file') == 2
    opts = optimoptions('linprog','Display','none','Algorithm','dual-simplex');
else
    opts = optimset('Display','off'); %#ok<OPTIMSET>
end
end

function save_figure(fig, filename)
set(fig,'Color','w');
if exist('exportgraphics','file') == 2
    exportgraphics(fig, filename, 'Resolution', 200);
else
    saveas(fig, filename);
end
end

function outDir = prepare_results_dir(prefix)
stamp = datestr(now,'yyyymmdd_HHMMSS');
outDir = fullfile(pwd, [prefix '_' stamp]);
if ~exist(outDir,'dir'), mkdir(outDir); end
end
